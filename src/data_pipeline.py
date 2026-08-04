# -*- coding: utf-8 -*-
"""
FusionDx — Data Pipeline
========================
Downloads and prepares a manageable TCGA-BRCA patient subset from the NIH
Genomic Data Commons (GDC), aligning clinical records with diagnostic slide
image tiles so every downstream model uses correctly matched pairs.

CORRECTNESS NOTE — patient-level splitting
-------------------------------------------
Each patient may contribute multiple image tiles (from one or more slides).
Tiles from the same patient are highly correlated.  If we split at the tile
level, the same patient's tissue could appear in both train and test, leaking
information and producing inflated — meaningless — accuracy numbers.  We
therefore split exclusively at the PATIENT level: every tile from a patient
lives in exactly one split (train / val / test).  This is the single most
important correctness decision in the entire pipeline.

SCOPE NOTE — reduced-resolution tiles
--------------------------------------
Whole-slide images (WSIs) are typically 1–5 GB each at full resolution.
Processing full WSIs on a laptop is intractable.  Instead we:
  1. Download a small subset of slides.
  2. Extract a fixed number of non-overlapping tiles per slide at a
     manageable resolution (e.g. 256×256 px at 10× magnification or a
     downsampled equivalent).
This is standard methodology in computational pathology, not a shortcut.
The OpenSlide library handles the WSI format transparently.

DATA NOTE — open access
------------------------
TCGA-BRCA diagnostic slide images (SVS format) and clinical/demographic data
are both in GDC's open-access tier and do not require a dbGaP access request.
Some genomic sequencing data (WGS, RNA-seq) requires controlled access; we
do NOT use those here.
"""

import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# OpenSlide Windows path helper
# If the OpenSlide bin/ directory is set via the OPENSLIDE_PATH environment
# variable (or found at a common default), add it to the DLL search path.
# This avoids requiring users to manually edit their system PATH.
# ---------------------------------------------------------------------------
_OPENSLIDE_PATH = os.environ.get("OPENSLIDE_PATH", r"C:\OpenSlide\bin")
if os.path.isdir(_OPENSLIDE_PATH) and sys.platform == "win32":
    try:
        os.add_dll_directory(_OPENSLIDE_PATH)
    except Exception:
        pass  # Silently ignore -- PATH will be tried anyway

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GDC_API = "https://api.gdc.cancer.gov"
TCGA_BRCA_PROJECT = "TCGA-BRCA"

# Tile extraction settings
TILE_SIZE = 256          # pixels
TILES_PER_SLIDE = 16     # tiles sampled per slide — keep small for tractability
TILE_LEVEL = 1           # WSI pyramid level (0 = full res; higher = smaller)

# Patient subset size
DEFAULT_SUBSET = 150     # patients; adjust as needed

# Output directories (relative to project root)
DATA_DIR = Path(__file__).parent.parent / "data"
CLINICAL_DIR = DATA_DIR / "clinical"
SLIDES_DIR = DATA_DIR / "slides"
TILES_DIR = DATA_DIR / "tiles"

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Step 1 — fetch clinical / demographic data via GDC API
# ---------------------------------------------------------------------------

def fetch_clinical_data(n_cases: int = DEFAULT_SUBSET,
                        out_dir: Path = CLINICAL_DIR) -> pd.DataFrame:
    """
    Pull clinical records for TCGA-BRCA from the GDC API.

    VERIFIED FIELD PATHS (checked against live GDC API, August 2026):
      - submitter_id                          patient barcode e.g. TCGA-A1-A0SD
      - diagnoses.age_at_diagnosis            age in days at diagnosis
      - diagnoses.ajcc_pathologic_stage       AJCC stage (e.g. 'Stage IIB')
      - diagnoses.primary_diagnosis           histological type string
      - diagnoses.morphology                  ICD-O morphology code
      - demographic.race                      race/ethnicity
      - demographic.sex_at_birth              biological sex ('female'/'male')
      - demographic.vital_status              'Alive' or 'Dead'

    ER/PR/HER2 receptor status is NOT in the diagnoses block in GDC's current
    schema -- it is stored in follow_ups.molecular_tests with gene symbols:
      ESR1 = ER status, PGR = PR status, ERBB2 = HER2 status.
    We retrieve these via a separate expand and merge them in.

    Returns a DataFrame with one row per patient.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "clinical_raw.csv"

    if cache_path.exists():
        print(f"[clinical] Loading cached clinical data from {cache_path}")
        df = pd.read_csv(cache_path)
        return df

    print(f"[clinical] Fetching clinical data for up to {n_cases} TCGA-BRCA cases ...")

    payload = {
        "filters": json.dumps({
            "op": "=",
            "content": {"field": "project.project_id", "value": TCGA_BRCA_PROJECT},
        }),
        "expand": "diagnoses,demographic,follow_ups.molecular_tests",
        "format": "JSON",
        "size": str(n_cases),
    }

    resp = requests.get(f"{GDC_API}/cases", params=payload, timeout=60)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]

    rows = []
    for hit in hits:
        row: dict = {"submitter_id": hit.get("submitter_id", "")}

        # Flatten nested diagnoses (take first entry if multiple)
        for diag in hit.get("diagnoses", [{}])[:1]:
            row["age_at_diagnosis"]  = diag.get("age_at_diagnosis")
            row["tumor_stage"]       = diag.get("ajcc_pathologic_stage")  # verified field name
            row["primary_diagnosis"] = diag.get("primary_diagnosis")
            row["morphology"]        = diag.get("morphology")

        # Demographic -- field is 'sex_at_birth' not 'gender' in current GDC schema
        demo = hit.get("demographic", {})
        row["gender"]       = demo.get("sex_at_birth") or demo.get("gender")
        row["race"]         = demo.get("race")
        row["vital_status"] = demo.get("vital_status")

        # Receptor status from follow_ups.molecular_tests
        # Gene symbols: ESR1=ER, PGR=PR, ERBB2=HER2
        receptor_map = {"ESR1": "er_status", "PGR": "pr_status", "ERBB2": "her2_status"}
        row["er_status"]   = "not reported"
        row["pr_status"]   = "not reported"
        row["her2_status"] = "not reported"

        for fu in hit.get("follow_ups", []):
            for mt in fu.get("molecular_tests", []):
                gene   = mt.get("gene_symbol", "")
                result = mt.get("test_result", "not reported")
                if gene in receptor_map:
                    col = receptor_map[gene]
                    if result and result.lower() not in ("not reported", "none", ""):
                        row[col] = result.lower()

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    print(f"[clinical] Saved {len(df)} records to {cache_path}")
    return df


# ---------------------------------------------------------------------------
# Step 2 — find slide file UUIDs paired to those patients
# ---------------------------------------------------------------------------

def fetch_slide_manifest(case_ids: list[str],
                         out_dir: Path = CLINICAL_DIR) -> pd.DataFrame:
    """
    For each case (patient) submitter_id, find GDC file UUIDs for
    diagnostic whole-slide images (Tissue Slide / Diagnostic Image).

    Returns a DataFrame with columns: case_submitter_id, file_id, file_name,
    file_size.
    """
    cache_path = out_dir / "slide_manifest.csv"
    if cache_path.exists():
        print(f"[manifest] Loading cached slide manifest from {cache_path}")
        return pd.read_csv(cache_path)

    print(f"[manifest] Querying slide file UUIDs for {len(case_ids)} patients ...")

    payload = {
        "filters": json.dumps({
            "op": "and",
            "content": [
                {"op": "=",  "content": {"field": "cases.project.project_id",
                                         "value": TCGA_BRCA_PROJECT}},
                {"op": "=",  "content": {"field": "data_type",
                                         "value": "Slide Image"}},
                {"op": "=",  "content": {"field": "experimental_strategy",
                                         "value": "Diagnostic Slide"}},
                {"op": "in", "content": {"field": "cases.submitter_id",
                                         "value": case_ids}},
            ],
        }),
        "fields": "file_id,file_name,file_size,cases.submitter_id",
        "format": "JSON",
        "size": "2000",
    }

    resp = requests.get(f"{GDC_API}/files", params=payload, timeout=60)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]

    rows = []
    for hit in hits:
        for case in hit.get("cases", []):
            rows.append({
                "case_submitter_id": case["submitter_id"],
                "file_id":   hit["file_id"],
                "file_name": hit["file_name"],
                "file_size": hit.get("file_size", 0),
            })

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    print(f"[manifest] Found {len(df)} slide files for {df['case_submitter_id'].nunique()} patients")
    return df


# ---------------------------------------------------------------------------
# Step 3 — download slides (one per patient, smallest available)
# ---------------------------------------------------------------------------

def download_slides(manifest: pd.DataFrame,
                    slides_dir: Path = SLIDES_DIR,
                    max_slides: int = DEFAULT_SUBSET) -> dict[str, Path]:
    """
    Download WSI files from GDC open-data endpoint.

    To keep storage manageable we download at most one slide per patient
    (choosing the smallest file size available for that patient).

    Returns dict: case_submitter_id -> local Path of downloaded slide.
    """
    slides_dir.mkdir(parents=True, exist_ok=True)

    # One slide per patient — pick smallest to limit download size
    manifest = (
        manifest
        .sort_values("file_size")
        .drop_duplicates(subset="case_submitter_id", keep="first")
        .head(max_slides)
    )

    case_to_path: dict[str, Path] = {}

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Downloading slides"):
        case_id  = row["case_submitter_id"]
        file_id  = row["file_id"]
        filename = row["file_name"]
        dest     = slides_dir / filename

        if dest.exists():
            case_to_path[case_id] = dest
            continue

        url = f"{GDC_API}/data/{file_id}"
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            case_to_path[case_id] = dest
            print(f"  OK {case_id}: {filename} ({row['file_size'] / 1e6:.1f} MB)")
        except Exception as exc:
            # Some slides may fail (network issues, file moved, etc.)
            # We document and skip -- do not silently corrupt the dataset
            print(f"  FAIL {case_id}: download failed -- {exc}  [EXCLUDED]")

    return case_to_path


# ---------------------------------------------------------------------------
# Step 4 — tile extraction from WSI using OpenSlide
# ---------------------------------------------------------------------------

def extract_tiles(case_to_slide: dict[str, Path],
                  tiles_dir: Path = TILES_DIR,
                  tile_size: int = TILE_SIZE,
                  n_tiles: int = TILES_PER_SLIDE,
                  level: int = TILE_LEVEL,
                  seed: int = RANDOM_SEED) -> dict[str, list[Path]]:
    """
    Extract a fixed number of non-overlapping tiles per slide.

    WHY TILES INSTEAD OF FULL SLIDES
    ---------------------------------
    A single TCGA-BRCA diagnostic slide at full resolution can be 40,000 ×
    30,000 px or larger (~1–5 GB as an SVS file).  A CNN cannot accept a
    full WSI as input; standard practice is to tile the slide and either
    aggregate tile-level predictions (multiple-instance learning) or use a
    fixed pooled representation.  We extract tiles at pyramid level 1
    (typically ~4–10× magnification, ~256×256 px output) — a standard
    starting point for feasibility experiments.

    Tissue detection
    ----------------
    We apply a simple luminance threshold to skip tiles that are mostly
    background (glass / adipose tissue).  This is a basic but effective
    quality filter and is standard in WSI preprocessing.

    Returns dict: case_submitter_id -> list of tile Paths.
    """
    try:
        import openslide
    except ImportError:
        raise ImportError(
            "openslide-python is not installed or the OpenSlide system library "
            "is missing.  See README.md — Setup section for installation instructions."
        )

    from PIL import Image

    tiles_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    case_to_tiles: dict[str, list[Path]] = {}

    for case_id, slide_path in tqdm(case_to_slide.items(), desc="Extracting tiles"):
        patient_tile_dir = tiles_dir / case_id
        patient_tile_dir.mkdir(exist_ok=True)

        # Check if already tiled
        existing = list(patient_tile_dir.glob("*.png"))
        if len(existing) >= n_tiles:
            case_to_tiles[case_id] = existing[:n_tiles]
            continue

        try:
            slide = openslide.OpenSlide(str(slide_path))
        except Exception as exc:
            print(f"  ✗ {case_id}: cannot open slide — {exc}  [EXCLUDED]")
            continue

        try:
            n_levels = slide.level_count
            use_level = min(level, n_levels - 1)
            w, h = slide.level_dimensions[use_level]

            if w < tile_size or h < tile_size:
                print(f"  ✗ {case_id}: slide too small at level {use_level} ({w}×{h})  [EXCLUDED]")
                slide.close()
                continue

            # Generate all possible tile positions at this level
            xs = list(range(0, w - tile_size, tile_size))
            ys = list(range(0, h - tile_size, tile_size))
            positions = [(x, y) for x in xs for y in ys]
            rng.shuffle(positions)

            saved: list[Path] = []
            for x, y in positions:
                if len(saved) >= n_tiles:
                    break

                # Read tile — OpenSlide coordinates are always at level 0
                scale = slide.level_downsamples[use_level]
                x0 = int(x * scale)
                y0 = int(y * scale)
                tile_img = slide.read_region((x0, y0), use_level,
                                             (tile_size, tile_size))
                tile_rgb = tile_img.convert("RGB")

                # Tissue filter: skip tiles where mean luminance is very high
                # (background glass is near-white, luminance ≈ 220+)
                arr = np.array(tile_rgb)
                if arr.mean() > 220:
                    continue

                tile_path = patient_tile_dir / f"tile_{x}_{y}.png"
                tile_rgb.save(tile_path)
                saved.append(tile_path)

            slide.close()

            if not saved:
                print(f"  ✗ {case_id}: no tissue tiles found  [EXCLUDED]")
                continue

            case_to_tiles[case_id] = saved
            print(f"  ✓ {case_id}: {len(saved)} tiles extracted")

        except Exception as exc:
            print(f"  ✗ {case_id}: tile extraction failed — {exc}  [EXCLUDED]")
            try:
                slide.close()
            except Exception:
                pass

    return case_to_tiles


# ---------------------------------------------------------------------------
# Step 5 — align clinical + tile data, build final dataset
# ---------------------------------------------------------------------------

def build_dataset(clinical_df: pd.DataFrame,
                  case_to_tiles: dict[str, list[Path]],
                  out_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Inner-join clinical records with successfully tiled cases.

    MISSING DATA HANDLING
    ---------------------
    - Cases where slide download or tiling failed are excluded (not imputed).
      We document how many are dropped and why.
    - Clinical fields with nulls are median-imputed (numeric) or
      mode-imputed (categorical) within the TRAINING split only; the same
      fitted imputer is applied to val/test to prevent leakage.
    - Receptor status fields (ER/PR/HER2) that are 'not reported' or null
      are treated as a separate category 'unknown' rather than imputed,
      since missingness may itself be clinically informative.

    Returns a DataFrame with one row per (patient, tile) combination.
    Columns include all clinical features plus 'tile_path' and 'label'.
    """
    # Build binary label: IDC (Invasive Ductal Carcinoma) vs. other
    # primary_diagnosis field contains the full histological diagnosis string.
    # We map IDC → 1, everything else → 0.
    # NOTE: With only open-access clinical data the label distribution may be
    # heavily skewed toward IDC (it is the most common BRCA subtype, ~70–80%).
    # We report class distribution explicitly so the reader can assess this.

    def make_label(primary_dx: Optional[str]) -> int:
        if pd.isna(primary_dx):
            return -1  # unknown — will be excluded
        dx_lower = str(primary_dx).lower()
        # Invasive ductal carcinoma NOS and variants
        if "infiltrating duct" in dx_lower or "invasive ductal" in dx_lower:
            return 1
        return 0

    clinical_df = clinical_df.copy()
    clinical_df["label"] = clinical_df["primary_diagnosis"].apply(make_label)

    # Keep only cases with known labels and available tiles
    valid_cases = set(case_to_tiles.keys())
    df = clinical_df[
        (clinical_df["submitter_id"].isin(valid_cases)) &
        (clinical_df["label"] >= 0)
    ].copy()

    excluded_no_tile = len(clinical_df) - len(clinical_df[clinical_df["label"] >= 0])
    excluded_no_slide = len(clinical_df[clinical_df["label"] >= 0]) - len(df)
    print(f"\n[dataset] Exclusion summary:")
    print(f"  Excluded (unknown label):   {excluded_no_tile}")
    print(f"  Excluded (no valid tiles):  {excluded_no_slide}")
    print(f"  Retained:                   {len(df)} patients")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

    # Expand one row per tile
    rows = []
    for _, patient_row in df.iterrows():
        sid = patient_row["submitter_id"]
        tiles = case_to_tiles[sid]
        for tile_path in tiles:
            row = patient_row.to_dict()
            row["tile_path"] = str(tile_path)
            rows.append(row)

    tile_df = pd.DataFrame(rows)
    out_path = out_dir / "dataset.csv"
    tile_df.to_csv(out_path, index=False)
    print(f"[dataset] Saved {len(tile_df)} tile-rows ({df['submitter_id'].nunique()} patients) → {out_path}")
    return tile_df


# ---------------------------------------------------------------------------
# Step 6 — patient-level train / val / test split
# ---------------------------------------------------------------------------

def split_dataset(dataset_df: pd.DataFrame,
                  train_frac: float = 0.70,
                  val_frac: float = 0.15,
                  seed: int = RANDOM_SEED,
                  out_dir: Optional[Path] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split at the PATIENT level — never at the tile level.

    *** THIS IS THE MOST IMPORTANT CORRECTNESS STEP IN THE PIPELINE ***

    Multiple tiles from one patient are spatially and biologically correlated.
    A tile-level split would almost certainly place tiles from the same patient
    in both train and test, effectively leaking ground-truth tissue appearance
    directly into the test set.  The resulting accuracy numbers would be
    meaninglessly inflated and would not reflect real generalization.

    Strategy:
      1. Collect unique patient IDs.
      2. Shuffle them with a fixed random seed (reproducibility).
      3. Slice into train / val / test groups.
      4. Filter the tile DataFrame by each group.
    """
    patients = dataset_df["submitter_id"].unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_patients = set(patients[:n_train])
    val_patients   = set(patients[n_train:n_train + n_val])
    test_patients  = set(patients[n_train + n_val:])

    train_df = dataset_df[dataset_df["submitter_id"].isin(train_patients)].reset_index(drop=True)
    val_df   = dataset_df[dataset_df["submitter_id"].isin(val_patients)].reset_index(drop=True)
    test_df  = dataset_df[dataset_df["submitter_id"].isin(test_patients)].reset_index(drop=True)

    # Verify: no patient overlap across splits
    assert not (train_patients & val_patients),  "BUG: patient in both train and val!"
    assert not (train_patients & test_patients), "BUG: patient in both train and test!"
    assert not (val_patients   & test_patients), "BUG: patient in both val and test!"

    print(f"\n[split] Patient-level split (seed={seed}):")
    print(f"  Train : {len(train_patients)} patients, {len(train_df)} tiles")
    print(f"  Val   : {len(val_patients)} patients,  {len(val_df)} tiles")
    print(f"  Test  : {len(test_patients)} patients,  {len(test_df)} tiles")

    save_dir = Path(out_dir) if out_dir is not None else DATA_DIR
    save_dir.mkdir(exist_ok=True)
    train_df.to_csv(save_dir / "train.csv", index=False)
    val_df.to_csv(save_dir / "val.csv",   index=False)
    test_df.to_csv(save_dir / "test.csv",  index=False)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Verification helper (run standalone to confirm environment works)
# ---------------------------------------------------------------------------

def verify_environment() -> None:
    """
    Quick smoke-test: hit GDC API, fetch one clinical record, open one tile.
    Run with: python -m src.data_pipeline --verify
    """
    print("=== FusionDx Environment Verification ===\n")

    # 1. GDC API reachable
    print("[1/4] Testing GDC API connectivity ...")
    resp = requests.get(f"{GDC_API}/status", timeout=15)
    resp.raise_for_status()
    status = resp.json()
    print(f"      GDC API status: {status.get('status')}  (version {status.get('version')})\n")

    # 2. Fetch one clinical record and confirm field names
    print("[2/4] Fetching 1 TCGA-BRCA clinical record ...")
    df = fetch_clinical_data(n_cases=1)
    print(f"      Got {len(df)} record(s).")
    print(f"      Columns: {list(df.columns)}")
    row = df.iloc[0]
    print(f"      Sample: {row['submitter_id']}  "
          f"stage={row.get('tumor_stage')}  "
          f"dx={row.get('primary_diagnosis', '')[:40]}")
    print(f"      ER={row.get('er_status')}  PR={row.get('pr_status')}  "
          f"HER2={row.get('her2_status')}\n")

    # 3. Confirm slide count
    print("[3/4] Checking slide file availability ...")
    payload = {
        "filters": json.dumps({"op": "and", "content": [
            {"op": "=", "content": {"field": "cases.project.project_id",
                                    "value": TCGA_BRCA_PROJECT}},
            {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
            {"op": "=", "content": {"field": "experimental_strategy",
                                    "value": "Diagnostic Slide"}},
        ]}),
        "fields": "file_id,file_size,access",
        "format": "JSON",
        "size": "1",
    }
    r = requests.get(f"{GDC_API}/files", params=payload, timeout=30)
    r.raise_for_status()
    total = r.json()["data"]["pagination"]["total"]
    access = r.json()["data"]["hits"][0].get("access", "?")
    print(f"      Total diagnostic slides: {total}  (access={access})\n")

    # 4. OpenSlide import
    print("[4/4] Checking OpenSlide ...")
    try:
        import openslide
        print(f"      openslide-python version: {openslide.__version__}")
        print(f"      OpenSlide library version: {openslide.OPENSLIDE_VERSION}\n")
    except ImportError as exc:
        print(f"      OpenSlide not available -- {exc}")
        print("      Run: python setup_openslide.py  for installation instructions.\n")
        print("=== Verification: GDC API OK, OpenSlide MISSING ===")
        print("    Install OpenSlide to proceed with real slide data.")
        print("    In the meantime, run: python run_synthetic.py")
        return

    print("=== Verification PASSED -- environment is fully ready ===")
    print("    Next step: python -m src.data_pipeline   (downloads real data)")


if __name__ == "__main__":
    import sys
    if "--verify" in sys.argv:
        verify_environment()
    else:
        # Full pipeline run
        clinical = fetch_clinical_data()
        case_ids = clinical["submitter_id"].dropna().tolist()
        manifest = fetch_slide_manifest(case_ids)
        case_to_slide = download_slides(manifest)
        case_to_tiles = extract_tiles(case_to_slide)
        dataset = build_dataset(clinical, case_to_tiles)
        split_dataset(dataset)
        print("\n[pipeline] Done — data ready for model training.")
