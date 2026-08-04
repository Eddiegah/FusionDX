# -*- coding: utf-8 -*-
"""
FusionDx -- Synthetic Data Generator
=======================================
Generates realistic synthetic TCGA-BRCA-like data for development and testing
when real GDC data is not yet available (e.g. before OpenSlide is installed or
while the data pipeline is downloading).

THIS IS NOT REAL PATIENT DATA.
The synthetic generator produces:
  - Fake patient clinical records with plausible distributions
    (age, stage, receptor status) drawn from published TCGA-BRCA statistics
  - Synthetic image tiles that mimic hematoxylin & eosin (H&E) stain colour
    distributions -- pink/purple tissue on white background

The synthetic pipeline lets you:
  1. Verify the full training + evaluation + dashboard stack works end-to-end
  2. Develop model architecture changes without waiting for data downloads
  3. Run the test suite in CI without real patient data

IMPORTANT: Synthetic results have NO clinical meaning whatsoever.
Only results from the real TCGA-BRCA data should be interpreted or reported.
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from src.config import (
    SYNTH_DIR, TILES_DIR, CLINICAL_DIR, DATA_DIR,
    SYNTH_N_PATIENTS, SYNTH_TILES_EACH, SYNTH_LABEL_RATIO,
    TRAIN_FRAC, VAL_FRAC, TILE_SIZE, RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Plausible distributions drawn from TCGA-BRCA published statistics
# ---------------------------------------------------------------------------

STAGES = [
    ("stage i",   0.20),
    ("stage ia",  0.05),
    ("stage ib",  0.02),
    ("stage ii",  0.18),
    ("stage iia", 0.18),
    ("stage iib", 0.12),
    ("stage iii", 0.05),
    ("stage iiia",0.08),
    ("stage iiib",0.02),
    ("stage iiic",0.03),
    ("stage iv",  0.02),
    ("not reported", 0.05),
]

RECEPTOR_STATUS = [
    ("positive",     0.72),
    ("negative",     0.22),
    ("not reported", 0.06),
]

HER2_STATUS = [
    ("positive",     0.20),
    ("negative",     0.72),
    ("not reported", 0.08),
]

RACES = [
    ("white",                                        0.68),
    ("black or african american",                    0.19),
    ("asian",                                        0.06),
    ("american indian or alaska native",             0.01),
    ("native hawaiian or other pacific islander",    0.01),
    ("not reported",                                 0.05),
]

PRIMARY_DX = {
    1: "Infiltrating duct carcinoma, NOS",      # IDC -- label 1
    0: "Lobular carcinoma, NOS",                # other -- label 0
}


def _weighted_choice(choices: list[tuple], rng: random.Random) -> str:
    """Sample from (value, weight) list."""
    values, weights = zip(*choices)
    return rng.choices(values, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Synthetic H&E tile generator
# ---------------------------------------------------------------------------

def _make_he_tile(label: int, rng: random.Random,
                  size: int = TILE_SIZE) -> Image.Image:
    """
    Generate a synthetic H&E-stained tile.

    Strategy:
    - Background: near-white (240-255 in all channels)
    - Nuclei: small dark purple ellipses
    - Cytoplasm / stroma: pink regions
    - IDC tiles (label=1) have higher nuclear density + slightly darker stain
      to create a signal the image model can (weakly) learn from.
    This is intentionally simple -- the goal is a realistic RGB range,
    not actual tissue morphology.
    """
    arr = np.ones((size, size, 3), dtype=np.uint8)
    # White background
    arr[:] = [245, 238, 240]

    # Pink eosin regions (cytoplasm)
    n_pink = rng.randint(3, 8)
    for _ in range(n_pink):
        cx = rng.randint(20, size - 20)
        cy = rng.randint(20, size - 20)
        rx = rng.randint(15, 45)
        ry = rng.randint(15, 45)
        for dy in range(-ry, ry):
            for dx in range(-rx, rx):
                if (dx / rx) ** 2 + (dy / ry) ** 2 <= 1:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < size and 0 <= py < size:
                        arr[py, px] = [
                            min(255, 220 + rng.randint(-10, 10)),
                            min(255, 160 + rng.randint(-15, 15)),
                            min(255, 165 + rng.randint(-15, 15)),
                        ]

    # Purple haematoxylin nuclei -- more for IDC
    n_nuclei = rng.randint(8, 16) if label == 1 else rng.randint(2, 7)
    nucleus_darkness = rng.randint(50, 90) if label == 1 else rng.randint(90, 130)
    for _ in range(n_nuclei):
        cx = rng.randint(10, size - 10)
        cy = rng.randint(10, size - 10)
        r  = rng.randint(4, 10)
        for dy in range(-r, r):
            for dx in range(-r, r):
                if dx * dx + dy * dy <= r * r:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < size and 0 <= py < size:
                        arr[py, px] = [
                            nucleus_darkness + rng.randint(-10, 10),
                            nucleus_darkness - 20 + rng.randint(-10, 10),
                            min(255, nucleus_darkness + 30 + rng.randint(-10, 10)),
                        ]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_patients: int = SYNTH_N_PATIENTS,
    tiles_per_patient: int = SYNTH_TILES_EACH,
    label_ratio: float = SYNTH_LABEL_RATIO,
    seed: int = RANDOM_SEED,
    out_data_dir: Path = DATA_DIR,
    tiles_dir: Optional[Path] = None,
    clinical_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate synthetic clinical + image tile data.

    Returns a DataFrame with one row per (patient, tile) -- same schema as
    the real data_pipeline.build_dataset() output, so all downstream code
    works identically.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    tiles_dir    = tiles_dir    or out_data_dir / "tiles"
    clinical_dir = clinical_dir or out_data_dir / "clinical"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    clinical_dir.mkdir(parents=True, exist_ok=True)

    print(f"[synthetic] Generating {n_patients} synthetic patients "
          f"({tiles_per_patient} tiles each) ...")

    rows = []
    for i in range(n_patients):
        sid   = f"SYNTH-{i:04d}"
        label = 1 if rng.random() < label_ratio else 0

        # Clinical record
        age_days = rng.gauss(55 * 365.25, 10 * 365.25)  # ~55 +/- 10 yrs
        stage    = _weighted_choice(STAGES, rng)
        er       = _weighted_choice(RECEPTOR_STATUS, rng)
        pr       = _weighted_choice(RECEPTOR_STATUS, rng)
        her2     = _weighted_choice(HER2_STATUS, rng)
        race     = _weighted_choice(RACES, rng)

        # Generate tiles
        patient_tile_dir = tiles_dir / sid
        patient_tile_dir.mkdir(exist_ok=True)

        for t in range(tiles_per_patient):
            tile_path = patient_tile_dir / f"tile_{t:03d}.png"
            if not tile_path.exists():
                img = _make_he_tile(label, rng)
                img.save(tile_path)

            rows.append({
                "submitter_id":       sid,
                "age_at_diagnosis":   max(20 * 365.25, age_days),
                "tumor_stage":        stage,
                "primary_diagnosis":  PRIMARY_DX[label],
                "morphology":         "8500/3" if label == 1 else "8520/3",
                "er_status":          er,
                "pr_status":          pr,
                "her2_status":        her2,
                "vital_status":       rng.choice(["Alive", "Dead"]),
                "gender":             "female",
                "race":               race,
                "label":              label,
                "tile_path":          str(tile_path),
            })

    df = pd.DataFrame(rows)
    dataset_path = out_data_dir / "dataset.csv"
    df.to_csv(dataset_path, index=False)

    label_counts = df.drop_duplicates("submitter_id")["label"].value_counts()
    print(f"[synthetic] Generated {n_patients} patients "
          f"(IDC={label_counts.get(1,0)}, Other={label_counts.get(0,0)})")
    print(f"[synthetic] Dataset saved to {dataset_path}")

    # Save synthetic clinical CSV for reference
    clinical_df = df.drop_duplicates("submitter_id").drop(columns=["tile_path"])
    clinical_df.to_csv(clinical_dir / "clinical_raw.csv", index=False)

    return df


def split_synthetic(dataset_df: pd.DataFrame,
                    train_frac: float = TRAIN_FRAC,
                    val_frac: float = VAL_FRAC,
                    seed: int = RANDOM_SEED,
                    out_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-level split for synthetic data -- same logic as real pipeline.
    Reusing data_pipeline.split_dataset would create a circular import;
    we replicate the logic here for clarity.
    """
    patients = dataset_df["submitter_id"].unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(patients)

    n       = len(patients)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_p = set(patients[:n_train])
    val_p   = set(patients[n_train:n_train + n_val])
    test_p  = set(patients[n_train + n_val:])

    train_df = dataset_df[dataset_df["submitter_id"].isin(train_p)].reset_index(drop=True)
    val_df   = dataset_df[dataset_df["submitter_id"].isin(val_p)].reset_index(drop=True)
    test_df  = dataset_df[dataset_df["submitter_id"].isin(test_p)].reset_index(drop=True)

    # Safety assertions -- same as real pipeline
    assert not (train_p & val_p)
    assert not (train_p & test_p)
    assert not (val_p & test_p)

    out_dir.mkdir(exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv",     index=False)
    test_df.to_csv(out_dir / "test.csv",   index=False)

    print(f"[synthetic] Patient-level split:")
    print(f"  Train : {len(train_p)} patients, {len(train_df)} tiles")
    print(f"  Val   : {len(val_p)} patients,  {len(val_df)} tiles")
    print(f"  Test  : {len(test_p)} patients,  {len(test_df)} tiles")
    return train_df, val_df, test_df
