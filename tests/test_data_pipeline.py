# -*- coding: utf-8 -*-
"""
Tests for data_pipeline.py.

Network tests (fetch_clinical_data, fetch_slide_manifest) are marked
with @pytest.mark.gdc and only run when the GDC API is reachable.
Heavy download tests are not run in CI -- they require disk space and time.
"""

import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (
    split_dataset,
    build_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_dataset(n_patients: int = 20,
                           tiles_per: int = 4,
                           seed: int = 0) -> pd.DataFrame:
    """Build a minimal in-memory dataset without real files."""
    rng = random.Random(seed)
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for i in range(n_patients):
            sid   = f"TCGA-XX-{i:04d}"
            label = rng.randint(0, 1)
            for t in range(tiles_per):
                tile_path = tmpdir / sid / f"tile_{t}.png"
                tile_path.parent.mkdir(exist_ok=True)
                # Write a minimal 1-byte file so the path is valid-looking
                tile_path.write_bytes(b"\x00")
                rows.append({
                    "submitter_id":    sid,
                    "age_at_diagnosis": rng.uniform(20*365, 80*365),
                    "tumor_stage":      rng.choice(["Stage I", "Stage II", "Stage III"]),
                    "primary_diagnosis": "Infiltrating duct carcinoma, NOS" if label else "Lobular carcinoma, NOS",
                    "er_status":   rng.choice(["positive", "negative"]),
                    "pr_status":   rng.choice(["positive", "negative"]),
                    "her2_status": rng.choice(["positive", "negative"]),
                    "gender":      "female",
                    "race":        "white",
                    "label":       label,
                    "tile_path":   str(tile_path),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# split_dataset tests (pure logic, no I/O)
# ---------------------------------------------------------------------------

def test_split_dataset_no_patient_overlap(tmp_path):
    df = _make_minimal_dataset(30, 4)
    train_df, val_df, test_df = split_dataset(df, out_dir=tmp_path)

    train_p = set(train_df["submitter_id"].unique())
    val_p   = set(val_df["submitter_id"].unique())
    test_p  = set(test_df["submitter_id"].unique())

    assert len(train_p & val_p)  == 0, "Patient overlap: train/val"
    assert len(train_p & test_p) == 0, "Patient overlap: train/test"
    assert len(val_p   & test_p) == 0, "Patient overlap: val/test"


def test_split_dataset_covers_all_patients(tmp_path):
    df  = _make_minimal_dataset(20, 4)
    all_p = set(df["submitter_id"].unique())
    train_df, val_df, test_df = split_dataset(df, out_dir=tmp_path)
    split_p = (set(train_df["submitter_id"].unique()) |
               set(val_df["submitter_id"].unique()) |
               set(test_df["submitter_id"].unique()))
    assert all_p == split_p


def test_split_dataset_writes_csvs(tmp_path):
    df = _make_minimal_dataset(20, 4)
    split_dataset(df, out_dir=tmp_path)
    for fname in ("train.csv", "val.csv", "test.csv"):
        assert (tmp_path / fname).exists(), f"Missing {fname}"


def test_split_dataset_fractions(tmp_path):
    df = _make_minimal_dataset(40, 2)
    n  = df["submitter_id"].nunique()
    train_df, val_df, test_df = split_dataset(df, train_frac=0.7, val_frac=0.15,
                                               out_dir=tmp_path)
    assert abs(train_df["submitter_id"].nunique() - round(n * 0.70)) <= 2
    assert abs(val_df["submitter_id"].nunique()   - round(n * 0.15)) <= 2


def test_split_reproducible(tmp_path):
    df = _make_minimal_dataset(20, 4)
    t1, v1, ts1 = split_dataset(df, seed=99, out_dir=tmp_path)
    t2, v2, ts2 = split_dataset(df, seed=99, out_dir=tmp_path)
    assert list(t1["submitter_id"].unique()) == list(t2["submitter_id"].unique())
    assert list(ts1["submitter_id"].unique()) == list(ts2["submitter_id"].unique())


# ---------------------------------------------------------------------------
# build_dataset tests
# ---------------------------------------------------------------------------

def test_build_dataset_excludes_unknown_labels(tmp_path):
    """Rows with label == -1 (unknown primary diagnosis) should be excluded."""
    from src.synthetic_data import generate_synthetic_dataset
    df = generate_synthetic_dataset(n_patients=20, tiles_per_patient=4,
                                    seed=5, out_data_dir=tmp_path)
    # Manually corrupt one patient's label to -1 in clinical_df
    clinical_df = df.drop_duplicates("submitter_id").copy()
    clinical_df.loc[clinical_df.index[0], "primary_diagnosis"] = None  # will map to -1
    case_to_tiles = {
        sid: [Path(p)] for sid, p in
        df.groupby("submitter_id")["tile_path"].first().items()
    }
    result = build_dataset(clinical_df, case_to_tiles, out_dir=tmp_path)
    # The patient with None diagnosis should be excluded
    assert -1 not in result["label"].values


def test_build_dataset_idc_label_mapping(tmp_path):
    """Infiltrating duct carcinoma should map to label=1."""
    from src.synthetic_data import generate_synthetic_dataset
    df = generate_synthetic_dataset(n_patients=10, tiles_per_patient=4,
                                    seed=6, out_data_dir=tmp_path)
    clinical_df = df.drop_duplicates("submitter_id").copy()
    case_to_tiles = {
        sid: [Path(p)] for sid, p in
        df.groupby("submitter_id")["tile_path"].first().items()
    }
    result = build_dataset(clinical_df, case_to_tiles, out_dir=tmp_path)
    idc_rows = result[result["primary_diagnosis"].str.contains("duct", case=False, na=False)]
    assert (idc_rows["label"] == 1).all(), "IDC cases should all have label=1"


# ---------------------------------------------------------------------------
# GDC live fetch tests (require internet)
# ---------------------------------------------------------------------------

@pytest.mark.gdc
def test_fetch_clinical_data_returns_dataframe(tmp_path):
    from src.data_pipeline import fetch_clinical_data
    df = fetch_clinical_data(n_cases=5, out_dir=tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "submitter_id" in df.columns
    assert "primary_diagnosis" in df.columns


@pytest.mark.gdc
def test_fetch_clinical_data_stage_field(tmp_path):
    """Confirm ajcc_pathologic_stage is fetched (not the old tumor_stage field)."""
    from src.data_pipeline import fetch_clinical_data
    df = fetch_clinical_data(n_cases=5, out_dir=tmp_path)
    # tumor_stage column stores the ajcc_pathologic_stage values
    has_stage = df["tumor_stage"].notna().any()
    assert has_stage, "No non-null stage values found -- check field name"


@pytest.mark.gdc
def test_fetch_slide_manifest_returns_dataframe(tmp_path):
    from src.data_pipeline import fetch_clinical_data, fetch_slide_manifest
    clinical_df = fetch_clinical_data(n_cases=10, out_dir=tmp_path)
    case_ids    = clinical_df["submitter_id"].dropna().tolist()
    manifest    = fetch_slide_manifest(case_ids, out_dir=tmp_path)
    assert isinstance(manifest, pd.DataFrame)
    assert "file_id"  in manifest.columns
    assert "file_name" in manifest.columns
    assert "case_submitter_id" in manifest.columns
    assert len(manifest) > 0
