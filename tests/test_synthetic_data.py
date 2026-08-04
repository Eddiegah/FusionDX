# -*- coding: utf-8 -*-
"""Tests for synthetic data generation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.synthetic_data import generate_synthetic_dataset, split_synthetic


@pytest.fixture(scope="module")
def synth_dataset(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("synth")
    df = generate_synthetic_dataset(
        n_patients=30,
        tiles_per_patient=4,
        seed=0,
        out_data_dir=tmp,
    )
    return df, tmp


def test_dataset_shape(synth_dataset):
    df, _ = synth_dataset
    # 30 patients x 4 tiles = 120 rows
    assert len(df) == 120


def test_dataset_columns(synth_dataset):
    df, _ = synth_dataset
    required = {"submitter_id", "label", "tile_path", "age_at_diagnosis",
                "tumor_stage", "er_status", "pr_status", "her2_status"}
    assert required.issubset(set(df.columns))


def test_labels_binary(synth_dataset):
    df, _ = synth_dataset
    assert set(df["label"].unique()).issubset({0, 1})


def test_tile_files_exist(synth_dataset):
    df, _ = synth_dataset
    missing = [p for p in df["tile_path"] if not Path(p).exists()]
    assert len(missing) == 0, f"{len(missing)} tile files missing"


def test_patient_level_split(synth_dataset):
    df, tmp = synth_dataset
    train_df, val_df, test_df = split_synthetic(df, seed=0, out_dir=tmp)

    # No overlap between splits at patient level
    train_p = set(train_df["submitter_id"].unique())
    val_p   = set(val_df["submitter_id"].unique())
    test_p  = set(test_df["submitter_id"].unique())

    assert len(train_p & val_p)  == 0, "Patient leak: train/val"
    assert len(train_p & test_p) == 0, "Patient leak: train/test"
    assert len(val_p   & test_p) == 0, "Patient leak: val/test"


def test_split_covers_all_patients(synth_dataset):
    df, tmp = synth_dataset
    train_df, val_df, test_df = split_synthetic(df, seed=0, out_dir=tmp)

    all_in  = set(df["submitter_id"].unique())
    all_out = (set(train_df["submitter_id"].unique()) |
               set(val_df["submitter_id"].unique()) |
               set(test_df["submitter_id"].unique()))
    assert all_in == all_out, "Some patients not assigned to any split"


def test_split_fractions_approximately_correct(synth_dataset):
    df, tmp = synth_dataset
    n_patients = df["submitter_id"].nunique()
    train_df, val_df, test_df = split_synthetic(
        df, train_frac=0.7, val_frac=0.15, seed=0, out_dir=tmp
    )
    train_n = train_df["submitter_id"].nunique()
    val_n   = val_df["submitter_id"].nunique()
    test_n  = test_df["submitter_id"].nunique()

    # Allow +-2 patients tolerance due to rounding
    assert abs(train_n - round(n_patients * 0.70)) <= 2
    assert abs(val_n   - round(n_patients * 0.15)) <= 2


def test_tile_images_valid_rgb(synth_dataset):
    """Sampled tiles should be 256x256 RGB PNGs."""
    from PIL import Image
    df, _ = synth_dataset
    sample = df["tile_path"].sample(5, random_state=0)
    for path in sample:
        img = Image.open(path)
        assert img.mode == "RGB"
        assert img.size == (256, 256)
