# -*- coding: utf-8 -*-
"""
Shared pytest fixtures and configuration.

The `small_splits` fixture is module-scoped and shared across test files
that import it.  Using it avoids regenerating synthetic tiles multiple
times, which is slow (~30s for 40 patients x 4 tiles).
"""

import pytest
import pandas as pd
from pathlib import Path

from src.synthetic_data import generate_synthetic_dataset, split_synthetic
from src.clinical_model import preprocess_clinical


@pytest.fixture(scope="session")
def session_tmp(tmp_path_factory):
    """Single shared temp directory for the whole test session."""
    return tmp_path_factory.mktemp("session")


@pytest.fixture(scope="session")
def small_dataset(session_tmp):
    """Generate a small synthetic dataset once per test session."""
    df = generate_synthetic_dataset(
        n_patients=30,
        tiles_per_patient=4,
        seed=99,
        out_data_dir=session_tmp,
    )
    return df, session_tmp


@pytest.fixture(scope="session")
def small_splits(small_dataset):
    """Patient-level splits of the small dataset."""
    df, tmp = small_dataset
    train_df, val_df, test_df = split_synthetic(df, seed=99, out_dir=tmp)
    return train_df, val_df, test_df


@pytest.fixture(scope="session")
def clinical_state(small_splits):
    """Fitted clinical preprocessing state from train split."""
    train_df, _, _ = small_splits
    _, state = preprocess_clinical(
        train_df.drop_duplicates("submitter_id"), fit=True
    )
    return state
