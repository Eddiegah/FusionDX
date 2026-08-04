# -*- coding: utf-8 -*-
"""Tests for clinical model preprocessing and training."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.synthetic_data import generate_synthetic_dataset, split_synthetic
from src.clinical_model import (
    preprocess_clinical,
    ClinicalMLP,
    train_clinical_model_lgbm,
    train_clinical_mlp,
    predict_patients_lgbm,
    predict_patients_mlp,
)


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("clin")
    df = generate_synthetic_dataset(n_patients=60, tiles_per_patient=4, seed=1, out_data_dir=tmp)
    train_df, val_df, test_df = split_synthetic(df, seed=1, out_dir=tmp)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

def test_preprocess_returns_correct_shape(splits):
    train_df, _, _ = splits
    patients = train_df.drop_duplicates("submitter_id")
    X, state = preprocess_clinical(patients, fit=True)
    assert X.ndim == 2
    assert X.shape[0] == len(patients)
    assert X.shape[1] > 5  # at least age, stage, 3 receptor + gender + races


def test_preprocess_no_nans_after_imputation(splits):
    train_df, val_df, _ = splits
    train_patients = train_df.drop_duplicates("submitter_id")
    val_patients   = val_df.drop_duplicates("submitter_id")

    X_train, state = preprocess_clinical(train_patients, fit=True)
    X_val, _       = preprocess_clinical(val_patients, fit=False, fitted_state=state)

    assert not np.isnan(X_train).any(), "NaNs in training features after imputation"
    assert not np.isnan(X_val).any(),   "NaNs in val features after imputation"


def test_preprocess_fitted_state_required_for_val(splits):
    _, val_df, _ = splits
    patients = val_df.drop_duplicates("submitter_id")
    with pytest.raises(AssertionError):
        preprocess_clinical(patients, fit=False, fitted_state=None)


def test_preprocess_feature_cols_consistent(splits):
    train_df, val_df, _ = splits
    X_train, state = preprocess_clinical(
        train_df.drop_duplicates("submitter_id"), fit=True
    )
    X_val, _ = preprocess_clinical(
        val_df.drop_duplicates("submitter_id"), fit=False, fitted_state=state
    )
    assert X_train.shape[1] == X_val.shape[1], "Feature dimension mismatch train vs val"


# ---------------------------------------------------------------------------
# ClinicalMLP architecture tests
# ---------------------------------------------------------------------------

def test_mlp_forward_shape():
    model = ClinicalMLP(input_dim=11)
    x = torch.randn(8, 11)
    logits = model(x)
    assert logits.shape == (8, 2), f"Expected (8, 2), got {logits.shape}"


def test_mlp_embedding_shape():
    model = ClinicalMLP(input_dim=11)
    x = torch.randn(8, 11)
    emb = model.get_embedding(x)
    assert emb.shape == (8, ClinicalMLP.EMBED_DIM)


def test_mlp_embedding_differentiable():
    model = ClinicalMLP(input_dim=11)
    x = torch.randn(4, 11, requires_grad=False)
    emb = model.get_embedding(x)
    loss = emb.sum()
    loss.backward()  # should not raise


# ---------------------------------------------------------------------------
# LightGBM training test (fast)
# ---------------------------------------------------------------------------

def test_lgbm_trains_and_predicts(splits, tmp_path):
    train_df, val_df, test_df = splits
    booster, state = train_clinical_model_lgbm(train_df, val_df, checkpoint_dir=tmp_path)
    preds = predict_patients_lgbm(booster, test_df, state)

    test_patients = set(test_df["submitter_id"].unique())
    assert set(preds.keys()) == test_patients

    for sid, result in preds.items():
        assert 0.0 <= result["prob_idc"] <= 1.0
        assert result["pred"] in (0, 1)
        assert result["label"] in (0, 1)


# ---------------------------------------------------------------------------
# MLP training test (fast, few epochs)
# ---------------------------------------------------------------------------

def test_mlp_trains_and_predicts(splits, tmp_path):
    train_df, val_df, test_df = splits
    model, state = train_clinical_mlp(
        train_df, val_df, checkpoint_dir=tmp_path, epochs=5
    )
    preds = predict_patients_mlp(model, test_df, state)

    test_patients = set(test_df["submitter_id"].unique())
    assert set(preds.keys()) == test_patients

    for sid, result in preds.items():
        assert 0.0 <= result["prob_idc"] <= 1.0
        assert result["pred"] in (0, 1)
