# -*- coding: utf-8 -*-
"""Tests for src/utils.py."""

import json
import numpy as np
import pytest
import torch
from pathlib import Path

from src.utils import (
    set_seed,
    save_json,
    load_json,
    preds_to_arrays,
    preds_overlap,
    compute_class_weights,
    fmt_age,
    fmt_prob,
    fmt_stage,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_set_seed_makes_torch_deterministic():
    set_seed(0)
    a = torch.randn(5)
    set_seed(0)
    b = torch.randn(5)
    assert torch.allclose(a, b), "Same seed should produce same random tensor"


def test_set_seed_makes_numpy_deterministic():
    set_seed(7)
    a = np.random.randn(5)
    set_seed(7)
    b = np.random.randn(5)
    assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def test_save_and_load_json(tmp_path):
    obj = {"key": [1, 2, 3], "nested": {"a": "b"}}
    path = tmp_path / "test.json"
    save_json(obj, path)
    loaded = load_json(path)
    assert loaded == obj


def test_load_json_returns_none_if_missing(tmp_path):
    result = load_json(tmp_path / "nonexistent.json")
    assert result is None


def test_save_json_creates_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "nested" / "file.json"
    save_json({"x": 1}, path)
    assert path.exists()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _fake_preds(n=10, seed=0):
    rng = np.random.RandomState(seed)
    return {
        f"P-{i:04d}": {
            "prob_idc": float(rng.uniform(0, 1)),
            "pred":     int(rng.randint(0, 2)),
            "label":    int(rng.randint(0, 2)),
        }
        for i in range(n)
    }


def test_preds_to_arrays_shape():
    preds = _fake_preds(8)
    sids, y_true, y_prob = preds_to_arrays(preds)
    assert len(sids) == 8
    assert y_true.shape == (8,)
    assert y_prob.shape == (8,)


def test_preds_to_arrays_sorted():
    preds = _fake_preds(5)
    sids, _, _ = preds_to_arrays(preds)
    assert sids == sorted(sids)


def test_preds_overlap_true():
    p = _fake_preds(5, seed=0)
    assert preds_overlap(p, p, p)


def test_preds_overlap_false():
    a = _fake_preds(5, seed=0)
    b = {f"OTHER-{k}": v for k, v in _fake_preds(5, seed=1).items()}
    assert not preds_overlap(a, b, a)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def test_compute_class_weights_shape():
    labels = [0, 1, 0, 1, 1, 0, 0, 1]
    w = compute_class_weights(labels, n_classes=2)
    assert w.shape == (2,)


def test_compute_class_weights_inverse_frequency():
    # 1 class-0, 3 class-1 -> class-0 should be weighted higher
    labels = [0, 1, 1, 1]
    w = compute_class_weights(labels, n_classes=2)
    assert w[0] > w[1], f"Expected w[0]>w[1] for imbalanced labels, got {w}"


def test_compute_class_weights_returns_tensor():
    w = compute_class_weights([0, 1, 0, 1])
    assert isinstance(w, torch.Tensor)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def test_fmt_age_none():
    assert fmt_age(None) == "unknown"


def test_fmt_age_years():
    result = fmt_age(365.25 * 55)
    assert "55.0" in result


def test_fmt_prob():
    assert fmt_prob(0.753) == "75.3%"
    assert fmt_prob(0.0)   == "0.0%"
    assert fmt_prob(1.0)   == "100.0%"


def test_fmt_stage_none():
    assert fmt_stage(None) == "unknown"


def test_fmt_stage_normalises_case():
    assert fmt_stage("stage iia") == "Stage iia"
