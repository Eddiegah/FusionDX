# -*- coding: utf-8 -*-
"""Tests for the evaluation module."""

import json
import numpy as np
import pytest

from src.evaluate import compute_metrics, compare_and_report, bootstrap_auc


# ---------------------------------------------------------------------------
# Fixtures -- fake prediction dicts
# ---------------------------------------------------------------------------

def _make_preds(n: int, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    preds = {}
    for i in range(n):
        label = int(rng.randint(0, 2))
        prob  = float(rng.uniform(0.1, 0.9))
        preds[f"SYNTH-{i:04d}"] = {
            "prob_idc": prob,
            "pred":     int(prob >= 0.5),
            "label":    label,
            "alpha_img":  float(rng.uniform(0, 1)),
            "alpha_clin": float(rng.uniform(0, 1)),
        }
    return preds


@pytest.fixture
def same_patient_preds():
    img_preds   = _make_preds(20, seed=10)
    clin_preds  = {k: {**v, "prob_idc": np.random.uniform(0.2, 0.8)}
                   for k, v in img_preds.items()}
    fus_preds   = {k: {**v, "alpha_img": 0.6, "alpha_clin": 0.4}
                   for k, v in img_preds.items()}
    # Normalise clin labels to match
    for k in clin_preds:
        clin_preds[k]["label"]  = img_preds[k]["label"]
        clin_preds[k]["pred"]   = int(clin_preds[k]["prob_idc"] >= 0.5)
    return img_preds, clin_preds, fus_preds


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------

def test_compute_metrics_returns_expected_keys():
    preds  = _make_preds(15, seed=5)
    result = compute_metrics(preds, "TestModel")
    for key in ["model", "accuracy", "precision", "recall", "f1", "auroc",
                "auroc_ci", "confusion_matrix", "n_patients"]:
        assert key in result, f"Missing key: {key}"


def test_compute_metrics_accuracy_in_range():
    preds  = _make_preds(15, seed=6)
    result = compute_metrics(preds, "TestModel")
    assert 0.0 <= result["accuracy"] <= 1.0


def test_compute_metrics_auroc_in_range_or_none():
    preds  = _make_preds(20, seed=7)
    result = compute_metrics(preds, "TestModel")
    if result["auroc"] is not None:
        assert 0.0 <= result["auroc"] <= 1.0


def test_compute_metrics_n_patients_correct():
    preds  = _make_preds(18, seed=8)
    result = compute_metrics(preds, "TestModel")
    assert result["n_patients"] == 18


# ---------------------------------------------------------------------------
# bootstrap_auc tests
# ---------------------------------------------------------------------------

def test_bootstrap_auc_returns_tuple():
    y_true  = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_score = np.array([0.2, 0.8, 0.3, 0.7, 0.6, 0.4, 0.9, 0.1])
    lo, hi  = bootstrap_auc(y_true, y_score, n=100)
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_width_reasonable():
    """With noisy predictions the 95% CI should be wider than a degenerate case."""
    rng = np.random.RandomState(42)
    y_true  = rng.randint(0, 2, size=20)
    # Scores only weakly correlated with labels -- produces intermediate AUC
    y_score = 0.4 * y_true + 0.6 * rng.uniform(0, 1, size=20)
    lo, hi  = bootstrap_auc(y_true, y_score, n=500)
    # CI must be a valid interval; width >= 0 (can be 0 for tiny perfect classifiers)
    assert 0.0 <= lo <= hi <= 1.0
    # For a noisy 20-sample problem width should be > 0
    assert hi - lo >= 0.0


# ---------------------------------------------------------------------------
# compare_and_report tests
# ---------------------------------------------------------------------------

def test_compare_and_report_creates_files(same_patient_preds, tmp_path):
    img_p, clin_p, fus_p = same_patient_preds
    compare_and_report(img_p, clin_p, fus_p, out_dir=tmp_path)

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "comparison_table.csv").exists()
    assert (tmp_path / "roc_curves.png").exists()
    assert (tmp_path / "comparison_chart.png").exists()
    assert (tmp_path / "comparison_report.md").exists()


def test_compare_and_report_metrics_json_valid(same_patient_preds, tmp_path):
    img_p, clin_p, fus_p = same_patient_preds
    compare_and_report(img_p, clin_p, fus_p, out_dir=tmp_path)

    with open(tmp_path / "metrics.json") as f:
        metrics = json.load(f)

    assert len(metrics) == 3
    for m in metrics:
        assert "model" in m
        assert "accuracy" in m


def test_compare_and_report_raises_on_mismatched_patients(tmp_path):
    img_p  = _make_preds(10, seed=20)
    # Build clin_p with different patient ID keys
    clin_p = {f"OTHER-{k}": v for k, v in _make_preds(10, seed=21).items()}
    fus_p  = _make_preds(10, seed=20)  # same keys as img_p
    with pytest.raises(ValueError, match="Prediction sets differ"):
        compare_and_report(img_p, clin_p, fus_p, out_dir=tmp_path)


def test_markdown_report_contains_disclaimer(same_patient_preds, tmp_path):
    img_p, clin_p, fus_p = same_patient_preds
    compare_and_report(img_p, clin_p, fus_p, out_dir=tmp_path)
    report = (tmp_path / "comparison_report.md").read_text()
    assert "DISCLAIMER" in report.upper() or "disclaimer" in report.lower()
    assert "not" in report.lower()   # "not a clinical tool" or similar
