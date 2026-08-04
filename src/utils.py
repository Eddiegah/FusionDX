# -*- coding: utf-8 -*-
"""
FusionDx -- Shared Utilities
==============================
Small helpers used across multiple modules.
"""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic mode -- slightly slower but reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# JSON persistence helpers
# ---------------------------------------------------------------------------

def save_json(obj: Any, path: Path | str) -> None:
    """Save a JSON-serialisable object, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path | str) -> Any:
    """Load JSON, returning None if file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prediction dict helpers
# ---------------------------------------------------------------------------

def preds_to_arrays(preds: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack a predictions dict into sorted parallel arrays.

    Returns
    -------
    sids    : sorted patient IDs
    y_true  : ground truth labels
    y_prob  : predicted P(IDC)
    """
    sids   = sorted(preds.keys())
    y_true = np.array([preds[s]["label"]    for s in sids])
    y_prob = np.array([preds[s]["prob_idc"] for s in sids])
    return sids, y_true, y_prob


def preds_overlap(a: dict, b: dict, c: dict) -> bool:
    """Return True if all three prediction dicts cover the same patient set."""
    return set(a.keys()) == set(b.keys()) == set(c.keys())


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(prefer_cuda: bool = True) -> str:
    """Return 'cuda' if available and preferred, else 'cpu'."""
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Model checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: torch.nn.Module, path: Path | str,
                    extra: dict | None = None) -> None:
    """Save model state dict with optional metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: Path | str,
                    device: str = "cpu") -> torch.nn.Module:
    """
    Load state dict from a checkpoint saved by save_checkpoint or torch.save.
    Handles both bare state_dict and wrapped {'state_dict': ...} formats.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        model.load_state_dict(payload["state_dict"])
    else:
        model.load_state_dict(payload)
    return model


# ---------------------------------------------------------------------------
# Class imbalance helpers
# ---------------------------------------------------------------------------

def compute_class_weights(labels: list[int] | np.ndarray,
                           n_classes: int = 2) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for use in CrossEntropyLoss.

    Weights are normalised so the mean weight is 1.0, which keeps the loss
    scale comparable to the unweighted case.
    """
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    # Normalise
    weights = weights / (weights.mean() + 1e-8)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Formatting helpers for dashboard / reports
# ---------------------------------------------------------------------------

def fmt_age(age_days: float | None) -> str:
    """Convert GDC age_at_diagnosis (days) to a readable string."""
    if age_days is None:
        return "unknown"
    years = age_days / 365.25
    return f"{years:.1f} yrs"


def fmt_prob(p: float) -> str:
    """Format a probability as a percentage string."""
    return f"{p * 100:.1f}%"


def fmt_stage(stage: str | None) -> str:
    """Clean up AJCC stage string for display."""
    if not stage:
        return "unknown"
    return stage.replace("stage ", "Stage ").strip()
