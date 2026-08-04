# -*- coding: utf-8 -*-
"""
FusionDx -- Clinical-Data-Only Baseline Model
=============================================
A LightGBM gradient-boosted tree trained on tabular clinical features alone.

WHY LightGBM?
--------------
- Gradient-boosted trees are consistently the strongest baseline for tabular
  clinical data in the medical literature (the 'XGBoost/LightGBM is hard to
  beat on tabular data' finding from many benchmarks holds here).
- They handle mixed numeric/categorical features and moderate missingness
  naturally without extensive preprocessing.
- They are fast to train, interpretable via SHAP feature-importance scores,
  and don't require hyperparameter tuning to produce a reasonable baseline.
- An MLP alternative is also included as a secondary option, since the fusion
  model needs a learned embedding from the clinical branch (LightGBM's leaf
  embeddings are not differentiable; the MLP is used for that purpose).

FEATURES USED
--------------
  - age_at_diagnosis      (numeric — in days from GDC; converted to years)
  - tumor_stage           (ordinal-encoded: I < II < III < IV)
  - er_status             (binary: positive / negative / unknown)
  - pr_status             (binary)
  - her2_status           (binary)
  - gender                (binary; TCGA-BRCA is overwhelmingly female)
  - race                  (one-hot — 5 categories + unknown)

MISSING DATA
-----------
  - Numeric: median imputation (fit on train only)
  - Categorical: mode imputation / 'unknown' category
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

CLINICAL_FEATURES = [
    "age_at_diagnosis",
    "tumor_stage",
    "er_status",
    "pr_status",
    "her2_status",
    "gender",
    "race",
]

STAGE_ORDER = [
    "stage i", "stage ia", "stage ib",
    "stage ii", "stage iia", "stage iib",
    "stage iii", "stage iiia", "stage iiib", "stage iiic",
    "stage iv",
    "not reported", "unknown",
]

BINARY_CATEGORIES = ["positive", "negative", "not reported", "unknown"]
GENDER_CATEGORIES = ["female", "male", "unknown"]
RACE_CATEGORIES   = [
    "white", "black or african american", "asian",
    "american indian or alaska native",
    "native hawaiian or other pacific islander",
    "not reported", "unknown",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def preprocess_clinical(df: pd.DataFrame,
                        fit: bool = True,
                        fitted_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    """
    Encode and impute clinical features.

    Parameters
    ----------
    df          : DataFrame with raw clinical columns
    fit         : If True, compute and return imputation stats (train mode).
                  If False, use fitted_state (val/test mode — prevents leakage).
    fitted_state: dict returned by a previous call with fit=True.

    Returns
    -------
    X : np.ndarray of shape (n_patients, n_features)
    state : dict with fitted imputation values (or None if fit=False)
    """
    df = df.drop_duplicates(subset="submitter_id").copy()

    # --- Age (days → years, numeric) ---
    df["age_years"] = pd.to_numeric(df["age_at_diagnosis"], errors="coerce") / 365.25

    # --- Tumor stage (ordinal) ---
    # GDC field is 'ajcc_pathologic_stage' (e.g. 'Stage IIB')
    stage_col = df["tumor_stage"].str.lower().str.strip().fillna("unknown")
    stage_map = {s: i for i, s in enumerate(STAGE_ORDER)}
    df["stage_enc"] = stage_col.map(stage_map).fillna(len(STAGE_ORDER) - 1).astype(float)

    # --- Receptor statuses (ordinal: positive=2, negative=1, unknown=0) ---
    receptor_map = {"positive": 2, "negative": 1, "not reported": 0, "unknown": 0}
    for col in ["er_status", "pr_status", "her2_status"]:
        df[f"{col}_enc"] = (
            df[col].str.lower().str.strip()
              .fillna("unknown")
              .map(receptor_map)
              .fillna(0)
              .astype(float)
        )

    # --- Gender (binary) ---
    df["gender_enc"] = (df["gender"].str.lower().str.strip() == "female").astype(float)

    # --- Race (one-hot, 7 categories) ---
    race_clean = df["race"].str.lower().str.strip().fillna("unknown")
    for cat in RACE_CATEGORIES[:-2]:  # skip 'not reported' and 'unknown' as reference
        df[f"race_{cat.replace(' ', '_')}"] = (race_clean == cat).astype(float)

    feature_cols = (
        ["age_years", "stage_enc", "er_status_enc", "pr_status_enc", "her2_status_enc",
         "gender_enc"]
        + [f"race_{c.replace(' ', '_')}" for c in RACE_CATEGORIES[:-2]]
    )

    X = df[feature_cols].values.astype(np.float32)

    # --- Impute missing numeric values ---
    if fit:
        col_medians = np.nanmedian(X, axis=0)
        state = {"col_medians": col_medians, "feature_cols": feature_cols}
    else:
        assert fitted_state is not None, "fitted_state required when fit=False"
        col_medians = fitted_state["col_medians"]
        state = None

    nan_mask = np.isnan(X)
    for j in range(X.shape[1]):
        X[nan_mask[:, j], j] = col_medians[j]

    return X, state


# ---------------------------------------------------------------------------
# LightGBM classifier (for standalone clinical baseline)
# ---------------------------------------------------------------------------

def train_clinical_model_lgbm(train_df: pd.DataFrame,
                               val_df: pd.DataFrame,
                               checkpoint_dir: Path = Path("results"),
                               seed: int = 42):
    """
    Train a LightGBM binary classifier on patient-level clinical features.

    Returns (booster, fitted_state).
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm is not installed — run: pip install lightgbm==4.3.0")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # One row per patient (drop tile duplication)
    train_patients = train_df.drop_duplicates("submitter_id")
    val_patients   = val_df.drop_duplicates("submitter_id")

    X_train, state = preprocess_clinical(train_patients, fit=True)
    y_train = train_patients["label"].astype(int).values

    X_val, _ = preprocess_clinical(val_patients, fit=False, fitted_state=state)
    y_val   = val_patients["label"].astype(int).values

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    params = {
        "objective":       "binary",
        "metric":          "binary_logloss",
        "num_leaves":      31,
        "learning_rate":   0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq":    5,
        "verbose":        -1,
        "seed":            seed,
    }

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=300,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(30, verbose=True),
                   lgb.log_evaluation(50)],
    )

    model_path = checkpoint_dir / "clinical_lgbm.txt"
    booster.save_model(str(model_path))
    print(f"[clinical_model] LightGBM saved to {model_path}")
    return booster, state


def predict_patients_lgbm(booster, df: pd.DataFrame,
                           fitted_state: dict) -> dict[str, dict]:
    """Patient-level predictions from the LightGBM model."""
    patients = df.drop_duplicates("submitter_id")
    X, _ = preprocess_clinical(patients, fit=False, fitted_state=fitted_state)
    probs = booster.predict(X)  # P(IDC)

    results = {}
    for sid, prob, label in zip(
        patients["submitter_id"].values,
        probs,
        patients["label"].astype(int).values,
    ):
        results[sid] = {
            "prob_idc": float(prob),
            "pred":     int(prob >= 0.5),
            "label":    int(label),
        }
    return results


# ---------------------------------------------------------------------------
# MLP clinical model (differentiable — used for fusion model embeddings)
# ---------------------------------------------------------------------------

class ClinicalMLP(nn.Module):
    """
    Small MLP that produces a differentiable embedding from clinical features.
    Used as the clinical branch inside the fusion model.

    WHY MLP ALONGSIDE LIGHTGBM?
    ----------------------------
    LightGBM is the stronger standalone classifier for tabular data, so we
    report it for the clinical-only baseline numbers.  However, the fusion
    model needs a differentiable, end-to-end trainable clinical encoder so
    that gradient signals from the fusion loss can flow back through both
    modalities.  The MLP serves this role.

    Architecture:
      input → FC(input_dim, 128) → LayerNorm → ReLU → Dropout(0.3)
            → FC(128, 64) → LayerNorm → ReLU
            → FC(64, embed_dim)   ← embedding exposed to fusion layer
            → FC(embed_dim, 2)    ← classification head (standalone use)
    """

    EMBED_DIM = 64

    def __init__(self, input_dim: int = 11):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.embedding_layer = nn.Linear(64, self.EMBED_DIM)
        self.classifier      = nn.Linear(self.EMBED_DIM, 2)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return 64-d embedding (used by fusion model)."""
        h = self.encoder(x)
        return self.embedding_layer(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding(x)
        return self.classifier(emb)


# ---------------------------------------------------------------------------
# MLP training (for standalone evaluation)
# ---------------------------------------------------------------------------

def train_clinical_mlp(train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        checkpoint_dir: Path = Path("results"),
                        epochs: int = 50,
                        lr: float = 1e-3,
                        batch_size: int = 64,
                        seed: int = 42,
                        device: Optional[str] = None) -> tuple[ClinicalMLP, dict]:
    """Train ClinicalMLP for standalone evaluation."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_patients = train_df.drop_duplicates("submitter_id")
    val_patients   = val_df.drop_duplicates("submitter_id")

    X_train, state = preprocess_clinical(train_patients, fit=True)
    y_train = train_patients["label"].astype(int).values

    X_val, _ = preprocess_clinical(val_patients, fit=False, fitted_state=state)
    y_val   = val_patients["label"].astype(int).values

    # Class weights for imbalance
    counts = np.bincount(y_train, minlength=2).astype(float)
    weights = torch.tensor([1.0 / (c + 1e-6) for c in counts], dtype=torch.float32).to(device)

    model     = ClinicalMLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long).to(device)

    best_val_loss = float("inf")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "clinical_mlp_best.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        # Mini-batch training
        idx = torch.randperm(len(X_train_t))
        total_loss = 0.0
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            logits = model(X_train_t[batch_idx])
            loss   = criterion(logits, y_train_t[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)
        train_loss = total_loss / len(X_train_t)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"[clinical_model] MLP training complete. Best val_loss={best_val_loss:.4f}")
    return model, state


def predict_patients_mlp(model: ClinicalMLP, df: pd.DataFrame,
                          fitted_state: dict,
                          device: Optional[str] = None) -> dict[str, dict]:
    """Patient-level predictions from the ClinicalMLP."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    patients = df.drop_duplicates("submitter_id")
    X, _ = preprocess_clinical(patients, fit=False, fitted_state=fitted_state)
    X_t  = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_t)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    results = {}
    for sid, prob, label in zip(
        patients["submitter_id"].values,
        probs,
        patients["label"].astype(int).values,
    ):
        results[sid] = {
            "prob_idc": float(prob),
            "pred":     int(prob >= 0.5),
            "label":    int(label),
        }
    return results
