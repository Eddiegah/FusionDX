# -*- coding: utf-8 -*-
"""
FusionDx -- Full Training Orchestrator
=======================================
Run this after the data pipeline to train all three models and evaluate them.

Usage:
    python train_all.py                 # full run
    python train_all.py --skip-image    # skip image model (reuse saved checkpoint)
    python train_all.py --skip-clinical # skip clinical models (reuse saved)
    python train_all.py --eval-only     # skip all training, just re-run evaluation

This script:
  1. Loads the pre-built train/val/test splits (from data_pipeline.py).
  2. Trains the image-only model (ResNet-34).
  3. Trains the clinical-only LightGBM baseline.
  4. Trains the clinical MLP (needed for fusion embeddings).
  5. Trains the fusion model (cross-modal attention).
  6. Evaluates all three on the test set.
  7. Saves all predictions and the comparison report to results/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.image_model    import train_image_model, predict_patients, ImageModel
from src.clinical_model import (
    train_clinical_model_lgbm, predict_patients_lgbm,
    train_clinical_mlp,
    preprocess_clinical,
)
from src.fusion_model   import train_fusion_model, predict_patients_fusion, FusionModel
from src.evaluate       import compare_and_report, plot_attention_weights
from src.utils          import set_seed, save_json, load_json, get_device

RESULTS_DIR = Path("results")
DATA_DIR    = Path("data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for name in ("train.csv", "val.csv", "test.csv"):
        if not (DATA_DIR / name).exists():
            print(f"\nERROR: {DATA_DIR / name} not found.")
            print("Run the data pipeline first:")
            print("  python -m src.data_pipeline   (real data)")
            print("  python run_synthetic.py        (synthetic data)")
            sys.exit(1)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df   = pd.read_csv(DATA_DIR / "val.csv")
    test_df  = pd.read_csv(DATA_DIR / "test.csv")
    print(f"  Train: {train_df['submitter_id'].nunique()} patients, {len(train_df)} tiles")
    print(f"  Val  : {val_df['submitter_id'].nunique()} patients,  {len(val_df)} tiles")
    print(f"  Test : {test_df['submitter_id'].nunique()} patients,  {len(test_df)} tiles")
    return train_df, val_df, test_df


def _banner(msg: str) -> None:
    print("\n" + "=" * 62)
    print(msg)
    print("=" * 62)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    set_seed(42)
    device = get_device()

    _banner("FusionDx -- Training & Evaluation")
    print(f"  Device  : {device}")
    print(f"  Results : {RESULTS_DIR.resolve()}")

    # ---- Load splits -------------------------------------------------------
    _banner("Loading splits")
    train_df, val_df, test_df = _load_splits()

    # ---- Step 1: Image-only model ------------------------------------------
    _banner("Step 1 / 3 -- Image-only model (ResNet-34)")
    img_ckpt = RESULTS_DIR / "image_model_best.pth"
    img_preds_path = RESULTS_DIR / "image_predictions.json"

    if args.skip_image and img_ckpt.exists():
        print("  Skipping training -- loading saved checkpoint")
        image_model = ImageModel(num_classes=2, pretrained=False)
        image_model.load_state_dict(torch.load(img_ckpt, map_location=device))
    else:
        image_model = train_image_model(
            train_df, val_df,
            checkpoint_dir=RESULTS_DIR,
            epochs=10,
            batch_size=32,
            device=device,
        )

    image_preds = predict_patients(image_model, test_df, device=device)
    save_json(image_preds, img_preds_path)
    print(f"  Image predictions saved -- {len(image_preds)} test patients")

    # ---- Step 2: Clinical-only models (LightGBM + MLP) --------------------
    _banner("Step 2 / 3 -- Clinical-only models")
    state_path   = RESULTS_DIR / "clinical_state.json"
    clin_preds_path = RESULTS_DIR / "clinical_predictions.json"

    if args.skip_clinical and state_path.exists():
        print("  Skipping training -- loading saved clinical state")
        raw_state = load_json(state_path)
        clinical_state = {
            "col_medians":  np.array(raw_state["col_medians"]),
            "feature_cols": raw_state["feature_cols"],
        }
        import lightgbm as lgb
        lgbm_model = lgb.Booster(model_file=str(RESULTS_DIR / "clinical_lgbm.txt"))
    else:
        lgbm_model, clinical_state = train_clinical_model_lgbm(
            train_df, val_df, checkpoint_dir=RESULTS_DIR
        )
        save_json({
            "col_medians":  clinical_state["col_medians"].tolist(),
            "feature_cols": clinical_state["feature_cols"],
        }, state_path)
        # MLP is always trained when clinical is trained (needed for fusion)
        train_clinical_mlp(train_df, val_df, checkpoint_dir=RESULTS_DIR, epochs=50)

    clinical_preds = predict_patients_lgbm(lgbm_model, test_df, clinical_state)
    save_json(clinical_preds, clin_preds_path)
    print(f"  Clinical predictions saved -- {len(clinical_preds)} test patients")

    # ---- Step 3: Fusion model ---------------------------------------------
    _banner("Step 3 / 3 -- Fusion model (cross-modal attention)")
    fusion_ckpt  = RESULTS_DIR / "fusion_model_best.pth"
    fus_preds_path = RESULTS_DIR / "fusion_predictions.json"

    # Infer clinical input dimension from the fitted state
    X_tmp, _ = preprocess_clinical(
        train_df.drop_duplicates("submitter_id"), fit=False, fitted_state=clinical_state
    )
    clin_input_dim = X_tmp.shape[1]

    if args.skip_fusion and fusion_ckpt.exists():
        print("  Skipping training -- loading saved checkpoint")
        fusion_model = FusionModel(clinical_input_dim=clin_input_dim)
        fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location=device))
    else:
        fusion_model = train_fusion_model(
            train_df, val_df,
            clinical_state=clinical_state,
            checkpoint_dir=RESULTS_DIR,
            warmup_epochs=10,
            finetune_epochs=20,
            batch_size=32,
            device=device,
        )

    fusion_preds = predict_patients_fusion(fusion_model, test_df, clinical_state,
                                           device=device)
    save_json(fusion_preds, fus_preds_path)
    print(f"  Fusion predictions saved -- {len(fusion_preds)} test patients")

    # ---- Evaluation --------------------------------------------------------
    _banner("Evaluation -- honest comparison on test set")
    compare_and_report(image_preds, clinical_preds, fusion_preds,
                       out_dir=RESULTS_DIR)
    plot_attention_weights(fusion_preds, out_dir=RESULTS_DIR)

    _banner("Done")
    print("  results/comparison_report.md  -- full written report")
    print("  results/roc_curves.png        -- ROC curves")
    print("  results/comparison_chart.png  -- bar chart")
    print("  results/attention_weights.png -- per-patient attention")
    print()
    print("  Launch dashboard:  streamlit run app.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FusionDx training orchestrator")
    parser.add_argument("--skip-image",    action="store_true",
                        help="Skip image model training, use saved checkpoint")
    parser.add_argument("--skip-clinical", action="store_true",
                        help="Skip clinical model training, use saved state")
    parser.add_argument("--skip-fusion",   action="store_true",
                        help="Skip fusion model training, use saved checkpoint")
    parser.add_argument("--eval-only",     action="store_true",
                        help="Skip all training, just re-run evaluation from saved preds")
    args = parser.parse_args()

    if args.eval_only:
        # Load saved predictions and just re-run evaluation
        img_p  = load_json(RESULTS_DIR / "image_predictions.json")
        clin_p = load_json(RESULTS_DIR / "clinical_predictions.json")
        fus_p  = load_json(RESULTS_DIR / "fusion_predictions.json")
        if not all([img_p, clin_p, fus_p]):
            print("ERROR: Not all prediction files found in results/")
            print("Run without --eval-only first to generate them.")
            sys.exit(1)
        RESULTS_DIR.mkdir(exist_ok=True)
        compare_and_report(img_p, clin_p, fus_p, out_dir=RESULTS_DIR)
        plot_attention_weights(fus_p, out_dir=RESULTS_DIR)
    else:
        if args.skip_image:
            args.skip_fusion = False  # fusion needs fresh clinical state anyway
        main(args)
