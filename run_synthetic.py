# -*- coding: utf-8 -*-
"""
FusionDx -- Full Pipeline on Synthetic Data
=============================================
Runs the complete train + evaluate + save pipeline using synthetic (fake)
patient data.  No GDC account, no OpenSlide, no downloads required.

Use this to:
  - Verify the full code path works end-to-end
  - Develop and test model architecture changes
  - Check the dashboard renders correctly

Results produced here have NO clinical meaning.
Only results from real TCGA-BRCA data should be reported.

Usage:
    venv\\Scripts\\activate
    python run_synthetic.py
"""

import json
from pathlib import Path

import pandas as pd

from src.synthetic_data  import generate_synthetic_dataset, split_synthetic
from src.image_model     import train_image_model, predict_patients
from src.clinical_model  import (
    train_clinical_model_lgbm, predict_patients_lgbm,
    train_clinical_mlp, predict_patients_mlp,
)
from src.fusion_model    import train_fusion_model, predict_patients_fusion
from src.evaluate        import compare_and_report, plot_attention_weights
from src.utils           import set_seed, save_json

RESULTS_DIR = Path("results")
DATA_DIR    = Path("data")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("FusionDx -- Synthetic Pipeline Run")
    print("NOTE: Results below have NO clinical meaning.")
    print("=" * 65)

    # Fix all random seeds for reproducibility
    from src.utils import set_seed
    set_seed(42)

    # ---- Generate synthetic data -------------------------------------------
    print("\n[Step 1] Generating synthetic data ...")
    dataset = generate_synthetic_dataset(
        n_patients=120,
        tiles_per_patient=8,
        seed=42,
    )
    train_df, val_df, test_df = split_synthetic(dataset)

    # ---- Image-only model ---------------------------------------------------
    print("\n[Step 2] Training image-only model (ResNet-34) ...")
    image_model = train_image_model(
        train_df, val_df,
        checkpoint_dir=RESULTS_DIR,
        epochs=5,           # fewer epochs for synthetic speed
        batch_size=32,
    )
    image_preds = predict_patients(image_model, test_df)
    save_json(image_preds, RESULTS_DIR / "image_predictions.json")
    print(f"  Image predictions: {len(image_preds)} test patients")

    # ---- Clinical-only model ------------------------------------------------
    print("\n[Step 3] Training clinical-only models ...")
    lgbm_model, clinical_state = train_clinical_model_lgbm(
        train_df, val_df, checkpoint_dir=RESULTS_DIR
    )

    # Serialise state for dashboard use
    state_out = {
        "col_medians":  clinical_state["col_medians"].tolist(),
        "feature_cols": clinical_state["feature_cols"],
    }
    save_json(state_out, RESULTS_DIR / "clinical_state.json")

    clinical_preds = predict_patients_lgbm(lgbm_model, test_df, clinical_state)
    save_json(clinical_preds, RESULTS_DIR / "clinical_predictions.json")
    print(f"  Clinical predictions: {len(clinical_preds)} test patients")

    # MLP for fusion embeddings
    mlp_model, _ = train_clinical_mlp(
        train_df, val_df,
        checkpoint_dir=RESULTS_DIR,
        epochs=30,
    )

    # ---- Fusion model -------------------------------------------------------
    print("\n[Step 4] Training fusion model (cross-modal attention) ...")
    fusion_model = train_fusion_model(
        train_df, val_df,
        clinical_state=clinical_state,
        checkpoint_dir=RESULTS_DIR,
        warmup_epochs=5,
        finetune_epochs=10,
        batch_size=32,
    )
    fusion_preds = predict_patients_fusion(fusion_model, test_df, clinical_state)
    save_json(fusion_preds, RESULTS_DIR / "fusion_predictions.json")
    print(f"  Fusion predictions: {len(fusion_preds)} test patients")

    # ---- Evaluation ---------------------------------------------------------
    print("\n[Step 5] Evaluating all models ...")
    compare_and_report(image_preds, clinical_preds, fusion_preds)
    plot_attention_weights(fusion_preds)

    print("\n" + "=" * 65)
    print("Synthetic run complete.")
    print("  results/comparison_report.md  -- written comparison")
    print("  results/roc_curves.png         -- ROC plot")
    print("  results/comparison_chart.png   -- bar chart")
    print("  results/attention_weights.png  -- per-patient attention")
    print()
    print("Launch dashboard:  streamlit run app.py")
    print()
    print("REMINDER: These results use SYNTHETIC data and have no")
    print("clinical meaning. Run the real pipeline after installing")
    print("OpenSlide and downloading TCGA-BRCA data from GDC.")
    print("=" * 65)


if __name__ == "__main__":
    main()
