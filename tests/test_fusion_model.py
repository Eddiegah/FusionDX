# -*- coding: utf-8 -*-
"""Tests for the fusion model and cross-modal attention."""

import torch
import pytest

from src.fusion_model import (
    CrossModalAttention,
    FusionModel,
    FusionDataset,
    train_fusion_model,
    predict_patients_fusion,
)
from src.image_model    import ImageModel
from src.clinical_model import ClinicalMLP
from src.synthetic_data import generate_synthetic_dataset, split_synthetic
from src.clinical_model import preprocess_clinical


@pytest.fixture(scope="module")
def splits_and_state(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fusion")
    df = generate_synthetic_dataset(n_patients=40, tiles_per_patient=4, seed=3, out_data_dir=tmp)
    train_df, val_df, test_df = split_synthetic(df, seed=3, out_dir=tmp)
    # Build clinical state from train
    X, state = preprocess_clinical(train_df.drop_duplicates("submitter_id"), fit=True)
    return train_df, val_df, test_df, state


# ---------------------------------------------------------------------------
# CrossModalAttention architecture tests
# ---------------------------------------------------------------------------

def test_cross_attn_output_shapes():
    attn = CrossModalAttention(img_dim=512, clin_dim=64, d_attn=128, out_dim=256)
    e_img  = torch.randn(8, 512)
    e_clin = torch.randn(8, 64)
    z, a_img, a_clin = attn(e_img, e_clin)
    assert z.shape == (8, 256),    f"fused z: expected (8,256), got {z.shape}"
    assert a_img.shape == (8,),    f"alpha_img: expected (8,), got {a_img.shape}"
    assert a_clin.shape == (8,),   f"alpha_clin: expected (8,), got {a_clin.shape}"


def test_cross_attn_weights_in_0_1():
    """Attention weights (pre-normalisation) should be in [0,1] since we use sigmoid."""
    attn = CrossModalAttention(img_dim=512, clin_dim=64)
    e_img  = torch.randn(16, 512)
    e_clin = torch.randn(16, 64)
    _, a_img, a_clin = attn(e_img, e_clin)
    assert (a_img  >= 0).all() and (a_img  <= 1).all(), "alpha_img out of [0,1]"
    assert (a_clin >= 0).all() and (a_clin <= 1).all(), "alpha_clin out of [0,1]"


def test_cross_attn_differentiable():
    """Gradients must flow through the attention mechanism."""
    attn = CrossModalAttention(img_dim=512, clin_dim=64)
    e_img  = torch.randn(4, 512, requires_grad=True)
    e_clin = torch.randn(4, 64,  requires_grad=True)
    z, a_img, a_clin = attn(e_img, e_clin)
    (z.sum() + a_img.sum() + a_clin.sum()).backward()
    assert e_img.grad is not None
    assert e_clin.grad is not None


def test_cross_attn_different_inputs_give_different_weights():
    """Two different input pairs should produce different attention weights."""
    attn = CrossModalAttention(img_dim=512, clin_dim=64)
    attn.eval()
    with torch.no_grad():
        _, a1, _ = attn(torch.randn(1, 512), torch.randn(1, 64))
        _, a2, _ = attn(torch.randn(1, 512), torch.randn(1, 64))
    # Very unlikely to be identical with random inputs
    assert not torch.allclose(a1, a2), "Attention weights identical for different inputs"


# ---------------------------------------------------------------------------
# FusionModel architecture tests
# ---------------------------------------------------------------------------

def test_fusion_model_forward_shapes():
    model = FusionModel(clinical_input_dim=11)
    x_img  = torch.randn(4, 3, 256, 256)
    x_clin = torch.randn(4, 11)
    logits, a_img, a_clin = model(x_img, x_clin)
    assert logits.shape  == (4, 2)
    assert a_img.shape   == (4,)
    assert a_clin.shape  == (4,)


def test_fusion_model_attention_weights_sum_to_approx_one():
    """
    After normalisation in predict_patients_fusion, alpha_img + alpha_clin ~ 1.
    Here we check that raw sigmoid outputs are both in (0,1) and the model
    can produce sample-specific weightings.
    """
    model = FusionModel(clinical_input_dim=11)
    model.eval()
    x_img  = torch.randn(8, 3, 256, 256)
    x_clin = torch.randn(8, 11)
    with torch.no_grad():
        _, a_img, a_clin = model(x_img, x_clin)
    assert (a_img  >= 0).all() and (a_img  <= 1).all()
    assert (a_clin >= 0).all() and (a_clin <= 1).all()


# ---------------------------------------------------------------------------
# FusionDataset tests
# ---------------------------------------------------------------------------

def test_fusion_dataset_length(splits_and_state):
    train_df, _, _, state = splits_and_state
    ds = FusionDataset(train_df, state)
    assert len(ds) == len(train_df.dropna(subset=["tile_path", "label"]))


def test_fusion_dataset_item(splits_and_state):
    train_df, _, _, state = splits_and_state
    ds = FusionDataset(train_df, state)
    img_t, clin_t, label, sid = ds[0]
    assert img_t.shape == (3, 256, 256)
    assert clin_t.ndim == 1
    assert label in (0, 1)
    assert isinstance(sid, str)


# ---------------------------------------------------------------------------
# Training and inference tests (minimal epochs)
# ---------------------------------------------------------------------------

def test_fusion_model_trains(splits_and_state, tmp_path):
    train_df, val_df, _, state = splits_and_state
    model = train_fusion_model(
        train_df, val_df,
        clinical_state=state,
        checkpoint_dir=tmp_path,
        warmup_epochs=1,
        finetune_epochs=1,
        batch_size=16,
    )
    assert isinstance(model, FusionModel)


def test_fusion_model_predicts_with_attention(splits_and_state, tmp_path):
    train_df, val_df, test_df, state = splits_and_state
    model = train_fusion_model(
        train_df, val_df,
        clinical_state=state,
        checkpoint_dir=tmp_path,
        warmup_epochs=1,
        finetune_epochs=1,
        batch_size=16,
    )
    preds = predict_patients_fusion(model, test_df, state)

    test_patients = set(test_df["submitter_id"].unique())
    assert set(preds.keys()) == test_patients

    for sid, result in preds.items():
        assert 0.0 <= result["prob_idc"] <= 1.0
        assert result["pred"] in (0, 1)
        # Normalised attention weights should be present
        assert "alpha_img"  in result
        assert "alpha_clin" in result
        total = result["alpha_img"] + result["alpha_clin"]
        assert abs(total - 1.0) < 0.01, f"Attention weights don't sum to ~1: {total}"
