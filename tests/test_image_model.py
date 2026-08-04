# -*- coding: utf-8 -*-
"""Tests for the image model."""

import torch
import pytest

from src.image_model import ImageModel, TileDataset, predict_patients, EVAL_TRANSFORM
from src.synthetic_data import generate_synthetic_dataset, split_synthetic


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("img")
    df = generate_synthetic_dataset(n_patients=40, tiles_per_patient=4, seed=2, out_data_dir=tmp)
    train_df, val_df, test_df = split_synthetic(df, seed=2, out_dir=tmp)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

def test_image_model_forward_shape():
    model = ImageModel(num_classes=2, pretrained=False)
    x = torch.randn(4, 3, 256, 256)
    logits = model(x)
    assert logits.shape == (4, 2)


def test_image_model_embedding_shape():
    model = ImageModel(num_classes=2, pretrained=False)
    x = torch.randn(4, 3, 256, 256)
    emb = model.get_embedding(x)
    assert emb.shape == (4, ImageModel.EMBED_DIM)


def test_image_model_embedding_differentiable():
    model = ImageModel(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    emb = model.get_embedding(x)
    emb.sum().backward()  # must not raise


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

def test_tile_dataset_length(splits):
    train_df, _, _ = splits
    ds = TileDataset(train_df, transform=EVAL_TRANSFORM)
    # All rows with valid tile_path and label
    assert len(ds) == len(train_df.dropna(subset=["tile_path", "label"]))


def test_tile_dataset_item_shapes(splits):
    train_df, _, _ = splits
    ds = TileDataset(train_df, transform=EVAL_TRANSFORM)
    img, label, sid = ds[0]
    assert img.shape == (3, 256, 256), f"Expected (3,256,256), got {img.shape}"
    assert label in (0, 1)
    assert isinstance(sid, str)


# ---------------------------------------------------------------------------
# Training test (minimal epochs, no pretrained weights to save CI time)
# ---------------------------------------------------------------------------

def test_image_model_trains(splits, tmp_path):
    from src.image_model import train_image_model
    train_df, val_df, test_df = splits
    model = train_image_model(
        train_df, val_df,
        checkpoint_dir=tmp_path,
        epochs=2,
        batch_size=16,
    )
    assert isinstance(model, ImageModel)


def test_image_model_predicts(splits, tmp_path):
    from src.image_model import train_image_model
    train_df, val_df, test_df = splits
    model = train_image_model(
        train_df, val_df,
        checkpoint_dir=tmp_path,
        epochs=1,
        batch_size=16,
    )
    preds = predict_patients(model, test_df)
    test_patients = set(test_df["submitter_id"].unique())
    assert set(preds.keys()) == test_patients
    for sid, result in preds.items():
        assert 0.0 <= result["prob_idc"] <= 1.0
        assert result["pred"] in (0, 1)
