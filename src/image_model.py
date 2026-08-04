"""
FusionDx — Image-Only Baseline Model
=====================================
A fine-tuned ResNet-34 trained on image tiles alone.

WHY ResNet-34?
--------------
- Well-studied, widely used in computational pathology literature as a
  baseline backbone (e.g. Kather et al. 2019, TCGA tile classification).
- Small enough to train on a laptop GPU (or CPU, slowly) with our tile subset.
- ImageNet pretraining gives a strong starting representation; we fine-tune
  the full network rather than only the head, since histopathology textures
  differ significantly from natural images and the lower layers benefit
  from domain adaptation.
- A larger backbone (ResNet-50, EfficientNet) would likely improve results
  with more data; note this as a future direction in the README.

AGGREGATION ACROSS TILES
-------------------------
A patient may have multiple tiles.  At inference time we average the softmax
probabilities across all tiles for a patient and take the argmax (mean pooling
across instances).  This is the simplest form of multiple-instance learning
(MIL) and is a reasonable baseline.  More sophisticated MIL (attention-based
pooling) is a natural extension.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class TileDataset(Dataset):
    """Single-tile dataset for image-only model training and inference."""

    def __init__(self, df, transform=None):
        self.records = df[["submitter_id", "tile_path", "label"]].dropna().to_dict("records")
        self.transform = transform or EVAL_TRANSFORM

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        image = Image.open(rec["tile_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(rec["label"])
        return image, label, rec["submitter_id"]


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class ImageModel(nn.Module):
    """
    ResNet-34 fine-tuned for binary tile classification.

    The final fully-connected layer is replaced with a linear head of size 2
    (binary).  We also expose `get_embedding()` which returns the 512-d
    feature vector before the classification head — used by the fusion model
    to build the image representation.
    """

    EMBED_DIM = 512  # ResNet-34 penultimate layer output dimension

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)

        # Remove the original FC layer; we'll add our own
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)
        self.classifier = nn.Linear(self.EMBED_DIM, num_classes)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return 512-d embedding (used by fusion model)."""
        emb = self.features(x)          # (B, 512, 1, 1)
        return emb.squeeze(-1).squeeze(-1)  # (B, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding(x)
        return self.classifier(emb)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_image_model(
    train_df,
    val_df,
    checkpoint_dir: Path = Path("results"),
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: Optional[str] = None,
) -> ImageModel:
    """
    Fine-tune ResNet-34 on training tiles.

    Returns the best-checkpoint model (chosen by validation loss).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[image_model] Training on device: {device}")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TileDataset(train_df, transform=TRAIN_TRANSFORM)
    val_ds   = TileDataset(val_df,   transform=EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Class weights to handle imbalance (IDC is majority class)
    labels = [int(r["label"]) for r in train_ds.records]
    class_counts = np.bincount(labels, minlength=2).astype(float)
    class_weights = torch.tensor(
        [1.0 / (c + 1e-6) for c in class_counts], dtype=torch.float32
    ).to(device)

    model = ImageModel(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_path = checkpoint_dir / "image_model_best.pth"

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for imgs, labels_batch, _ in train_loader:
            imgs, labels_batch = imgs.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_ds)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels_batch, _ in val_loader:
                imgs, labels_batch = imgs.to(device), labels_batch.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, labels_batch).item() * len(imgs)
        val_loss /= len(val_ds)

        scheduler.step()
        print(f"  Epoch {epoch:02d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"    ↳ New best checkpoint saved ({best_val_loss:.4f})")

    # Reload best weights
    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"[image_model] Training complete. Best val_loss={best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Patient-level inference (tile aggregation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_patients(model: ImageModel, df, batch_size: int = 64,
                     device: Optional[str] = None) -> dict[str, dict]:
    """
    Aggregate tile-level predictions to patient-level by mean-pooling softmax
    probabilities across all tiles belonging to each patient.

    Returns dict: submitter_id -> {"prob_idc": float, "pred": int, "label": int}
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    ds     = TileDataset(df, transform=EVAL_TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    patient_probs: dict[str, list[float]] = {}
    patient_labels: dict[str, int] = {}

    for imgs, labels_batch, sids in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(IDC)

        for sid, prob, label in zip(sids, probs, labels_batch.numpy()):
            patient_probs.setdefault(sid, []).append(float(prob))
            patient_labels[sid] = int(label)

    results = {}
    for sid, probs in patient_probs.items():
        avg_prob = float(np.mean(probs))
        results[sid] = {
            "prob_idc": avg_prob,
            "pred":     int(avg_prob >= 0.5),
            "label":    patient_labels[sid],
        }
    return results
