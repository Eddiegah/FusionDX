# -*- coding: utf-8 -*-
"""
FusionDx — Multimodal Fusion Model with Cross-Modal Attention
==============================================================
This is the technical core of FusionDx.  Read these comments carefully —
they explain a real, current, well-documented approach to multimodal ML.

WHAT IS CROSS-MODAL ATTENTION AND WHY DOES IT MATTER?
-------------------------------------------------------
Simple late fusion (train two separate models, average their output
probabilities) does not allow the modalities to influence each other.
The image model never sees the clinical context; the clinical model never
sees the tissue appearance.  This is fundamentally limited: in real clinical
reasoning, a pathologist knows the patient's HER2 status before examining
the slide, and that prior shapes how they interpret subtle morphological
patterns.

Cross-modal attention allows genuine information exchange between modalities:

  1.  Each modality produces an embedding vector:
        • Image branch   → e_img    ∈ ℝ^{512}  (from ResNet-34)
        • Clinical branch → e_clin  ∈ ℝ^{64}   (from ClinicalMLP)

  2.  We project both into a shared attention space of dimension d_attn.

  3.  Attention weight for the image modality (α_img):
        α_img = σ( W_q_img · e_img  ⊙  W_k_clin · e_clin )
      where ⊙ is element-wise product followed by summation (dot-product
      attention), and σ is sigmoid.  Intuitively: how much does the clinical
      context 'agree with' or 'activate' what the image encoder found?

  4.  Attention weight for the clinical modality (α_clin) is computed
      symmetrically — the image embedding modulates how much the clinical
      features are relied upon.

  5.  The attended representation is:
        z = LayerNorm(α_img · W_v_img · e_img  +  α_clin · W_v_clin · e_clin)

  6.  A shared classification head maps z → logits.

WHAT DO THE ATTENTION WEIGHTS TELL US?
---------------------------------------
α_img and α_clin are scalar values in [0, 1] for each sample.  A high α_img
means the fusion model leaned heavily on the image branch for that prediction;
a high α_clin means it leaned on clinical data.  These weights are the
interpretability signal visualized in the dashboard — a genuine, sample-
specific multimodal interpretability signal not available in either
single-modality model.

This is related to (but simpler than) the cross-modal attention in
Transformer-based fusion models (e.g. Chen et al., "Multimodal Co-Attention
Transformer for Survival Prediction", ICCV 2021).  The simplification makes
it tractable to train from scratch on a small dataset while retaining the
key property: bidirectional, learned, sample-level modality weighting.

TRAINING STRATEGY
-----------------
We use a two-phase approach:

  Phase 1 — Warm-up (20 epochs):
    Freeze the ResNet backbone and the ClinicalMLP encoder; train only
    the attention and classification layers.  This stabilizes the fusion
    head before the unimodal encoders are perturbed.

  Phase 2 — End-to-end (remaining epochs):
    Unfreeze all parameters and fine-tune with a lower learning rate.
    The unimodal encoders adapt jointly with the fusion layer.

This staged approach is common when one modality encoder is much larger than
the other (ResNet-34 at 21M params vs. ClinicalMLP at ~10K params) — without
warm-up the attention layer would overfit to whichever encoder happened to
produce more discriminative early gradients.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.image_model    import ImageModel, EVAL_TRANSFORM, TRAIN_TRANSFORM
from src.clinical_model import ClinicalMLP, preprocess_clinical
from PIL import Image


# ---------------------------------------------------------------------------
# Multimodal dataset — pairs a tile with its patient's clinical features
# ---------------------------------------------------------------------------

class FusionDataset(Dataset):
    """
    Returns (image_tensor, clinical_tensor, label, submitter_id) for each tile.
    Clinical features are the same for all tiles from the same patient.
    """

    def __init__(self, df, clinical_state: dict, transform=None):
        self.df = df.dropna(subset=["tile_path", "label"]).reset_index(drop=True)
        self.transform = transform or EVAL_TRANSFORM
        self.clinical_state = clinical_state

        # Pre-compute patient → clinical feature vector (avoid recomputing per tile)
        patients = self.df.drop_duplicates("submitter_id")
        X_clin, _ = preprocess_clinical(patients, fit=False, fitted_state=clinical_state)
        self.sid_to_clin: dict[str, np.ndarray] = {
            sid: X_clin[i] for i, sid in enumerate(patients["submitter_id"].values)
        }

        self.records = self.df[["submitter_id", "tile_path", "label"]].to_dict("records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        sid   = rec["submitter_id"]
        label = int(rec["label"])

        # Image
        img = Image.open(rec["tile_path"]).convert("RGB")
        img_t = self.transform(img)

        # Clinical features
        clin_t = torch.tensor(self.sid_to_clin[sid], dtype=torch.float32)

        return img_t, clin_t, label, sid


# ---------------------------------------------------------------------------
# Cross-Modal Attention Fusion Module
# ---------------------------------------------------------------------------

class CrossModalAttention(nn.Module):
    """
    Implements the bidirectional cross-modal attention described in the module
    docstring above.

    Parameters
    ----------
    img_dim   : Dimension of image embedding (default 512 from ResNet-34)
    clin_dim  : Dimension of clinical embedding (default 64 from ClinicalMLP)
    d_attn    : Projection dimension for attention computation
    out_dim   : Dimension of the fused output vector
    """

    def __init__(self,
                 img_dim:  int = ImageModel.EMBED_DIM,    # 512
                 clin_dim: int = ClinicalMLP.EMBED_DIM,   # 64
                 d_attn:   int = 128,
                 out_dim:  int = 256):
        super().__init__()

        # --- Query / Key projections for dot-product attention ---
        # Image query attends to clinical key (and vice versa)
        self.W_q_img  = nn.Linear(img_dim,  d_attn, bias=False)
        self.W_k_clin = nn.Linear(clin_dim, d_attn, bias=False)

        self.W_q_clin = nn.Linear(clin_dim, d_attn, bias=False)
        self.W_k_img  = nn.Linear(img_dim,  d_attn, bias=False)

        # --- Value projections (project each modality to shared out_dim) ---
        self.W_v_img  = nn.Linear(img_dim,  out_dim)
        self.W_v_clin = nn.Linear(clin_dim, out_dim)

        # --- Output normalisation ---
        self.norm = nn.LayerNorm(out_dim)

        # Scaling factor (stabilises dot-product attention)
        self.scale = d_attn ** -0.5

    def forward(self,
                e_img:  torch.Tensor,   # (B, img_dim)
                e_clin: torch.Tensor,   # (B, clin_dim)
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z        : (B, out_dim) — fused representation
        alpha_img  : (B,) — scalar attention weight on image modality
        alpha_clin : (B,) — scalar attention weight on clinical modality
        """
        # --- Compute attention weights ---
        # α_img: how much the clinical context activates the image evidence
        #   dot product between image query and clinical key → scalar per sample
        q_img  = self.W_q_img(e_img)    # (B, d_attn)
        k_clin = self.W_k_clin(e_clin)  # (B, d_attn)
        alpha_img = torch.sigmoid(
            (q_img * k_clin).sum(dim=-1, keepdim=True) * self.scale
        )  # (B, 1)

        # α_clin: how much the image evidence activates the clinical features
        q_clin = self.W_q_clin(e_clin)  # (B, d_attn)
        k_img  = self.W_k_img(e_img)    # (B, d_attn)
        alpha_clin = torch.sigmoid(
            (q_clin * k_img).sum(dim=-1, keepdim=True) * self.scale
        )  # (B, 1)

        # --- Attended value computation ---
        v_img  = self.W_v_img(e_img)    # (B, out_dim)
        v_clin = self.W_v_clin(e_clin)  # (B, out_dim)

        # Weighted sum of value projections
        z = self.norm(alpha_img * v_img + alpha_clin * v_clin)  # (B, out_dim)

        return z, alpha_img.squeeze(-1), alpha_clin.squeeze(-1)


# ---------------------------------------------------------------------------
# Full Fusion Model
# ---------------------------------------------------------------------------

class FusionModel(nn.Module):
    """
    End-to-end multimodal fusion model combining ResNet-34 (image) and
    ClinicalMLP with cross-modal attention.

    Architecture summary:
      ResNet-34 backbone → 512-d image embedding  ─┐
                                                    ├─ CrossModalAttention → 256-d fused → FC(2)
      ClinicalMLP encoder → 64-d clinical embedding ┘
    """

    def __init__(self,
                 img_dim:  int = ImageModel.EMBED_DIM,
                 clin_dim: int = ClinicalMLP.EMBED_DIM,
                 d_attn:   int = 128,
                 fused_dim: int = 256,
                 num_classes: int = 2,
                 clinical_input_dim: int = 11):
        super().__init__()

        self.image_encoder    = ImageModel(num_classes=num_classes, pretrained=True)
        self.clinical_encoder = ClinicalMLP(input_dim=clinical_input_dim)
        self.cross_attn       = CrossModalAttention(img_dim, clin_dim, d_attn, fused_dim)
        self.classifier       = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(fused_dim, num_classes),
        )

    def forward(self,
                x_img:  torch.Tensor,   # (B, 3, 256, 256)
                x_clin: torch.Tensor,   # (B, clin_features)
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits     : (B, 2) — class logits
        alpha_img  : (B,) — per-sample image attention weight  [interpretability]
        alpha_clin : (B,) — per-sample clinical attention weight [interpretability]
        """
        e_img  = self.image_encoder.get_embedding(x_img)   # (B, 512)
        e_clin = self.clinical_encoder.get_embedding(x_clin)  # (B, 64)

        z, alpha_img, alpha_clin = self.cross_attn(e_img, e_clin)
        logits = self.classifier(z)

        return logits, alpha_img, alpha_clin


# ---------------------------------------------------------------------------
# Two-phase training
# ---------------------------------------------------------------------------

def _freeze_encoders(model: FusionModel) -> None:
    """Freeze ResNet backbone and ClinicalMLP encoder (warm-up phase)."""
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    for p in model.clinical_encoder.parameters():
        p.requires_grad = False


def _unfreeze_encoders(model: FusionModel) -> None:
    """Unfreeze all parameters (end-to-end fine-tuning phase)."""
    for p in model.parameters():
        p.requires_grad = True


def train_fusion_model(
    train_df,
    val_df,
    clinical_state: dict,
    checkpoint_dir: Path = Path("results"),
    warmup_epochs: int = 10,
    finetune_epochs: int = 20,
    batch_size: int = 32,
    lr_warmup: float = 1e-3,
    lr_finetune: float = 1e-4,
    device: Optional[str] = None,
) -> FusionModel:
    """
    Train the fusion model in two phases.

    Phase 1 — warm-up (warmup_epochs):
      Only attention + classifier layers are trained.
      Both unimodal encoders are frozen.
      Learning rate: lr_warmup (higher, since only ~500K params are active).

    Phase 2 — end-to-end (finetune_epochs):
      All parameters unfrozen.
      Lower learning rate (lr_finetune) to avoid destabilising pretrained weights.

    Returns best-checkpoint FusionModel (chosen by validation loss).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fusion_model] Training on device: {device}")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Infer clinical input dimension from preprocess
    _patients = train_df.drop_duplicates("submitter_id")
    X_tmp, _ = preprocess_clinical(_patients, fit=False, fitted_state=clinical_state)
    clin_dim_input = X_tmp.shape[1]

    train_ds = FusionDataset(train_df, clinical_state, transform=TRAIN_TRANSFORM)
    val_ds   = FusionDataset(val_df,   clinical_state, transform=EVAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = FusionModel(clinical_input_dim=clin_dim_input).to(device)

    # Load pretrained unimodal weights if available
    img_ckpt  = checkpoint_dir / "image_model_best.pth"
    clin_ckpt = checkpoint_dir / "clinical_mlp_best.pth"
    if img_ckpt.exists():
        model.image_encoder.load_state_dict(
            torch.load(img_ckpt, map_location=device), strict=False)
        print(f"  Loaded image encoder weights from {img_ckpt}")
    if clin_ckpt.exists():
        model.clinical_encoder.load_state_dict(
            torch.load(clin_ckpt, map_location=device), strict=False)
        print(f"  Loaded clinical encoder weights from {clin_ckpt}")

    # Class weights
    labels = [int(r["label"]) for r in train_ds.records]
    counts = np.bincount(labels, minlength=2).astype(float)
    class_weights = torch.tensor(
        [1.0 / (c + 1e-6) for c in counts], dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    best_path = checkpoint_dir / "fusion_model_best.pth"

    def run_epoch(loader, training: bool, optimizer=None) -> float:
        model.train() if training else model.eval()
        total_loss = 0.0
        with torch.set_grad_enabled(training):
            for imgs, clins, lbls, _ in loader:
                imgs  = imgs.to(device)
                clins = clins.to(device)
                lbls  = lbls.to(device)
                logits, _, _ = model(imgs, clins)
                loss = criterion(logits, lbls)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * len(imgs)
        return total_loss / len(loader.dataset)

    # ---- Phase 1: Warm-up ------------------------------------------------
    print(f"\n[Phase 1] Warm-up — freezing encoders ({warmup_epochs} epochs)")
    _freeze_encoders(model)
    opt_warmup = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_warmup, weight_decay=1e-4
    )
    sched_warmup = torch.optim.lr_scheduler.CosineAnnealingLR(opt_warmup, T_max=warmup_epochs)

    for epoch in range(1, warmup_epochs + 1):
        train_loss = run_epoch(train_loader, training=True, optimizer=opt_warmup)
        val_loss   = run_epoch(val_loader,   training=False)
        sched_warmup.step()
        print(f"  [WU] Epoch {epoch:02d}/{warmup_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"    ↳ Checkpoint saved ({best_val_loss:.4f})")

    # ---- Phase 2: End-to-end fine-tuning ---------------------------------
    print(f"\n[Phase 2] End-to-end fine-tuning ({finetune_epochs} epochs)")
    _unfreeze_encoders(model)
    opt_ft = torch.optim.AdamW(model.parameters(), lr=lr_finetune, weight_decay=1e-4)
    sched_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=finetune_epochs)

    for epoch in range(1, finetune_epochs + 1):
        train_loss = run_epoch(train_loader, training=True, optimizer=opt_ft)
        val_loss   = run_epoch(val_loader,   training=False)
        sched_ft.step()
        print(f"  [FT] Epoch {epoch:02d}/{finetune_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"    ↳ Checkpoint saved ({best_val_loss:.4f})")

    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"\n[fusion_model] Training complete. Best val_loss={best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Patient-level inference with attention weights
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_patients_fusion(
    model: FusionModel,
    df,
    clinical_state: dict,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> dict[str, dict]:
    """
    Aggregate tile-level fusion predictions to patient level.

    For each patient we average:
      - softmax P(IDC) across tiles
      - alpha_img across tiles
      - alpha_clin across tiles

    The per-patient mean alpha_img and alpha_clin are the interpretability
    signal: which modality did the fusion model rely on more for this patient?
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    ds     = FusionDataset(df, clinical_state, transform=EVAL_TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    patient_probs:      dict[str, list[float]] = {}
    patient_alpha_img:  dict[str, list[float]] = {}
    patient_alpha_clin: dict[str, list[float]] = {}
    patient_labels:     dict[str, int] = {}

    for imgs, clins, lbls, sids in loader:
        imgs  = imgs.to(device)
        clins = clins.to(device)
        logits, alpha_img, alpha_clin = model(imgs, clins)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        a_img  = alpha_img.cpu().numpy()
        a_clin = alpha_clin.cpu().numpy()

        for sid, prob, ai, ac, label in zip(sids, probs, a_img, a_clin, lbls.numpy()):
            patient_probs     .setdefault(sid, []).append(float(prob))
            patient_alpha_img .setdefault(sid, []).append(float(ai))
            patient_alpha_clin.setdefault(sid, []).append(float(ac))
            patient_labels[sid] = int(label)

    results = {}
    for sid in patient_probs:
        avg_prob  = float(np.mean(patient_probs[sid]))
        avg_ai    = float(np.mean(patient_alpha_img[sid]))
        avg_ac    = float(np.mean(patient_alpha_clin[sid]))
        # Normalise to sum to 1 for cleaner display
        total     = avg_ai + avg_ac + 1e-8
        results[sid] = {
            "prob_idc":         avg_prob,
            "pred":             int(avg_prob >= 0.5),
            "label":            patient_labels[sid],
            "alpha_img":        avg_ai / total,   # normalised: ≈ image reliance
            "alpha_clin":       avg_ac / total,   # normalised: ≈ clinical reliance
            "alpha_img_raw":    avg_ai,
            "alpha_clin_raw":   avg_ac,
        }
    return results
