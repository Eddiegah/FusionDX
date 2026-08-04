# -*- coding: utf-8 -*-
"""
FusionDx -- Central Configuration
===================================
All tuneable parameters in one place.  Import from here rather than
hard-coding values in individual modules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).parent.parent
DATA_DIR     = BASE_DIR / "data"
RESULTS_DIR  = BASE_DIR / "results"
CLINICAL_DIR = DATA_DIR / "clinical"
SLIDES_DIR   = DATA_DIR / "slides"
TILES_DIR    = DATA_DIR / "tiles"
SYNTH_DIR    = DATA_DIR / "synthetic"

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
GDC_API           = "https://api.gdc.cancer.gov"
TCGA_BRCA_PROJECT = "TCGA-BRCA"
DEFAULT_SUBSET    = 150     # number of patients to download
RANDOM_SEED       = 42

# Tile extraction
TILE_SIZE      = 256   # pixels (width = height)
TILES_PER_SLIDE = 16   # tiles sampled per slide
TILE_LEVEL     = 1     # WSI pyramid level (0=full res; 1=quarter res typical)

# Patient-level split fractions (must sum to 1.0)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC = 1 - TRAIN_FRAC - VAL_FRAC (implicit)

# ---------------------------------------------------------------------------
# Image model (ResNet-34)
# ---------------------------------------------------------------------------
IMAGE_EPOCHS     = 10
IMAGE_BATCH_SIZE = 32
IMAGE_LR         = 1e-4
IMAGE_EMBED_DIM  = 512   # ResNet-34 penultimate layer output

# ---------------------------------------------------------------------------
# Clinical model
# ---------------------------------------------------------------------------
CLINICAL_LGBM_ROUNDS    = 300
CLINICAL_LGBM_LR        = 0.05
CLINICAL_MLP_EPOCHS     = 50
CLINICAL_MLP_LR         = 1e-3
CLINICAL_MLP_BATCH      = 64
CLINICAL_EMBED_DIM      = 64

# ---------------------------------------------------------------------------
# Fusion model
# ---------------------------------------------------------------------------
FUSION_WARMUP_EPOCHS   = 10
FUSION_FINETUNE_EPOCHS = 20
FUSION_BATCH_SIZE      = 32
FUSION_LR_WARMUP       = 1e-3
FUSION_LR_FINETUNE     = 1e-4
FUSION_ATTN_DIM        = 128
FUSION_OUT_DIM         = 256

# ---------------------------------------------------------------------------
# Synthetic data (for development/testing without real GDC data)
# ---------------------------------------------------------------------------
SYNTH_N_PATIENTS   = 120   # patients to generate
SYNTH_TILES_EACH   = 8     # tiles per patient (fewer than real for speed)
SYNTH_LABEL_RATIO  = 0.75  # fraction that are IDC (matches TCGA-BRCA approx.)
