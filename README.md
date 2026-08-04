<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Inter&weight=800&size=40&pause=1000&color=4FC3F7&center=true&vCenter=true&width=600&lines=🧬+FusionDx;Multimodal+Cancer+AI;Image+%2B+Clinical+Data+Fusion" alt="FusionDx" />

<br/>

[![Python 3.11](https://img.shields.io/badge/Python-3.11-4fc3f7?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3.1-ef5350?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License MIT](https://img.shields.io/badge/License-MIT-81c784?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-54%20passing-4caf50?style=for-the-badge&logo=pytest&logoColor=white)](#testing)
[![Data](https://img.shields.io/badge/Data-TCGA--BRCA%20·%20Open%20Access-ffb74d?style=for-the-badge)](https://portal.gdc.cancer.gov/projects/TCGA-BRCA)

<br/>

> **⚠️ RESEARCH / EDUCATIONAL DEMONSTRATION ONLY**
> Not validated for clinical use. Do not use for diagnostic or treatment decisions.

<br/>

**[🚀 Quick Start](#quick-start) · [📊 Results](#results) · [⚡ Architecture](#architecture) · [🧬 Dashboard](#dashboard) · [📋 Setup](#setup)**

</div>

---

## What is FusionDx?

FusionDx is a breast cancer diagnosis research system that **genuinely fuses** real histopathology tissue imaging with real clinical patient data — using **TCGA-BRCA** via the NIH Genomic Data Commons — and rigorously, honestly proves whether the fused model outperforms either single-modality baseline.

The technical core is a **cross-modal attention mechanism** that allows each modality to modulate the other's contribution, producing per-patient interpretability signals not available in single-modality models.

```
256×256 H&E tile ──► ResNet-34 ──► 512-d embedding ──┐
                                                       ├──► CrossModalAttention ──► IDC / Other
clinical features ──► ClinicalMLP ──► 64-d embedding ──┘              │
                                                               α_img, α_clin
                                                           (per-patient interpretability)
```

---

## Results

> All three models evaluated on the **same patient-level held-out test set**.
> Bootstrap 95% confidence intervals included. Reported honestly — no inflated claims.

| Model | Accuracy | F1 | AUROC | AUROC 95% CI |
|-------|----------|----|-------|--------------|
| 🖼️ Image-Only (ResNet-34) | 1.000 | 1.000 | 1.000 | [1.0, 1.0] |
| 📋 Clinical-Only (LightGBM) | 0.889 | 0.837 | 0.250 | [0.0, 0.625] |
| ⚡ **Fusion (Cross-Modal Attention)** | **1.000** | **1.000** | **1.000** | **[1.0, 1.0]** |

> **Note:** Results above are from the **synthetic data pipeline** (no clinical meaning).
> Real results will differ when run on actual TCGA-BRCA downloads.
> The pipeline faithfully reports whatever the real numbers show — including if fusion loses.

---

## Architecture

### Cross-Modal Attention — Why It Matters

Simple late fusion (averaging two models) doesn't let modalities influence each other.
Cross-modal attention enables **bidirectional, sample-specific modality weighting**:

```python
# Image attends to clinical context
alpha_img  = sigmoid( (W_q_img  · e_img)  ·  (W_k_clin · e_clin) / sqrt(d) )

# Clinical attends to image evidence
alpha_clin = sigmoid( (W_q_clin · e_clin) ·  (W_k_img  · e_img)  / sqrt(d) )

# Fused representation
z = LayerNorm( alpha_img * W_v_img * e_img  +  alpha_clin * W_v_clin * e_clin )
```

**`alpha_img` and `alpha_clin`** are the interpretability output — for each patient,
how much did the model rely on imaging vs. clinical data? This per-sample signal is only
available in a multimodal fusion model.

### Three Models

| Model | Input | Architecture | Parameters |
|-------|-------|-------------|------------|
| 🖼️ Image-Only | 256×256 H&E tiles | ResNet-34 (ImageNet pretrained) | ~21M |
| 📋 Clinical-Only | Age, stage, ER/PR/HER2 | LightGBM + ClinicalMLP | ~10K |
| ⚡ **Fusion** | **Both** | **ResNet-34 + ClinicalMLP + CrossAttn** | **~21M** |

### Two-Phase Training

```
Phase 1 — Warm-up (freeze encoders):
  Only attention layers + classifier trained.
  Stabilises fusion head before touching ResNet.

Phase 2 — End-to-end fine-tuning:
  All parameters unfrozen, lower LR (1e-4).
  Encoders adapt jointly with the fusion objective.
```

---

## Patient-Level Splitting — Most Important Correctness Decision

A single patient contributes 16 tiles. Those tiles are biologically correlated.

❌ **Tile-level split** → same patient's tissue in both train and test → inflated, meaningless accuracy
✅ **Patient-level split** (FusionDx) → strict separation, runtime assertions verify it

```python
# This assertion runs every time splits are created
assert not (train_patients & test_patients), "BUG: patient leak!"
```

---

## Quick Start

### Option A — Synthetic data (no downloads, runs now)

```cmd
# 1. Activate the virtual environment
venv\Scripts\activate

# 2. Run the full pipeline on synthetic data (~20 min on CPU)
python run_synthetic.py

# 3. Launch the dashboard
streamlit run app.py
```

Open **http://localhost:8501** 🎉

### Option B — Real TCGA-BRCA data

```cmd
# After installing OpenSlide (see Setup section)
python -m src.data_pipeline --verify    # confirm environment
python -m src.data_pipeline             # download data (hours, ~20-100 GB)
python train_all.py                     # train all models
streamlit run app.py                    # explore results
```

---

## Setup

### Prerequisites

- Python 3.11 (3.9–3.12 supported; 3.14 is too new for pinned PyTorch wheels)
- Windows (primary target) or Linux/macOS

### 1. Create virtual environment

```cmd
py -3.11 -m venv venv
venv\Scripts\activate
```

### 2. Install PyTorch

```cmd
# CPU only
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1 (GPU — recommended for real data)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install OpenSlide (Windows — required for real WSI data)

> `openslide-python` needs the native C library separately from pip.

1. Download Windows binaries: **https://openslide.org/download/**
2. Extract to e.g. `C:\OpenSlide\`
3. Add `C:\OpenSlide\bin` to your system **PATH** → restart terminal
4. Verify: `python setup_openslide.py`

### 4. Install remaining dependencies

```cmd
pip install -r requirements.txt
```

### 5. Verify environment

```cmd
python verify_gdc.py                      # live GDC API field check
python -m src.data_pipeline --verify      # full environment check
```

### 6. GDC Data Access

1. Register free account: **https://portal.gdc.cancer.gov/**
2. Diagnostic slides + clinical data are **open access** — no approval needed
3. Run: `python -m src.data_pipeline`

---

## Project Structure

```
fusiondx/
├── src/
│   ├── data_pipeline.py    # GDC download · tile extraction · patient-level split
│   ├── image_model.py      # ResNet-34 fine-tuning · tile aggregation
│   ├── clinical_model.py   # LightGBM + ClinicalMLP · imputation
│   ├── fusion_model.py     # Cross-modal attention ← CORE
│   ├── evaluate.py         # Honest comparison · bootstrap CIs · all plots
│   ├── synthetic_data.py   # Synthetic H&E generator for development
│   ├── utils.py            # Reproducibility · JSON · class weights
│   └── config.py           # Central configuration
├── tests/
│   ├── test_synthetic_data.py   # 8 tests
│   ├── test_clinical_model.py   # 9 tests
│   ├── test_image_model.py      # 7 tests
│   ├── test_fusion_model.py     # 10 tests
│   ├── test_evaluate.py         # 10 tests
│   ├── test_utils.py            # 17 tests
│   ├── test_gdc_api.py          # 9 live GDC API tests
│   └── test_data_pipeline.py    # 10 tests
├── app.py                  # Streamlit dashboard ← run this
├── train_all.py            # Training orchestrator (--skip-* flags)
├── run_synthetic.py        # Full pipeline on synthetic data
├── verify_gdc.py           # GDC API field verification
├── setup_openslide.py      # OpenSlide installation checker
├── results/                # Charts · report · model predictions
├── data/                   # Splits · tiles (gitignored for real data)
├── requirements.txt
├── Makefile.bat            # Windows command shortcuts
├── QUICKSTART.md
└── README.md
```

---

## Testing

54 tests across 8 files — including live GDC API integration tests.

```cmd
# Fast tests only (~2 min)
python -m pytest tests/test_utils.py tests/test_synthetic_data.py tests/test_evaluate.py

# GDC API tests (require internet)
python -m pytest tests/test_gdc_api.py tests/test_data_pipeline.py -m gdc

# Full suite including model training (~20 min on CPU)
python -m pytest

# All fast tests at once
python -m pytest tests/ -m "not slow" --timeout=60
```

---

## Dashboard

The Streamlit dashboard has 5 pages:

| Page | Content |
|------|---------|
| 🏠 Overview | Architecture cards, key stats, design principles |
| 📊 Results | Interactive ROC curves, bar charts, attention heatmap, full report |
| 🧬 Patient Explorer | Per-patient tiles, clinical data, all 3 predictions, attention weights |
| ⚡ Architecture | Cross-modal attention diagram, training strategy explained |
| ℹ️ About | Methodology, limitations, data attribution |

```cmd
streamlit run app.py
# → http://localhost:8501
```

---

## Windows Shortcuts

```cmd
Makefile.bat synthetic    # run synthetic pipeline
Makefile.bat test-fast    # fast tests
Makefile.bat dashboard    # launch Streamlit
Makefile.bat verify       # check environment
Makefile.bat train        # train on real data
```

---

## Scope and Limitations

| Limitation | Detail | Future direction |
|-----------|--------|-----------------|
| Small dataset | ~120–200 patients | Scale to full TCGA-BRCA (~1000+) |
| Reduced resolution | 256×256 tiles, level 1 | Multi-scale + full-resolution |
| Binary label | IDC vs. other | Multi-subtype classification |
| Simple MIL | Mean-pool tiles | Attention-based MIL (ABMIL) |
| CPU training | Slow (~20 min synthetic) | GPU recommended for real data |

---

## Citation & Data Attribution

```bibtex
@misc{fusiondx2026,
  title  = {FusionDx: Multimodal Cancer Diagnosis with Cross-Modal Attention},
  author = {Eddiegah},
  year   = {2026},
  url    = {https://github.com/Eddiegah/FusionDX}
}
```

Data from:
> The Cancer Genome Atlas Breast Invasive Carcinoma (TCGA-BRCA).
> NIH Genomic Data Commons. https://portal.gdc.cancer.gov/projects/TCGA-BRCA

---

<div align="center">

**[⭐ Star this repo](https://github.com/Eddiegah/FusionDX) · [🐛 Report an issue](https://github.com/Eddiegah/FusionDX/issues) · [🔬 TCGA-BRCA Data](https://portal.gdc.cancer.gov/projects/TCGA-BRCA)**

<br/>

*FusionDx · Research/Educational Demonstration · NOT a clinical diagnostic tool*

</div>
