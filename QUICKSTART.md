# FusionDx Quickstart

> Full setup details are in README.md. This is the fast orientation.

## What's installed and ready right now

```
venv/          Python 3.11 virtual environment -- all deps installed
src/           Source modules -- all syntax-verified, 44+ tests passing
tests/         Full test suite (unit + integration + GDC API tests)
```

Installed packages: torch 2.3.1, torchvision 0.18.1, numpy 1.26.4,
pandas 2.2.2, scikit-learn 1.5.1, matplotlib 3.8.4, streamlit 1.36.0,
lightgbm 4.3.0, requests, tqdm, Pillow, seaborn, pytest

## What still needs a manual step

**OpenSlide Windows binaries** -- required for real WSI data:
1. Download: https://openslide.org/download/ (Windows 64-bit zip)
2. Extract to e.g. `C:\OpenSlide\`
3. Add `C:\OpenSlide\bin` to system PATH, restart terminal
4. Run: `venv\Scripts\python.exe setup_openslide.py`

## Run right now (no OpenSlide needed)

```cmd
venv\Scripts\activate
python run_synthetic.py       # full pipeline on synthetic data (~20 min CPU)
streamlit run app.py          # dashboard (open browser to localhost:8501)
```

## Run the test suite

```cmd
venv\Scripts\activate

# Fast tests (unit + evaluate + GDC API connectivity):
python -m pytest tests/test_synthetic_data.py tests/test_evaluate.py tests/test_gdc_api.py

# Model training tests (slow, ~10 min on CPU):
python -m pytest tests/test_clinical_model.py tests/test_image_model.py tests/test_fusion_model.py

# Everything:
python -m pytest
```

## After OpenSlide is installed

```cmd
python -m src.data_pipeline --verify    # confirm full environment
python -m src.data_pipeline             # download TCGA-BRCA data (hours, ~20-100 GB)
python train_all.py                     # train all 3 models + evaluate
streamlit run app.py                    # explore real results
```

## Key files to read

| File | Why |
|------|-----|
| `src/fusion_model.py` | Cross-modal attention -- the technical core |
| `src/data_pipeline.py` | Patient-level splitting logic -- most important correctness decision |
| `results/comparison_report.md` | Honest evaluation results |
| `README.md` | Full documentation |
