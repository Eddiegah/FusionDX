# data/ directory

This directory is intentionally mostly empty in version control.

## What goes here

| Subdirectory | Contents |
|---|---|
| `data/clinical/` | Clinical CSV files downloaded from GDC (`clinical_raw.csv`, `slide_manifest.csv`) |
| `data/slides/` | Raw WSI (.svs) files downloaded from GDC -- **NOT committed** (too large, see .gitignore) |
| `data/tiles/` | Extracted 256x256 PNG tiles -- **NOT committed** by default |

## How to populate

After setting up OpenSlide and a GDC account:

```cmd
venv\Scripts\activate
python -m src.data_pipeline
```

## Synthetic data

To run without real data:

```cmd
python run_synthetic.py
```

This generates fake patient data in `data/tiles/SYNTH-XXXX/` for
development and testing. Results have no clinical meaning.

## Storage estimates

- Clinical CSVs: < 1 MB
- Slides (150 patients, smallest available): 10--100 GB
- Tiles (150 patients x 16 tiles x 256x256 PNG): ~150 MB
