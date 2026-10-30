# CHECKPOINT 1 — Data

## Dataset
- **Dataset:** EuroSAT (RGB, 10 classes, 64x64 Sentinel-2 patches)
- **Source URL:** https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip
- **Loader:** torchvision.datasets.EuroSAT (download=True) into data/eurosat/2750/

## Split source
- torchvision.datasets.EuroSAT is a plain ImageFolder and ships NO train/validation/test split.
- No official split file is included in the downloaded archive.
- Split used: **stratified 70/15/15 train/validation/test**, fixed seed.
- This split is generated once and committed; it is never re-split.

## Seeds
- DATA_SPLIT_SEED = 42
- TRAIN_SEED = 42

## File counts

Full dataset (27,000 total):

| class | count |
|-------|-------|
| AnnualCrop | 3000 |
| Forest | 3000 |
| HerbaceousVegetation | 3000 |
| Highway | 2500 |
| Industrial | 2500 |
| Pasture | 2000 |
| PermanentCrop | 2500 |
| Residential | 3000 |
| River | 2500 |
| SeaLake | 3000 |

Class counts per split:

| class | train | val | test |
|-------|-------|-----|------|
| AnnualCrop | 2100 | 450 | 450 |
| Forest | 2100 | 450 | 450 |
| HerbaceousVegetation | 2100 | 450 | 450 |
| Highway | 1750 | 375 | 375 |
| Industrial | 1750 | 375 | 375 |
| Pasture | 1400 | 300 | 300 |
| PermanentCrop | 1750 | 375 | 375 |
| Residential | 2100 | 450 | 450 |
| River | 1750 | 375 | 375 |
| SeaLake | 2100 | 450 | 450 |
| **total** | **18900** | **4050** | **4050** |

## Split indices committed
- splits/train.csv (18900 rows)
- splits/val.csv (4050 rows)
- splits/test.csv (4050 rows)

## Environment
- torch: 2.15.0+cpu
- torchvision: 0.29.2+cpu
- GPU available: False (CPU training)

---

**STOP at CHECKPOINT 1.** Waiting for human confirmation before proceeding to Step 2 (model training).
