# Split counts

Source URL: https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip
Split source: stratified 70/15/15 fixed seed (DATA_SPLIT_SEED=42); torchvision EuroSAT ships no train/val/test split and no official split file is included.
Total samples: 27000
Train: 18900 | Val: 4050 | Test: 4050

Full dataset file counts per class:

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

## Class counts per split

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

Seeds: DATA_SPLIT_SEED=42, TRAIN_SEED=42
torch: 2.15.0+cpu, torchvision: 0.29.2+cpu
GPU available: False (CPU training)