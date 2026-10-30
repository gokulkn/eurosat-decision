# Test Results (EuroSAT) — single run

## Important notes
- This is a **benchmark result on EuroSAT only**. No accuracy is claimed on any other imagery.
- The figure 0.9679 (best validation accuracy, LR=0.0001, epoch 4) is **validation-selected**;
  the test accuracy below is the **only unbiased figure**.
- Test set evaluated **exactly once** by eval_test.py; results are committed and not re-run.

## Dataset & split
- Source URL: https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip
- Split source: our own fixed stratified 70/15/15 split, seed 42 (NOT a published benchmark split; torchvision EuroSAT ships no split)
- Train/val/test: 18900 / 4050 / 4050 (total 27000)
- Seeds: DATA_SPLIT_SEED=42, TRAIN_SEED=42
- Versions: torch 2.15.0+cpu, torchvision 0.29.2+cpu
- GPU available: False (CPU training)
- Input size: 64x64 (native EuroSAT patch; no resize to 224)
- Model: ResNet-18 ImageNet-pretrained, fc replaced for 10 classes
- Checkpoint: models/best.pt (44.79 MB, committed)

## Overall test accuracy
- **0.9627** (3899 / 4050)

## Abstention (frozen threshold = 0.5)
- Threshold rule: lowest threshold at which validation accuracy on answered cells >= 0.95; if none, no abstention
- Frozen threshold: 0.5
- Coverage at threshold: 0.9938 (4025 / 4050 cells answered)
- Accuracy on answered cells: 0.9650
- Cells abstained: 25

## Per-class precision / recall / F1 / support
| class | precision | recall | f1 | support |
|-------|-----------|--------|----|---------|
| AnnualCrop | 0.9690 | 0.9022 | 0.9344 | 450 |
| Forest | 0.9933 | 0.9933 | 0.9933 | 450 |
| HerbaceousVegetation | 0.9633 | 0.9333 | 0.9481 | 450 |
| Highway | 0.9450 | 0.9627 | 0.9538 | 375 |
| Industrial | 0.9788 | 0.9867 | 0.9827 | 375 |
| Pasture | 0.9597 | 0.9533 | 0.9565 | 300 |
| PermanentCrop | 0.8705 | 0.9680 | 0.9167 | 375 |
| Residential | 1.0000 | 0.9778 | 0.9888 | 450 |
| River | 0.9471 | 0.9547 | 0.9509 | 375 |
| SeaLake | 0.9912 | 0.9956 | 0.9933 | 450 |

## SeaLake (water) class — reported separately
- Class: SeaLake (index 9)
- Precision: 0.9912
- Recall: 0.9956
- F1: 0.9933
- Support: 450
- Per-class accuracy: 0.9956

## Confusion matrix (rows=true, cols=pred)
| true \ pred | AnnualCrop | Forest | HerbaceousVegetation | Highway | Industrial | Pasture | PermanentCrop | Residential | River | SeaLake |
|---|---|---|---|---|---|---|---|---|---|---|
| AnnualCrop | 406 | 0 | 2 | 2 | 0 | 2 | 27 | 0 | 8 | 3 |
| Forest | 0 | 447 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| HerbaceousVegetation | 0 | 0 | 420 | 4 | 0 | 6 | 19 | 0 | 1 | 0 |
| Highway | 5 | 0 | 0 | 361 | 2 | 2 | 0 | 0 | 5 | 0 |
| Industrial | 0 | 0 | 0 | 1 | 370 | 0 | 4 | 0 | 0 | 0 |
| Pasture | 1 | 2 | 5 | 1 | 0 | 286 | 2 | 0 | 3 | 0 |
| PermanentCrop | 4 | 1 | 4 | 1 | 0 | 1 | 363 | 0 | 1 | 0 |
| Residential | 0 | 0 | 3 | 0 | 5 | 0 | 2 | 440 | 0 | 0 |
| River | 3 | 0 | 0 | 12 | 1 | 1 | 0 | 0 | 358 | 0 |
| SeaLake | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 448 |

## Reproducibility
- Command: `python eval_test.py` (runs once; refuses to overwrite existing results)
- Files: models/best.pt, splits/test.csv, reports/abstention_threshold.json

## Limitations
- 64x64 Sentinel-2 patches at 10 m resolution; not field imagery.
- A single run with one seed (42); no confidence interval computed.
- Split is our own fixed stratified 70/15/15 split, not a published benchmark split.
- Benchmark on EuroSAT only.