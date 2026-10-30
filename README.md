# eurosat-decision

A small, free, reproducible aerial scene classification project on **EuroSAT** (RGB, 10 classes, 64×64 Sentinel-2 patches at 10 m resolution) with a **softmax-confidence abstention layer**.

Built with only free tools: Python, PyTorch, torchvision, NumPy, Matplotlib, scikit-learn. Runs on Google Colab's free tier or a local CPU — no GPU required, no paid service, no API key.

## What this is

Fine-tune a pretrained torchvision **ResNet-18** (ImageNet weights) on EuroSAT, replacing the final fully-connected layer for the 10 classes. A softmax-confidence abstention layer lets the model abstain (decline to answer) on low-confidence predictions. Accuracy is reported honestly, with the split, denominators, and limitations stated.

## Dataset & split

- **Dataset:** EuroSAT via `torchvision.datasets.EuroSAT`
- **Source URL:** `https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip`
- **Total samples:** 27,000 across 10 classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- **Class counts (full dataset):** AnnualCrop 3000, Forest 3000, HerbaceousVegetation 3000, Highway 2500, Industrial 2500, Pasture 2000, PermanentCrop 2500, Residential 3000, River 2500, SeaLake 3000

**Split source:** torchvision's EuroSAT ships no train/validation/test split, and no official split file is included in the archive. We use our own **fixed stratified 70/15/15 train/validation/test split** with seed 42. This is **not a published benchmark split**; it is generated once, committed, and never re-split.

| Split | Samples |
|-------|---------|
| Train | 18,900 |
| Val   | 4,050 |
| Test  | 4,050 |

Split indices are committed in `splits/train.csv`, `splits/val.csv`, `splits/test.csv` (byte-identical on rerun). Class counts per split are in `reports/split_counts.md`.

## Reproduction (exact commands)

From the repo root (`eurosat-decision`):

```bash
# 0. Environment
python -m pip install -r requirements.txt   # torch, torchvision, numpy, matplotlib, scikit-learn

# 1. Data (downloads EuroSAT, creates & saves split indices)
python step01_data.py

# 2. Model & training (validation-only LR search; writes models/best.pt)
python step02_train.py

# 3. Abstention layer (validation only; writes reports/abstention_val.csv + .png)
python step03_abstention.py

# 4. Final test evaluation (runs exactly once; refuses to overwrite)
python eval_test.py
```

**Seeds:** `DATA_SPLIT_SEED=42`, `TRAIN_SEED=42`.  
**Versions:** torch 2.15.0+cpu, torchvision 0.29.2+cpu.  
**GPU:** not available on this machine; CPU training is used. `models/best.pt` is 44.79 MB (under 100 MB, committed).

## Training settings

- Model: ResNet-18, ImageNet-pretrained (`ResNet18_Weights.IMAGENET1K_V1`), final `fc` replaced with `Linear(in_features, 10)`
- Input size: **64×64** (native EuroSAT patch). ResNet-18's adaptive average pooling accepts any spatial size, so no resize to 224.
- Optimizer: Adam
- LR grid (chosen on validation only): `[0.001, 0.0001, 0.00001]`
- Batch size: 64
- Max epochs: 6, early stopping patience=2 on validation accuracy
- Train augmentations: RandomHorizontalFlip(p=0.5), RandomRotation(15), ToTensor, Normalize(ImageNet)
- Val/test transforms: ToTensor, Normalize(ImageNet)
- **Chosen:** LR = 0.0001, epoch 4, best validation accuracy = **0.9679**

Every validation run is recorded in `reports/val_runs.csv`. The best checkpoint is saved to `models/best.pt`. The test set is **never** used during training or validation.

## Results (test set — evaluated exactly once)

Overall test accuracy: **0.9627 (3,899 / 4,050)**

The figure 0.9679 (best validation accuracy) is **validation-selected**; the test accuracy above is the **only unbiased figure**.

| Metric | Value |
|--------|-------|
| Overall test accuracy | 0.9627 (3,899 / 4,050) |
| Frozen abstention threshold | 0.50 |
| Coverage at threshold | 0.9938 (4,025 / 4,050 answered) |
| Accuracy on answered cells | 0.9650 |
| Cells abstained | 25 |

**SeaLake (water) class — reported separately:** precision 0.9912, recall 0.9956, F1 0.9933, support 450, per-class accuracy 0.9956.

Full per-class precision/recall/F1 and the confusion matrix: `reports/TEST_RESULTS.md` and `reports/test_results.json`. The validation coverage/accuracy sweep: `reports/abstention_val.csv` and `reports/abstention_val.png`.

## Abstention layer

Threshold rule (stated in advance, applied on validation only): choose the **lowest threshold** at which validation accuracy on the answered cells is at least 0.95. All thresholds 0.50–0.95 cleared the 0.95 bar, so the lowest — **0.50** — was frozen and applied once on the test set. The frozen threshold is recorded in `reports/abstention_threshold.json`.

## Limitations

- 64×64 Sentinel-2 patches at 10 m resolution; this is **benchmark imagery, not field imagery**.
- A **single run with one seed (42)**; **no confidence interval is computed**.
- The split is our own fixed stratified 70/15/15 split, not a published benchmark split.
- Result is a benchmark on **EuroSAT only**; no accuracy is claimed on any other imagery.
- CPU-only training; ~101 minutes total training time on this machine.

## Files

```
data/eurosat/2750/        downloaded EuroSAT (not committed)
splits/{train,val,test}.csv   committed split indices
models/best.pt            committed best checkpoint (44.79 MB)
reports/CHECKPOINT1.md    data checkpoint summary
reports/CHECKPOINT2.md    training checkpoint summary
reports/split_counts.md   class counts per split
reports/val_runs.csv      every validation run (LR grid)
reports/abstention_val.csv       coverage vs accuracy by threshold (validation)
reports/abstention_val.png       abstention plot
reports/abstention_threshold.json frozen threshold
reports/TEST_RESULTS.md          test results (single run)
reports/test_results.json        test results (machine-readable)
step01_data.py / step02_train.py / step03_abstention.py / eval_test.py
```

## Scope

This is a small reproducibility benchmark on EuroSAT only. It is not a broad claim about aerial scene classification generally, and it is not positioned as a production system.
