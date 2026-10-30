# CHECKPOINT 2 — Model and training

## Model
- Architecture: torchvision ResNet-18, pretrained on ImageNet (ResNet18_Weights.IMAGENET1K_V1)
- Final layer: replaced `fc` with `Linear(in_features, 10)` for the 10 EuroSAT classes
- Input size: 64x64 (native EuroSAT patch). Resize to 224: False.
  ResNet-18 adaptive average pooling accepts any spatial size, so we keep 64x64.

## Training settings
- Device: cpu (GPU available: False)
- Optimizer: Adam
- LR grid (validation only): [0.001, 0.0001, 1e-05]
- Batch size: 64
- Max epochs: 6
- Early stopping: patience=2 on validation accuracy
- Loss: CrossEntropyLoss
- Train augmentations: RandomHorizontalFlip(p=0.5), RandomRotation(15), ToTensor, Normalize(ImageNet)
- Val/Test transforms: ToTensor, Normalize(ImageNet)
- Seeds: TRAIN_SEED=42, DATA_SPLIT_SEED=42

## Data split (validation-only tuning)
- Train: 18900 | Val: 4050 | Test: 4050 (never used in training)
- Split source: our own fixed stratified 70/15/15 split, seed 42 (NOT a published benchmark split)
- Split indices: splits/train.csv, splits/val.csv, splits/test.csv (committed, byte-identical on rerun)

## Validation curve

| lr | epoch | train_acc | val_acc | train_loss | val_loss | epoch_seconds |
|----|-------|-----------|---------|------------|----------|---------------|
| 0.001 | 1 | 0.858254 | 0.91284 | 0.4418 | 0.279347 | 395.535 |
| 0.001 | 2 | 0.91582 | 0.92 | 0.264308 | 0.224968 | 388.731 |
| 0.001 | 3 | 0.924868 | 0.952346 | 0.230674 | 0.143685 | 389.426 |
| 0.001 | 4 | 0.930688 | 0.908148 | 0.208554 | 0.268219 | 351.842 |
| 0.001 | 5 | 0.940688 | 0.938025 | 0.182638 | 0.180272 | 377.754 |
| 0.0001 | 1 | 0.871376 | 0.944198 | 0.397203 | 0.165171 | 391.143 |
| 0.0001 | 2 | 0.938995 | 0.948148 | 0.179169 | 0.146786 | 385.53 |
| 0.0001 | 3 | 0.954603 | 0.961235 | 0.135371 | 0.120149 | 378.033 |
| 0.0001 | 4 | 0.965026 | 0.967901 | 0.103576 | 0.095713 | 395.258 |
| 0.0001 | 5 | 0.97 | 0.967654 | 0.089279 | 0.100022 | 353.366 |
| 0.0001 | 6 | 0.97291 | 0.964444 | 0.080993 | 0.104159 | 341.184 |
| 1e-05 | 1 | 0.634392 | 0.820247 | 1.199736 | 0.663134 | 348.26 |
| 1e-05 | 2 | 0.864074 | 0.893827 | 0.456263 | 0.372318 | 352.343 |
| 1e-05 | 3 | 0.895132 | 0.912099 | 0.32986 | 0.289807 | 337.799 |
| 1e-05 | 4 | 0.912011 | 0.924444 | 0.269635 | 0.23752 | 290.263 |
| 1e-05 | 5 | 0.924603 | 0.933827 | 0.229434 | 0.204018 | 290.301 |
| 1e-05 | 6 | 0.929577 | 0.940494 | 0.211623 | 0.19159 | 288.117 |

## Best epoch per LR
- LR 0.001: best val_acc=0.952346 at epoch 3
- LR 0.0001: best val_acc=0.967901 at epoch 4
- LR 1e-05: best val_acc=0.940494 at epoch 6

## Chosen settings
- Chosen LR: 0.0001
- Chosen epoch: 4
- Best validation accuracy: 0.9679
- Best checkpoint: models/best.pt (44.79 MB, sha256=c6c5eed4be400498d8c43941e1115da72dcde94a37d73b358e6182923201716c)
- Commit weights (<100 MB): True

## Runtime
- Total training runtime: 6056.7s (100.9 min) on CPU

## Test set status
- Test set NOT evaluated yet. Evaluation happens exactly once in Step 4 (eval_test.py).

---
STOP at CHECKPOINT 2 — waiting for human confirmation before the test run.