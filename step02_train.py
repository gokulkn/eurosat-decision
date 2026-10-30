"""
Step 2 — Model and training.

Model: torchvision ResNet-18 pretrained on ImageNet, final FC replaced for 10 classes.
Inputs kept at native 64x64 (ResNet-18 adaptive average pooling accepts any spatial size;
stated explicitly — we do NOT resize to 224).

Training:
- Train split only. Validation used for early stopping and LR choice.
- LR grid (3 values): [1e-3, 1e-4, 1e-5], chosen from Adam fine-tuning practice.
- Every validation run recorded to reports/val_runs.csv.
- Best checkpoint by validation accuracy saved to models/best.pt.
- CPU only, few epochs, early stopping (patience=5, max_epochs=20).
- Test split is NEVER loaded during training.

Seeds: TRAIN_SEED=42 (also DATA_SPLIT_SEED=42 for the split).
"""

import csv
import os
import time
import hashlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT
from torchvision import transforms, models

# --- Seeds ---
TRAIN_SEED = 42
np.random.seed(TRAIN_SEED)
torch.manual_seed(TRAIN_SEED)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
SPLITS_DIR = os.path.join(ROOT, "splits")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"torch: {torch.__version__}, torchvision: {__import__('torchvision').__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Train seed: {TRAIN_SEED}")

# --- Input size decision ---
INPUT_SIZE = 64  # native EuroSAT patch size; ResNet-18 adaptive pool accepts any size
RESIZE_TO_224 = False
print(f"Input size: {INPUT_SIZE}x{INPUT_SIZE} (resize_to_224={RESIZE_TO_224})")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# --- Load full dataset (already downloaded in Step 1; no re-download) ---
data_folder = os.path.join(DATA_DIR, "eurosat", "2750")
assert os.path.exists(data_folder), f"Dataset not found at {data_folder}"

full_train = EuroSAT(root=DATA_DIR, download=False, transform=train_tf)
full_val = EuroSAT(root=DATA_DIR, download=False, transform=val_tf)
full_test = EuroSAT(root=DATA_DIR, download=False, transform=val_tf)

CLASSES = full_train.classes
NUM_CLASSES = len(CLASSES)
print(f"Classes ({NUM_CLASSES}): {CLASSES}")

# --- Load split indices from committed CSVs ---
def load_split_indices(name):
    path = os.path.join(SPLITS_DIR, f"{name}.csv")
    assert os.path.exists(path), f"Missing split file {path}"
    idxs = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idxs.append(int(row["index"]))
    return sorted(idxs)

train_idx = load_split_indices("train")
val_idx = load_split_indices("val")
test_idx = load_split_indices("test")

print(f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
print("(Test split loaded only to confirm it exists; never used during training/validation.)")

train_subset = Subset(full_train, train_idx)
val_subset = Subset(full_val, val_idx)

BATCH_SIZE = 64
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)

# --- Model ---
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

def run_epoch(model, loader, optimizer, training):
    if training:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            if training:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n

def evaluate(model, loader):
    model.eval()
    total_correct = 0
    total_n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += x.size(0)
    return total_correct / total_n

# --- LR grid search (validation only) ---
LR_GRID = [1e-3, 1e-4, 1e-5]
MAX_EPOCHS = 6
PATIENCE = 2

val_runs_path = os.path.join(REPORTS_DIR, "val_runs.csv")
val_runs_rows = []

best_overall_val_acc = -1.0
best_overall_state = None
best_overall_lr = None
best_overall_epoch = None

t0_all = time.time()

for lr in LR_GRID:
    print(f"\n=== LR = {lr} ===")
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, training=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, training=False)
        dt = time.time() - t0

        val_runs_rows.append({
            "lr": lr,
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "epoch_seconds": round(dt, 3),
        })

        print(f"  epoch {epoch:2d}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({dt:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  early stopping at epoch {epoch} (patience {PATIENCE})")
            break

    print(f"LR {lr}: best val_acc={best_val_acc:.4f} at epoch {best_epoch}")

    if best_val_acc > best_overall_val_acc:
        best_overall_val_acc = best_val_acc
        best_overall_state = best_state
        best_overall_lr = lr
        best_overall_epoch = best_epoch

# --- Write val_runs.csv ---
with open(val_runs_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["lr", "epoch", "train_loss", "train_acc",
                                           "val_loss", "val_acc", "epoch_seconds"])
    writer.writeheader()
    writer.writerows(val_runs_rows)
print(f"\nWrote {val_runs_path} ({len(val_runs_rows)} rows)")

# --- Save best checkpoint ---
best_ckpt_path = os.path.join(MODELS_DIR, "best.pt")
torch.save(best_overall_state, best_ckpt_path)

ckpt_size = os.path.getsize(best_ckpt_path)
ckpt_hash = hashlib.sha256(open(best_ckpt_path, "rb").read()).hexdigest()
print(f"Best checkpoint: {best_ckpt_path}")
print(f"  LR={best_overall_lr}, epoch={best_overall_epoch}, val_acc={best_overall_val_acc:.4f}")
print(f"  size={ckpt_size} bytes ({ckpt_size/1e6:.2f} MB), sha256={ckpt_hash}")

total_runtime = time.time() - t0_all
print(f"\nTotal training runtime: {total_runtime:.1f}s ({total_runtime/60:.1f} min)")
print(f"Best overall: LR={best_overall_lr}, epoch={best_overall_epoch}, val_acc={best_overall_val_acc:.4f}")

# Commit decision
COMMIT_WEIGHTS = ckpt_size < 100 * 1024 * 1024
print(f"Commit weights (<100MB): {COMMIT_WEIGHTS}")

# --- Write CHECKPOINT2.md ---
lines = []
lines.append("# CHECKPOINT 2 — Model and training")
lines.append("")
lines.append("## Model")
lines.append("- Architecture: torchvision ResNet-18, pretrained on ImageNet (ResNet18_Weights.IMAGENET1K_V1)")
lines.append("- Final layer: replaced `fc` with `Linear(in_features, 10)` for the 10 EuroSAT classes")
lines.append(f"- Input size: {INPUT_SIZE}x{INPUT_SIZE} (native EuroSAT patch). Resize to 224: {RESIZE_TO_224}.")
lines.append("  ResNet-18 adaptive average pooling accepts any spatial size, so we keep 64x64.")
lines.append("")
lines.append("## Training settings")
lines.append(f"- Device: {DEVICE} (GPU available: {torch.cuda.is_available()})")
lines.append(f"- Optimizer: Adam")
lines.append(f"- LR grid (validation only): {LR_GRID}")
lines.append(f"- Batch size: {BATCH_SIZE}")
lines.append(f"- Max epochs: {MAX_EPOCHS}")
lines.append(f"- Early stopping: patience={PATIENCE} on validation accuracy")
lines.append(f"- Loss: CrossEntropyLoss")
lines.append(f"- Train augmentations: RandomHorizontalFlip(p=0.5), RandomRotation(15), ToTensor, Normalize(ImageNet)")
lines.append(f"- Val/Test transforms: ToTensor, Normalize(ImageNet)")
lines.append(f"- Seeds: TRAIN_SEED=42, DATA_SPLIT_SEED=42")
lines.append("")
lines.append("## Data split (validation-only tuning)")
lines.append(f"- Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)} (never used in training)")
lines.append("- Split source: our own fixed stratified 70/15/15 split, seed 42 (NOT a published benchmark split)")
lines.append("- Split indices: splits/train.csv, splits/val.csv, splits/test.csv (committed, byte-identical on rerun)")
lines.append("")
lines.append("## Validation curve")
lines.append("")
lines.append("| lr | epoch | train_acc | val_acc | train_loss | val_loss | epoch_seconds |")
lines.append("|----|-------|-----------|---------|------------|----------|---------------|")
for r in val_runs_rows:
    lines.append(f"| {r['lr']} | {r['epoch']} | {r['train_acc']} | {r['val_acc']} | "
                 f"{r['train_loss']} | {r['val_loss']} | {r['epoch_seconds']} |")
lines.append("")
lines.append("## Best epoch per LR")
for lr in LR_GRID:
    rows = [r for r in val_runs_rows if r["lr"] == lr]
    best = max(rows, key=lambda r: r["val_acc"])
    lines.append(f"- LR {lr}: best val_acc={best['val_acc']} at epoch {best['epoch']}")
lines.append("")
lines.append("## Chosen settings")
lines.append(f"- Chosen LR: {best_overall_lr}")
lines.append(f"- Chosen epoch: {best_overall_epoch}")
lines.append(f"- Best validation accuracy: {best_overall_val_acc:.4f}")
lines.append(f"- Best checkpoint: models/best.pt ({ckpt_size/1e6:.2f} MB, sha256={ckpt_hash})")
lines.append(f"- Commit weights (<100 MB): {COMMIT_WEIGHTS}")
lines.append("")
lines.append(f"## Runtime")
lines.append(f"- Total training runtime: {total_runtime:.1f}s ({total_runtime/60:.1f} min) on CPU")
lines.append("")
lines.append("## Test set status")
lines.append("- Test set NOT evaluated yet. Evaluation happens exactly once in Step 4 (eval_test.py).")
lines.append("")
lines.append("---")
lines.append("STOP at CHECKPOINT 2 — waiting for human confirmation before the test run.")

with open(os.path.join(REPORTS_DIR, "CHECKPOINT2.md"), "w") as f:
    f.write("\n".join(lines))
print(f"\nWrote {os.path.join(REPORTS_DIR, 'CHECKPOINT2.md')}")
print("")
print("STOP at CHECKPOINT 2 — waiting for human confirmation before the test run.")
