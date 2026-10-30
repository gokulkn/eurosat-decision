"""
Step 1 — Data: download EuroSAT via torchvision, record split source & counts,
save split indices to splits/ as CSV, save class counts per split to reports/split_counts.md.

Split source: torchvision.datasets.EuroSAT is a plain ImageFolder and does NOT ship a
train/validation/test split. There is no torchvision-provided split for EuroSAT and no
official split file ships in the downloaded archive. We therefore use a fixed, stratified
train/validation/test split (70/15/15) with a fixed seed. This split is generated once and
committed; training/validation/test never re-split.

Seeds: DATA_SPLIT_SEED=42, TRAIN_SEED=42
"""

import csv
import os
import shutil
import sys

import numpy as np
import torch
from torchvision.datasets import EuroSAT
from torchvision import transforms

# --- Seeds ---
DATA_SPLIT_SEED = 42
TRAIN_SEED = 42
np.random.seed(DATA_SPLIT_SEED)
torch.manual_seed(TRAIN_SEED)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
SPLITS_DIR = os.path.join(ROOT, "splits")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

SOURCE_URL = (
    "https://huggingface.co/datasets/torchgeo/eurosat/"
    "resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip"
)

print("=" * 60)
print("Step 1 — Data")
print("=" * 60)
print(f"Source URL: {SOURCE_URL}")
print(f"torch: {torch.__version__}")
print(f"torchvision: {__import__('torchvision').__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Data split seed: {DATA_SPLIT_SEED}")
print(f"Train seed: {TRAIN_SEED}")

# --- Download ---
print("\nDownloading EuroSAT via torchvision.datasets.EuroSAT ...")
ds = EuroSAT(root=DATA_DIR, download=True, transform=transforms.ToTensor())

data_folder = os.path.join(DATA_DIR, "eurosat", "2750")
print(f"Dataset root: {data_folder}")
print(f"Total samples: {len(ds)}")
print(f"Classes ({len(ds.classes)}): {ds.classes}")
print(f"Class-to-idx: {ds.class_to_idx}")

# --- File counts per class (full dataset) ---
class_counts_full = {c: 0 for c in ds.classes}
for path, label in ds.samples:
    class_counts_full[ds.classes[label]] += 1

print("\nFile counts per class (full dataset):")
for c in ds.classes:
    print(f"  {c}: {class_counts_full[c]}")

# --- Split: stratified 70/15/15 ---
rng = np.random.RandomState(DATA_SPLIT_SEED)
train_idx, val_idx, test_idx = [], [], []
for c in ds.classes:
    idxs = np.where(np.array([label for _, label in ds.samples]) == ds.class_to_idx[c])[0]
    rng.shuffle(idxs)
    n = len(idxs)
    n_train = int(round(0.70 * n))
    n_val = int(round(0.15 * n))
    n_test = n - n_train - n_val
    train_idx.extend(idxs[:n_train].tolist())
    val_idx.extend(idxs[n_train:n_train + n_val].tolist())
    test_idx.extend(idxs[n_train + n_val:].tolist())

train_idx = sorted(train_idx)
val_idx = sorted(val_idx)
test_idx = sorted(test_idx)

print(f"\nSplit sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
print(f"Split source: stratified 70/15/15 fixed seed (no torchvision/official split ships with EuroSAT)")

# --- Save split indices to CSV ---
def write_split_csv(name, idxs):
    path = os.path.join(SPLITS_DIR, f"{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "class", "filepath"])
        for i in idxs:
            fpath, label = ds.samples[i]
            rel = os.path.relpath(fpath, DATA_DIR)
            w.writerow([i, ds.classes[label], rel])
    print(f"  wrote {path} ({len(idxs)} rows)")

write_split_csv("train", train_idx)
write_split_csv("val", val_idx)
write_split_csv("test", test_idx)

# --- Class counts per split ---
def counts_for(idxs):
    return {c: 0 for c in ds.classes}, None

split_counts = {}
for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    counts = {c: 0 for c in ds.classes}
    for i in idxs:
        _, label = ds.samples[i]
        counts[ds.classes[label]] += 1
    split_counts[name] = counts

print("\nClass counts per split:")
header = ["class"] + list(split_counts.keys())
rows = []
for c in ds.classes:
    rows.append([c] + [split_counts[s][c] for s in ["train", "val", "test"]])

# --- Write reports/split_counts.md ---
md_path = os.path.join(REPORTS_DIR, "split_counts.md")
lines = []
lines.append("# Split counts")
lines.append("")
lines.append(f"Source URL: {SOURCE_URL}")
lines.append(f"Split source: stratified 70/15/15 fixed seed (DATA_SPLIT_SEED={DATA_SPLIT_SEED}); "
             "torchvision EuroSAT ships no train/val/test split and no official split file is included.")
lines.append(f"Total samples: {len(ds)}")
lines.append(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
lines.append("")
lines.append("Full dataset file counts per class:")
lines.append("")
lines.append("| class | count |")
lines.append("|-------|-------|")
for c in ds.classes:
    lines.append(f"| {c} | {class_counts_full[c]} |")
lines.append("")
lines.append("## Class counts per split")
lines.append("")
lines.append("| class | train | val | test |")
lines.append("|-------|-------|-----|------|")
for c in ds.classes:
    lines.append(f"| {c} | {split_counts['train'][c]} | {split_counts['val'][c]} | {split_counts['test'][c]} |")
lines.append("")
lines.append("| **total** | **{}** | **{}** | **{}** |".format(
    len(train_idx), len(val_idx), len(test_idx)))
lines.append("")
lines.append("Seeds: DATA_SPLIT_SEED=42, TRAIN_SEED=42")
lines.append(f"torch: {torch.__version__}, torchvision: {__import__('torchvision').__version__}")
lines.append("GPU available: False (CPU training)")

with open(md_path, "w") as f:
    f.write("\n".join(lines))
print(f"\nWrote {md_path}")

# --- Summary for checkpoint ---
print("\n" + "=" * 60)
print("CHECKPOINT 1 summary")
print("=" * 60)
print(f"Source URL: {SOURCE_URL}")
print(f"Total samples: {len(ds)}")
print(f"Classes: {ds.classes}")
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
print(f"Split indices saved to: {SPLITS_DIR}/{{train,val,test}}.csv")
print(f"Split counts saved to: {md_path}")
print("")
print("STOP at CHECKPOINT 1 — waiting for human confirmation before training.")
