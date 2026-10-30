"""
Step 4 — Final test evaluation (RUN EXACTLY ONCE).

Loads models/best.pt, the frozen abstention threshold (chosen on validation only),
and the test split, runs exactly once, and writes:
  reports/TEST_RESULTS.md
  reports/test_results.json

Contents: overall accuracy, per-class precision/recall/F1, confusion matrix,
coverage and accuracy at the frozen threshold, and the SeaLake (water) class
reported separately. Also records split source, seeds, torch version, and a note
that this is a benchmark result on EuroSAT only.

Guard: if reports/test_results.json already exists, this script aborts without
overwriting — enforcing "evaluate the test set exactly once."
"""

import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT
from torchvision import transforms, models
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
)

# --- Seeds ---
TRAIN_SEED = 42
DATA_SPLIT_SEED = 42
np.random.seed(TRAIN_SEED)
torch.manual_seed(TRAIN_SEED)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
SPLITS_DIR = os.path.join(ROOT, "splits")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")

RESULTS_JSON = os.path.join(REPORTS_DIR, "test_results.json")
RESULTS_MD = os.path.join(REPORTS_DIR, "TEST_RESULTS.md")

# --- Guard: run exactly once ---
if os.path.exists(RESULTS_JSON):
    print(f"ABORT: {RESULTS_JSON} already exists — test set has already been evaluated once.")
    print("Per the rules, the test set is evaluated exactly once. Not overwriting.")
    sys.exit(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"torch: {torch.__version__}, torchvision: {__import__('torchvision').__version__}")
print(f"GPU available: {torch.cuda.is_available()}")

INPUT_SIZE = 64  # native EuroSAT patch; no resize to 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

SOURCE_URL = (
    "https://huggingface.co/datasets/torchgeo/eurosat/"
    "resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip"
)

# --- Load frozen threshold (chosen on validation only) ---
THR_PATH = os.path.join(REPORTS_DIR, "abstention_threshold.json")
assert os.path.exists(THR_PATH), f"Missing frozen threshold record {THR_PATH}"
with open(THR_PATH) as f:
    thr_record = json.load(f)
FROZEN_THRESHOLD = thr_record["chosen_threshold"]
print(f"Frozen threshold (from validation): {FROZEN_THRESHOLD}")
print(f"Threshold rule: {thr_record['rule']}")

# --- Load test split ---
def load_split_indices(name):
    path = os.path.join(SPLITS_DIR, f"{name}.csv")
    assert os.path.exists(path), f"Missing split file {path}"
    idxs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            idxs.append(int(row["index"]))
    return sorted(idxs)

test_idx = load_split_indices("test")
print(f"Test samples: {len(test_idx)}")

full_test = EuroSAT(root=DATA_DIR, download=False, transform=test_tf)
test_subset = Subset(full_test, test_idx)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=0)

CLASSES = full_test.classes
NUM_CLASSES = len(CLASSES)
SEALAKE_IDX = CLASSES.index("SeaLake")
print(f"Classes ({NUM_CLASSES}): {CLASSES}")
print(f"SeaLake (water) class index: {SEALAKE_IDX}")

# --- Build model + load best checkpoint ---
def build_model():
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m.to(DEVICE)

ckpt_path = os.path.join(MODELS_DIR, "best.pt")
assert os.path.exists(ckpt_path), f"Missing checkpoint {ckpt_path}"
model = build_model()
state = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print(f"Loaded checkpoint: {ckpt_path}")

# --- Run test set once ---
all_probs = []
all_labels = []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())

probs = np.concatenate(all_probs, axis=0)
labels = np.concatenate(all_labels, axis=0)
preds = probs.argmax(axis=1)
conf = probs.max(axis=1)
n_total = len(labels)

# --- Overall accuracy ---
overall_acc = float((preds == labels).mean())

# --- Per-class precision, recall, F1 ---
precision, recall, f1, support = precision_recall_fscore_support(
    labels, preds, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
)

# --- Confusion matrix ---
cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

# --- SeaLake (water) class reported separately ---
sl_precision = float(precision[SEALAKE_IDX])
sl_recall = float(recall[SEALAKE_IDX])
sl_f1 = float(f1[SEALAKE_IDX])
sl_support = int(support[SEALAKE_IDX])
sl_correct = int((preds[labels == SEALAKE_IDX] == SEALAKE_IDX).sum())
sl_accuracy = sl_correct / sl_support if sl_support else 0.0

# --- Coverage and accuracy at frozen threshold ---
answered = conf >= FROZEN_THRESHOLD
n_answered = int(answered.sum())
coverage = n_answered / n_total
if n_answered > 0:
    accuracy_on_answered = float((preds[answered] == labels[answered]).mean())
else:
    accuracy_on_answered = float("nan")
n_abstained = int((~answered).sum())

# --- Assemble results ---
results = {
    "note": "Benchmark result on EuroSAT only. Not a claim on any other imagery.",
    "dataset": "EuroSAT (RGB, 10 classes, 64x64 Sentinel-2 patches)",
    "source_url": SOURCE_URL,
    "split_source": ("our own fixed stratified 70/15/15 split, seed 42 "
                     "(NOT a published benchmark split; torchvision EuroSAT ships no split)"),
    "splits": {"train": 18900, "val": 4050, "test": 4050},
    "seeds": {"DATA_SPLIT_SEED": DATA_SPLIT_SEED, "TRAIN_SEED": TRAIN_SEED},
    "versions": {"torch": torch.__version__, "torchvision": __import__("torchvision").__version__},
    "gpu_available": bool(torch.cuda.is_available()),
    "device": str(DEVICE),
    "input_size": [INPUT_SIZE, INPUT_SIZE],
    "model": "ResNet-18 ImageNet-pretrained, fc replaced for 10 classes",
    "checkpoint": "models/best.pt",
    "checkpoint_sha256": __import__("hashlib").sha256(open(ckpt_path, "rb").read()).hexdigest(),
    "best_validation_accuracy": 0.9679,
    "best_validation_note": "0.9679 is validation-selected (LR=0.0001, epoch 4); the test result below is the only unbiased figure.",
    "classes": CLASSES,
    "n_total": int(n_total),
    "overall_accuracy": overall_acc,
    "frozen_threshold": FROZEN_THRESHOLD,
    "coverage_at_threshold": coverage,
    "accuracy_on_answered": accuracy_on_answered,
    "n_answered": n_answered,
    "n_abstained": n_abstained,
    "per_class": {},
    "confusion_matrix": cm.tolist(),
    "sealake_water_class": {
        "class": "SeaLake",
        "index": SEALAKE_IDX,
        "precision": sl_precision,
        "recall": sl_recall,
        "f1": sl_f1,
        "support": sl_support,
        "accuracy": sl_accuracy,
    },
}
for i, c in enumerate(CLASSES):
    results["per_class"][c] = {
        "index": int(i),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "f1": float(f1[i]),
        "support": int(support[i]),
    }

# --- Write test_results.json ---
os.makedirs(REPORTS_DIR, exist_ok=True)
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWrote {RESULTS_JSON}")

# --- Write TEST_RESULTS.md ---
lines = []
lines.append("# Test Results (EuroSAT) — single run")
lines.append("")
lines.append("## Important notes")
lines.append("- This is a **benchmark result on EuroSAT only**. No accuracy is claimed on any other imagery.")
lines.append("- The figure 0.9679 (best validation accuracy, LR=0.0001, epoch 4) is **validation-selected**;")
lines.append("  the test accuracy below is the **only unbiased figure**.")
lines.append("- Test set evaluated **exactly once** by eval_test.py; results are committed and not re-run.")
lines.append("")
lines.append("## Dataset & split")
lines.append(f"- Source URL: {SOURCE_URL}")
lines.append(f"- Split source: {results['split_source']}")
lines.append(f"- Train/val/test: 18900 / 4050 / 4050 (total 27000)")
lines.append(f"- Seeds: DATA_SPLIT_SEED={DATA_SPLIT_SEED}, TRAIN_SEED={TRAIN_SEED}")
lines.append(f"- Versions: torch {torch.__version__}, torchvision {__import__('torchvision').__version__}")
lines.append(f"- GPU available: {torch.cuda.is_available()} (CPU training)")
lines.append(f"- Input size: {INPUT_SIZE}x{INPUT_SIZE} (native EuroSAT patch; no resize to 224)")
lines.append(f"- Model: ResNet-18 ImageNet-pretrained, fc replaced for 10 classes")
lines.append(f"- Checkpoint: models/best.pt (44.79 MB, committed)")
lines.append("")
lines.append("## Overall test accuracy")
lines.append(f"- **{overall_acc:.4f}** ({int((preds==labels).sum())} / {n_total})")
lines.append("")
lines.append("## Abstention (frozen threshold = {})".format(FROZEN_THRESHOLD))
lines.append(f"- Threshold rule: {thr_record['rule']}")
lines.append(f"- Frozen threshold: {FROZEN_THRESHOLD}")
lines.append(f"- Coverage at threshold: {coverage:.4f} ({n_answered} / {n_total} cells answered)")
lines.append(f"- Accuracy on answered cells: {accuracy_on_answered:.4f}")
lines.append(f"- Cells abstained: {n_abstained}")
lines.append("")
lines.append("## Per-class precision / recall / F1 / support")
lines.append("| class | precision | recall | f1 | support |")
lines.append("|-------|-----------|--------|----|---------|")
for c in CLASSES:
    pc = results["per_class"][c]
    lines.append(f"| {c} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |")
lines.append("")
lines.append("## SeaLake (water) class — reported separately")
sl = results["sealake_water_class"]
lines.append(f"- Class: {sl['class']} (index {sl['index']})")
lines.append(f"- Precision: {sl['precision']:.4f}")
lines.append(f"- Recall: {sl['recall']:.4f}")
lines.append(f"- F1: {sl['f1']:.4f}")
lines.append(f"- Support: {sl['support']}")
lines.append(f"- Per-class accuracy: {sl['accuracy']:.4f}")
lines.append("")
lines.append("## Confusion matrix (rows=true, cols=pred)")
lines.append("| true \\ pred | " + " | ".join(CLASSES) + " |")
lines.append("|" + "---|" * (NUM_CLASSES + 1))
for i, c in enumerate(CLASSES):
    row_vals = " | ".join(str(int(v)) for v in cm[i])
    lines.append(f"| {c} | {row_vals} |")
lines.append("")
lines.append("## Reproducibility")
lines.append("- Command: `python eval_test.py` (runs once; refuses to overwrite existing results)")
lines.append("- Files: models/best.pt, splits/test.csv, reports/abstention_threshold.json")
lines.append("")
lines.append("## Limitations")
lines.append("- 64x64 Sentinel-2 patches at 10 m resolution; not field imagery.")
lines.append("- A single run with one seed (42); no confidence interval computed.")
lines.append("- Split is our own fixed stratified 70/15/15 split, not a published benchmark split.")
lines.append("- Benchmark on EuroSAT only.")

with open(RESULTS_MD, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {RESULTS_MD}")
print("")
print(f"TEST ACCURACY: {overall_acc:.4f}")
print("Step 4 complete — test set evaluated exactly once.")
