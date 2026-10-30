"""
Step 3 — Abstention layer (validation ONLY).

Using validation predictions only, compute coverage vs accuracy for softmax
confidence thresholds from 0.50 to 0.95 in steps of 0.05.

Threshold-selection rule (stated in advance, applied here):
    Choose the LOWEST threshold at which validation accuracy on the answered
    cells is at least 0.95.
    If no threshold meets it, report that and use NO abstention (threshold=None).

The chosen (frozen) threshold is saved to reports/abstention_threshold.json
and used exactly once by eval_test.py on the test split.

Outputs:
- reports/abstention_val.csv   (threshold, coverage, accuracy_on_answered, n_answered, n_total)
- reports/abstention_val.png   (coverage + accuracy vs threshold)
- reports/abstention_threshold.json  (frozen threshold + rule)
"""

import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT
from torchvision import transforms, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Seeds ---
TRAIN_SEED = 42
np.random.seed(TRAIN_SEED)
torch.manual_seed(TRAIN_SEED)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
SPLITS_DIR = os.path.join(ROOT, "splits")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"torch: {torch.__version__}, torchvision: {__import__('torchvision').__version__}")

INPUT_SIZE = 64  # native EuroSAT patch; no resize to 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# --- Load validation subset (same split as Step 1/2) ---
def load_split_indices(name):
    path = os.path.join(SPLITS_DIR, f"{name}.csv")
    idxs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            idxs.append(int(row["index"]))
    return sorted(idxs)

val_idx = load_split_indices("val")
print(f"Validation samples: {len(val_idx)}")

full_val = EuroSAT(root=DATA_DIR, download=False, transform=val_tf)
val_subset = Subset(full_val, val_idx)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0)

CLASSES = full_val.classes
NUM_CLASSES = len(CLASSES)

# --- Build model + load best checkpoint ---
def build_model():
    m = models.resnet18(weights=None)  # weights already in checkpoint
    m.fc = torch.nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m.to(DEVICE)

ckpt_path = os.path.join(MODELS_DIR, "best.pt")
assert os.path.exists(ckpt_path), f"Missing checkpoint {ckpt_path}"
model = build_model()
state = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print(f"Loaded checkpoint: {ckpt_path}")

# --- Collect validation predictions (softmax probs) ---
all_probs = []
all_labels = []
with torch.no_grad():
    for x, y in val_loader:
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
print(f"Total validation predictions: {n_total}")

# --- Sweep thresholds ---
thresholds = [round(t, 2) for t in np.arange(0.50, 0.951, 0.05)]
rows = []
chosen_threshold = None
for th in thresholds:
    answered = conf >= th
    n_answered = int(answered.sum())
    coverage = n_answered / n_total
    if n_answered > 0:
        acc_answered = float((preds[answered] == labels[answered]).mean())
    else:
        acc_answered = float("nan")
    rows.append({
        "threshold": th,
        "coverage": round(coverage, 6),
        "accuracy_on_answered": round(acc_answered, 6),
        "n_answered": n_answered,
        "n_total": n_total,
    })
    print(f"th={th:.2f}: coverage={coverage:.4f} ({n_answered}/{n_total}) "
          f"acc_on_answered={acc_answered:.4f}")
    # Rule: lowest threshold with val accuracy on answered >= 0.95
    if chosen_threshold is None and not np.isnan(acc_answered) and acc_answered >= 0.95:
        chosen_threshold = th

# --- Save abstention_val.csv ---
csv_path = os.path.join(REPORTS_DIR, "abstention_val.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["threshold", "coverage", "accuracy_on_answered",
                                           "n_answered", "n_total"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\nWrote {csv_path}")

# --- Plot coverage + accuracy vs threshold ---
ths = [r["threshold"] for r in rows]
cov = [r["coverage"] for r in rows]
acc = [r["accuracy_on_answered"] for r in rows]

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(ths, cov, "o-", color="tab:blue", label="coverage")
ax1.set_xlabel("confidence threshold")
ax1.set_ylabel("coverage (fraction answered)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(ths, acc, "s-", color="tab:red", label="accuracy on answered")
ax2.set_ylabel("accuracy on answered cells", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylim(0, 1.05)

plt.title("EuroSAT abstention (validation): coverage & accuracy vs confidence threshold")
if chosen_threshold is not None:
    plt.axvline(chosen_threshold, color="green", linestyle="--",
                label=f"chosen threshold = {chosen_threshold:.2f}")
fig.tight_layout()
plot_path = os.path.join(REPORTS_DIR, "abstention_val.png")
plt.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Wrote {plot_path}")

# --- Threshold selection outcome ---
if chosen_threshold is None:
    print("\nNO threshold reached val accuracy >= 0.95 on answered cells.")
    print("-> Using NO abstention (threshold=None).")
else:
    print(f"\nChosen (frozen) threshold: {chosen_threshold:.2f}")
    row = [r for r in rows if r["threshold"] == chosen_threshold][0]
    print(f"  coverage={row['coverage']}, acc_on_answered={row['accuracy_on_answered']}, "
          f"n_answered={row['n_answered']}/{row['n_total']}")

# --- Save frozen threshold for Step 4 ---
threshold_record = {
    "rule": ("lowest threshold at which validation accuracy on answered cells >= 0.95; "
             "if none, no abstention"),
    "chosen_threshold": chosen_threshold,
    "thresholds_swept": thresholds,
    "source": "validation predictions only (test split not used)",
}
thr_path = os.path.join(REPORTS_DIR, "abstention_threshold.json")
with open(thr_path, "w") as f:
    json.dump(threshold_record, f, indent=2)
print(f"Wrote {thr_path}")

print("")
print("STOP at end of Step 3 — abstention layer complete (validation only).")
