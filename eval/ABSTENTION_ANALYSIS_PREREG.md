# Pre-registration — Abstention Layer Exploratory Analysis

**Status:** PRE-REGISTRATION (committed before any analysis code runs).  
**Repo:** eurosat-decision  
**Commit to pre-register against:** will be recorded at commit time  
**Frozen weights:** models/best.pt (sha256 c6c5eed4be400498d8c43941e1115da72dcde94a37d73b358e6182923201716c)  
**Frozen threshold:** 0.50 (from reports/abstention_threshold.json; rule: lowest threshold at which validation accuracy on answered cells >= 0.50; chosen on validation only)  
**Test split:** 4,050 samples (our own fixed stratified 70/15/15 split, seed 42; NOT a published benchmark split)  
**Test-set accuracy already reported:** 0.9627 (3,899 / 4,050) — the only unbiased figure  
**Label order:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake (indices 0–9)

---

## Questions (pre-registered)

**Question 1:** On the test split, for the classes SeaLake, River, Forest and HerbaceousVegetation, how often is the model confidently wrong (max softmax >= 0.50 and prediction != true label)? Report confident-error count and confident-error rate (confident errors / class support), per class.

**Question 2:** Does abstention at 0.50 remove those confident errors, and at what coverage cost, per class? Report, per class: total, answered, abstained, correct among answered, confident errors among answered, and confidence = confident_errors / answered (rate among answered cells). Also report coverage = answered / total.

**Question 3:** For the class pairs SeaLake–River and Forest–HerbaceousVegetation, what is the error rate among answered cells, and how does it change across thresholds 0.50 to 0.95 in steps of 0.05? For each pair and each threshold, report: answered count, error count among answered, error rate = errors / answered, coverage = answered / pair total, and TP/FP-style counts (true positives = correct answered; false positives = confident wrong answered).

---

## Statements (pre-registered, binding)

- The abstention threshold rule (0.50) is frozen. This analysis cannot choose a new threshold. Any threshold sweep is descriptive only.
- The test split was already used for the reported 0.9627 accuracy. This analysis is exploratory and gives no new accuracy claim. All outputs are labelled "exploratory, test split reused, not a new accuracy measure."
- Stated outcome: report both outcomes — whether the abstention layer does catch confident errors on the pre-registered pairs, or whether it does not.

---

## Analysis plan (to be executed after this pre-registration is committed)

1. Regenerate per-sample softmax outputs for the 4,050 test files using models/best.pt and the training transform (ToTensor + ImageNet normalize, no resize), saving to analysis/test_softmax.csv. Verify overall accuracy == 3,899 / 4,050; stop and report mismatch if not.
2. Compute per-class counts at threshold 0.50: total, answered, abstained, correct among answered, confident errors (answered and wrong), confident-error rate per class.
3. For pairs SeaLake–River and Forest–HerbaceousVegetation: error rate among answered cells at thresholds 0.50–0.95 (step 0.05), with counts.
4. Coverage and answered accuracy at each threshold 0.50–0.95.
5. Outputs: analysis/abstention_by_class.csv and analysis/abstention_pairs.png (plot title includes "exploratory, test split reused").
6. Report: analysis/ABSTENTION_REPORT.md with questions, answers with all counts, outcome (caught or not caught), and limitations.

---
