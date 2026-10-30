# Abstention Analysis Report (exploratory)

**Status:** EXPLORATORY — test split reused, not a new accuracy measure.  
**Repo:** eurosat-decision  
**Pre-registration commit:** c74885e (eval/ABSTENTION_ANALYSIS_PREREG.md)  
**Analysis commit:** d2db99d (analysis: abstention by class (exploratory))  
**Frozen weights:** models/best.pt (sha256 c6c5eed4be400498d8c43941e1115da72dcde94a37d73b358e6182923201716c)  
**Frozen threshold:** 0.50 (rule: lowest threshold at which validation accuracy on answered cells >= 0.95; chosen on validation only; NOT re-tuned here)  
**Per-sample predictions:** regenerated to analysis/test_softmax.csv using the frozen weights and training transform (ToTensor + ImageNet normalize, no resize). Regenerated predictions match the aggregate 3,899 / 4,050 = 0.9627 test result. Per-file originals were not saved; regenerated outputs are used throughout.  
**Test split:** 4,050 samples (our own fixed stratified 70/15/15 split, seed 42; NOT a published benchmark split)  
**Label order:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake (indices 0–9)

---

## Outcome: Frozen threshold 0.50 (pre-registered result first)

At the **frozen threshold of 0.50**, the abstention layer removes almost nothing: only **25 of 4,050 cells abstained** (coverage 0.9938). Nearly all confident errors remain answered:

| class | support | confident errors | confident-error rate | confident errors among answered | answered accuracy |
|-------|---------|------------------|---------------------|---------------------------------|-------------------|
| SeaLake | 450 | 2 | 0.0044 | 2 | 0.9956 |
| River | 375 | 17 | 0.0453 | 17 | 0.9545 |
| Forest | 450 | 3 | 0.0067 | 3 | 0.9933 |
| HerbaceousVegetation | 450 | 28 | 0.0622 | 28 | 0.9374 |

**The frozen layer did NOT catch these confident errors.** At 0.50, HerbaceousVegetation still has 28 confident errors answered (6.22% of its support), River has 17 (4.53%), Forest has 3 (0.67%), and SeaLake has 2 (0.44%). Abstention at 0.50 removes only 25 cells total — essentially none of the confident errors are caught at the frozen rule.

---

## Pre-registered questions (descriptive threshold sweep)

**Question 1:** On the test split, for the classes SeaLake, River, Forest and HerbaceousVegetation, how often is the model confidently wrong (max softmax >= 0.50 and prediction != true label)?

| class | support | confident errors | confident-error rate (confident errors / support) |
|-------|---------|------------------|---------------------------------------------------|
| SeaLake | 450 | 2 | 0.0044 |
| River | 375 | 17 | 0.0453 |
| Forest | 450 | 3 | 0.0067 |
| HerbaceousVegetation | 450 | 28 | 0.0622 |

HerbaceousVegetation has the highest confident-error rate (6.22%), followed by River (4.53%). SeaLake and Forest are low (0.44%, 0.67%).

**Question 2:** Does abstention at 0.50 remove those confident errors, and at what coverage cost, per class?

| class | total | answered | abstained | correct among answered | confident errors among answered | coverage (answered/total) | answered accuracy |
|-------|-------|----------|-----------|------------------------|---------------------------------|---------------------------|-------------------|
| SeaLake | 450 | 450 | 0 | 448 | 2 | 1.0000 | 0.9956 |
| River | 375 | 374 | 1 | 357 | 17 | 0.9973 | 0.9545 |
| Forest | 450 | 449 | 1 | 446 | 3 | 0.9978 | 0.9933 |
| HerbaceousVegetation | 450 | 447 | 3 | 419 | 28 | 0.9933 | 0.9374 |

At threshold 0.50, abstention removes almost nothing (only 1–3 cells per class abstained), so nearly all confident errors remain answered. HerbaceousVegetation still has 28 confident errors answered (6.22% of its support). Coverage is near 1.00 for all four classes, so the coverage cost of abstention at 0.50 is minimal — but so is the error removal.

**Question 3:** For the class pairs SeaLake–River and Forest–HerbaceousVegetation, what is the error rate among answered cells, and how does it change across thresholds 0.50 to 0.95 in steps of 0.05?

**Pair SeaLake–River (total 825):**

| threshold | answered | errors | error rate (errors/answered) | coverage (answered/pair total) | tp (correct answered) | fp (confident wrong answered) |
|-----------|----------|--------|------------------------------|--------------------------------|-----------------------|-------------------------------|
| 0.50 | 824 | 19 | 0.0231 | 0.9988 | 805 | 19 |
| 0.55 | 818 | 18 | 0.0220 | 0.9915 | 800 | 18 |
| 0.60 | 808 | 13 | 0.0161 | 0.9794 | 795 | 13 |
| 0.65 | 804 | 12 | 0.0149 | 0.9745 | 792 | 12 |
| 0.70 | 800 | 10 | 0.0125 | 0.9697 | 790 | 10 |
| 0.75 | 792 | 6 | 0.0076 | 0.9600 | 786 | 6 |
| 0.80 | 787 | 4 | 0.0051 | 0.9539 | 783 | 4 |
| 0.85 | 779 | 3 | 0.0039 | 0.9442 | 776 | 3 |
| 0.90 | 769 | 1 | 0.0013 | 0.9321 | 768 | 1 |
| 0.95 | 759 | 1 | 0.0013 | 0.9200 | 758 | 1 |

**Pair Forest–HerbaceousVegetation (total 900):**

| threshold | answered | errors | error rate (errors/answered) | coverage (answered/pair total) | tp (correct answered) | fp (confident wrong answered) |
|-----------|----------|--------|------------------------------|--------------------------------|-----------------------|-------------------------------|
| 0.50 | 896 | 31 | 0.0346 | 0.9956 | 865 | 31 |
| 0.55 | 888 | 27 | 0.0304 | 0.9867 | 861 | 27 |
| 0.60 | 886 | 26 | 0.0293 | 0.9844 | 860 | 26 |
| 0.65 | 874 | 22 | 0.0252 | 0.9711 | 852 | 22 |
| 0.70 | 863 | 16 | 0.0185 | 0.9589 | 847 | 16 |
| 0.75 | 853 | 13 | 0.0152 | 0.9478 | 840 | 13 |
| 0.80 | 846 | 12 | 0.0142 | 0.9400 | 834 | 12 |
| 0.85 | 831 | 6 | 0.0072 | 0.9233 | 825 | 6 |
| 0.90 | 816 | 4 | 0.0049 | 0.9067 | 812 | 4 |
| 0.95 | 773 | 1 | 0.0013 | 0.8589 | 772 | 1 |

For both pairs, raising the threshold monotonically reduces the error rate among answered cells and the count of confident wrong answers (fp), at a steady coverage cost. **This threshold sweep is descriptive only — the 0.50 threshold was chosen on validation and is not re-tuned here.**

---

## Overall coverage and answered accuracy (all classes, descriptive)

| threshold | answered | abstained | coverage | answered accuracy |
|-----------|----------|-----------|----------|-------------------|
| 0.50 | 4,025 | 25 | 0.9938 | 0.9650 |
| 0.55 | 3,993 | 57 | 0.9859 | 0.9677 |
| 0.60 | 3,963 | 87 | 0.9785 | 0.9707 |
| 0.65 | 3,930 | 120 | 0.9704 | 0.9740 |
| 0.70 | 3,889 | 161 | 0.9602 | 0.9789 |
| 0.75 | 3,843 | 207 | 0.9489 | 0.9826 |
| 0.80 | 3,801 | 249 | 0.9385 | 0.9853 |
| 0.85 | 3,754 | 296 | 0.9269 | 0.9880 |
| 0.90 | 3,682 | 368 | 0.9091 | 0.9905 |
| 0.95 | 3,535 | 515 | 0.8728 | 0.9941 |

---

## Outcome summary

At the **frozen threshold of 0.50**, the abstention layer does **not** catch confident errors — only 25 of 4,050 cells are abstained, and all confident errors (SeaLake 2, River 17, Forest 3, HerbaceousVegetation 28) remain answered. The layer catches confident errors **only when the threshold is raised above 0.50** (e.g., at 0.90, SeaLake–River drops to 1 error out of 769 answered; Forest–HerbaceousVegetation drops to 4 out of 816), but those higher thresholds were not the frozen rule and come at a substantial coverage cost (coverage falls from ~0.999 to ~0.86–0.92 across the pairs).

This matches the pre-registered statement: report both outcomes — the abstention layer does not catch confident errors at the frozen 0.50 threshold, but does reduce confident errors at higher thresholds.

---

## Limitations

- **Single seed:** one run with seed 42; no confidence interval computed.
- **64x64 patches:** Sentinel-2 at 10 m resolution; benchmark imagery, not field imagery.
- **Test split reused:** the test split was already used for the reported 0.9627 accuracy; this analysis is exploratory and gives no new accuracy claim.
- **Pairs chosen by pre-registration:** SeaLake–River and Forest–HerbaceousVegetation were selected before seeing results; other pairs were not analyzed.
- **Threshold frozen:** the 0.50 threshold was chosen on validation only; this analysis does not re-tune it. Any threshold sweep is descriptive only.
- **Not a new accuracy measure:** all figures are descriptive statistics on the reused test split.

---

## Outputs

- `analysis/test_softmax.csv` — regenerated per-sample softmax for all 4,050 test files; regenerated output matches the aggregate 3,899/4,050 (0.9627). Per-file originals were not saved.
- `analysis/abstention_by_class.csv` — per-class counts at threshold 0.50
- `analysis/abstention_pairs_by_threshold.csv` — pair error rates across thresholds
- `analysis/abstention_overall_by_threshold.csv` — overall coverage/answered accuracy across thresholds
- `analysis/abstention_pairs.png` — plot titled "exploratory, test split reused"

---
