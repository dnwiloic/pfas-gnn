# HGT encoder + embedding fusion + stacking ensemble — T1a (ONE experiment)

> smoke=False  seed=42  blocks=8  features=98 (strict, no PFAS measurement, no lat/lon).

Three architectures under ONE spatial-block protocol on a MULTI-RELATIONAL well-well graph (`near` spatial k-NN cap 1.5 km, `same_subbasin_knn` intra-sub-basin k-NN cap 2 km). HGT is a relational encoder over typed EDGES — no fabricated source/compound node type (C-NODE.1/2, eval_validation.md). Evaluation is row-level, comparable to the non-graph WALL.

**WALL (committed, full run k=8):** XGB spatial AUC = 0.6528, RF = 0.6493 (experiments/baseline_t1/metrics_spatial.json).
**In-run XGB-tabular wall (same 8 folds):** 0.7439.

## Spatial-block results (row-level OOF)

| architecture | AUC OOF | AUC 95% CI | F1@OOF | PR-AUC | bal.acc | Brier | ECE |
|---|---:|---|---:|---:|---:|---:|---:|
| HGT standalone | 0.7070 | [0.688, 0.727] | 0.6200 | 0.6278 | 0.6687 | 0.2672 | 0.2216 |
| Embedding fusion (XGB + PCA-HGT) | 0.7115 | [0.694, 0.729] | 0.6104 | 0.6351 | 0.6564 | 0.2405 | 0.1476 |
| Stacking (HGT+XGB+LGBM meta) | 0.7315 | [0.715, 0.747] | 0.6551 | 0.6570 | 0.6852 | 0.2194 | 0.1008 |
| XGB-tabular (in-run wall) | 0.7439 | [0.727, 0.760] | 0.6497 | 0.6837 | 0.6836 | 0.2239 | 0.1278 |
| LGBM-tabular (ref) | 0.7084 | [0.692, 0.725] | 0.6085 | 0.6384 | 0.6520 | 0.2504 | 0.1770 |

**PCA-to-95%-variance** kept 20.1 components on average (per fold: [17, 15, 13, 17, 48, 17, 16, 18]) out of 64 HGT-embed dims.

Cross-block edges remaining (must be 0): 0.

## Δ(random − spatial) — spatial-leakage inflation (C-SPAT.6)

| architecture | spatial AUC | random AUC | Δ |
|---|---:|---:|---:|
| HGT standalone | 0.7070 | 0.8165 | +0.1095 |
| Embedding fusion | 0.7115 | 0.8678 | +0.1563 |
| Stacking | 0.7315 | 0.9045 | +0.1730 |

## Paired tests vs the WALL (8 spatial folds; Nadeau-Bengio + Wilcoxon)

| architecture | gain vs in-run wall | gain vs committed 0.588 | NB p | Wilcoxon p | verdict |
|---|---:|---:|---:|---:|---|
| hgt_standalone | -0.0839 | +0.0542 | 0.5144 | 0.3828 | no_robust_gain |
| embedding_fusion | -0.0510 | +0.0587 | 0.9728 | 0.9453 | no_robust_gain |
| stacking | -0.0526 | +0.0787 | 0.9125 | 0.8438 | no_robust_gain |

**Reality rule (eval C-CMP):** a gain counts as robust only if it is paired-
significant (p<0.05) AND exceeds the inter-fold noise bar (0.03 AUC). The honest question — does graph context beat 0.588 spatial robustly — is answered in the table above; ~0.60 spatial with no robust gain is the expected, reportable outcome.
