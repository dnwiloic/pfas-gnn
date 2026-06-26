# HGT encoder + embedding fusion + stacking ensemble — T1a (ONE experiment)

> smoke=True  seed=42  blocks=3  features=98 (strict, no PFAS measurement, no lat/lon).

Three architectures under ONE spatial-block protocol on a MULTI-RELATIONAL well-well graph (`near` spatial k-NN cap 1.5 km, `same_subbasin_knn` intra-sub-basin k-NN cap 2 km). HGT is a relational encoder over typed EDGES — no fabricated source/compound node type (C-NODE.1/2, eval_validation.md). Evaluation is row-level, comparable to the non-graph WALL.

**WALL (committed, full run k=8):** XGB spatial AUC = 0.6528, RF = 0.6493 (experiments/baseline_t1/metrics_spatial.json).
**In-run XGB-tabular wall (same 8 folds):** 0.5971.

## Spatial-block results (row-level OOF)

| architecture | AUC OOF | AUC 95% CI | F1@OOF | PR-AUC | bal.acc | Brier | ECE |
|---|---:|---|---:|---:|---:|---:|---:|
| HGT standalone | 0.5287 | [0.441, 0.621] | 0.4362 | 0.4533 | 0.5444 | 0.2897 | 0.2182 |
| Embedding fusion (XGB + PCA-HGT) | 0.5840 | [0.503, 0.667] | 0.4567 | 0.4619 | 0.5621 | 0.2864 | 0.2269 |
| Stacking (HGT+XGB+LGBM meta) | 0.5089 | [0.430, 0.605] | 0.3826 | 0.4207 | 0.5514 | 0.2983 | 0.2479 |
| XGB-tabular (in-run wall) | 0.5971 | [0.519, 0.680] | 0.4312 | 0.4783 | 0.5603 | 0.2818 | 0.2166 |
| LGBM-tabular (ref) | 0.5664 | [0.487, 0.657] | 0.4162 | 0.4752 | 0.5665 | 0.2959 | 0.2414 |

**PCA-to-95%-variance** kept 21.3 components on average (per fold: [1, 62, 1]) out of 128 HGT-embed dims.

Cross-block edges remaining (must be 0): 0.

## Δ(random − spatial) — spatial-leakage inflation (C-SPAT.6)

| architecture | spatial AUC | random AUC | Δ |
|---|---:|---:|---:|
| HGT standalone | 0.5287 | 0.6678 | +0.1391 |
| Embedding fusion | 0.5840 | 0.6830 | +0.0989 |
| Stacking | 0.5089 | 0.7287 | +0.2198 |

## Paired tests vs the WALL (8 spatial folds; Nadeau-Bengio + Wilcoxon)

| architecture | gain vs in-run wall | gain vs committed 0.588 | NB p | Wilcoxon p | verdict |
|---|---:|---:|---:|---:|---|
| hgt_standalone | +0.1081 | -0.1241 | 0.7753 | nan | no_robust_gain |
| embedding_fusion | +0.0552 | -0.0688 | 0.3580 | nan | no_robust_gain |
| stacking | +0.0277 | -0.1439 | 0.2580 | nan | no_robust_gain |

**Reality rule (eval C-CMP):** a gain counts as robust only if it is paired-
significant (p<0.05) AND exceeds the inter-fold noise bar (0.03 AUC). The honest question — does graph context beat 0.588 spatial robustly — is answered in the table above; ~0.60 spatial with no robust gain is the expected, reportable outcome.
