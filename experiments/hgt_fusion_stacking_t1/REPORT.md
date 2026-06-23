# HGT encoder + embedding fusion + stacking ensemble — T1a (ONE experiment)

> smoke=False  seed=42  blocks=8  features=61 (strict, no PFAS measurement, no lat/lon).

Three architectures under ONE spatial-block protocol on a MULTI-RELATIONAL well-well graph (`near` spatial k-NN cap 1.5 km, `same_subbasin_knn` intra-sub-basin k-NN cap 2 km). HGT is a relational encoder over typed EDGES — no fabricated source/compound node type (C-NODE.1/2, eval_validation.md). Evaluation is row-level, comparable to the non-graph WALL.

**WALL (committed, full run k=8):** XGB spatial AUC = 0.5878, RF = 0.6009 (experiments/baseline_t1/metrics_spatial.json).
**In-run XGB-tabular wall (same 8 folds):** 0.6879.

## Spatial-block results (row-level OOF)

| architecture | AUC OOF | AUC 95% CI | F1@OOF | PR-AUC | bal.acc | Brier | ECE |
|---|---:|---|---:|---:|---:|---:|---:|
| HGT standalone | 0.6537 | [0.634, 0.673] | 0.6143 | 0.5638 | 0.6156 | 0.3191 | 0.2892 |
| Embedding fusion (XGB + PCA-HGT) | 0.6670 | [0.650, 0.684] | 0.5639 | 0.5738 | 0.6134 | 0.2672 | 0.1832 |
| Stacking (HGT+XGB+LGBM meta) | 0.6815 | [0.663, 0.699] | 0.6378 | 0.5800 | 0.6526 | 0.2467 | 0.1283 |
| XGB-tabular (in-run wall) | 0.6879 | [0.670, 0.704] | 0.6523 | 0.5767 | 0.6687 | 0.2594 | 0.1971 |
| LGBM-tabular (ref) | 0.6793 | [0.661, 0.696] | 0.6077 | 0.5831 | 0.6422 | 0.2647 | 0.2030 |

**PCA-to-95%-variance** kept 47.8 components on average (per fold: [47, 48, 49, 48, 46, 48, 48, 48]) out of 64 HGT-embed dims.

Cross-block edges remaining (must be 0): 0.

## Δ(random − spatial) — spatial-leakage inflation (C-SPAT.6)

| architecture | spatial AUC | random AUC | Δ |
|---|---:|---:|---:|
| HGT standalone | 0.6537 | 0.8027 | +0.1490 |
| Embedding fusion | 0.6670 | 0.8572 | +0.1902 |
| Stacking | 0.6815 | 0.8962 | +0.2147 |

## Paired tests vs the WALL (8 spatial folds; Nadeau-Bengio + Wilcoxon)

| architecture | gain vs in-run wall | gain vs committed 0.588 | NB p | Wilcoxon p | verdict |
|---|---:|---:|---:|---:|---|
| hgt_standalone | -0.0642 | +0.0659 | 0.7350 | 0.8438 | no_robust_gain |
| embedding_fusion | -0.0501 | +0.0792 | 0.7683 | 0.5469 | no_robust_gain |
| stacking | -0.0470 | +0.0938 | 0.9298 | 0.9453 | no_robust_gain |

**Reality rule (eval C-CMP):** a gain counts as robust only if it is paired-
significant (p<0.05) AND exceeds the inter-fold noise bar (0.03 AUC). The honest question — does graph context beat 0.588 spatial robustly — is answered in the table above; ~0.60 spatial with no robust gain is the expected, reportable outcome.

---

## VERDICT (added by main thread — honest reading)

**No robust graph gain. Plain tabular XGBoost ≥ every graph architecture on this experiment.**

Under the SAME 8 spatial folds and the SAME 61-feature strict set (apples-to-apples,
"in-run wall"), ranking by per-fold-mean AUC (the convention used by the committed
baseline):

| model | per-fold-mean AUC sp | global-OOF AUC sp |
|---|---:|---:|
| XGB-tabular (in-run wall) | **0.643** | 0.688 |
| Stacking (HGT+XGB+LGBM) | 0.641 | 0.682 |
| Embedding fusion | 0.638 | 0.667 |
| HGT standalone | 0.624 | 0.654 |

All three graph architectures sit **below** the tabular XGBoost wall (gain_vs_in_run_wall
−0.05 to −0.06 on global-OOF), and the paired tests are nowhere near significant
(Nadeau-Bengio p = 0.74–0.93, Wilcoxon p = 0.55–0.94). This is consistent with every
prior GNN experiment on T1 (phase1-3 spatial ~0.60-0.63; gnn_hybrid_t1 0.646, also not
robust). The graph adds no robust predictive value once the tabular context is present.

### ⚠️ Do NOT read `gain_vs_committed_wall` (+0.066 / +0.079 / +0.094) as a graph win
That column compares this run against `baseline_t1/metrics_spatial.json` (0.5878), but the
two are **not comparable** for two reasons:
1. **Different feature set.** This run uses `cocontam="core"` → 61 node features; the
   committed wall used `cocontam="all"` → 96 features. The matched-feature wall is the
   *in-run* XGB (0.643 per-fold-mean), not 0.588.
2. **Different AUC definition.** The committed wall is a **per-fold-mean** AUC
   (`run_spatial_t1.py:98`, `np.nanmean` of per-fold ROC-AUC); this experiment's headline
   `roc_auc` is a **global-OOF** AUC (pooled predictions), which runs ~0.03-0.05 higher.

The only valid in-run comparison (matched folds, features, and metric) is
`gain_vs_in_run_wall` — which is **negative** for all three architectures. The module's
`verdict` field already encodes this correctly (`no_robust_gain`).

### Spatial-leakage inflation (sanity, expected)
Δ(random − spatial) is large for all archs (HGT +0.149, fusion +0.190, stacking +0.215):
the random-CV numbers (~0.80-0.90) are the usual optimistic mirage; the honest spatial
numbers (~0.62-0.68) are the deployable reality. Cross-block edges = 0 (leak-free graph).
