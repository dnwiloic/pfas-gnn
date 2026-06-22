# HGT / R-GCN multi-relational T1 — rgcn

Multi-relational encoders over a HOMOGENEOUS well-well graph with TWO edge types (`near` spatial k-NN cap 1.5 km, `same_subbasin_knn` intra-sub-basin k-NN cap 2 km). Eval-validated design (experiments/hgt_rgcn_t1/eval_validation.md): no fabricated source node type; HGT/R-GCN used purely as relational encoders.

- model: **rgcn**  smoke=True  seed=42
- features: 61 cols, include_location=False (C-LOC.1: lat/lon NOT node features)
- inductive (C-SPAT.4): True
- relations: ['near', 'same_subbasin_knn']

## Results (row-level OOF, comparable to the non-graph wall and gnn_phase1)

| regime | AUC OOF | AUC 95% CI | AUC mean±std (folds) | F1@OOF | PR-AUC | bal.acc | Brier | ECE | xblock |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| spatial | 0.6464 | [0.544, 0.733] | 0.5441±0.0635 | 0.6077 | 0.5002 | 0.6638 | 0.2447 | 0.1669 | 0 |
| random | 0.6625 | [0.550, 0.757] | 0.6630±0.0662 | 0.5626 | 0.5841 | 0.6506 | 0.2421 | 0.1694 | 0 |

**Δ(random − spatial) AUC = 0.0161** (C-SPAT.6: the spatial-leakage inflation; a large Δ confirms random split is an optimistic artefact, not a real generalisation gain).

## Leakage guard (C-SPAT.2 / C-SPAT.5)
Cross-block edges are cut SEPARATELY per relation and asserted to 0. Total residual cross-block edges across all folds/regimes: 0 (must be 0).

## Calibration (C-CAL)
Brier + ECE reported above; per-bin reliability curve stored in metrics.json under `reliability_curve`.

## Positioning
gnn_phase1 single-relation spatial AUC: GraphSAGE 0.618±0.067, GCN 0.624±0.074. The honest comparison is THIS spatial AUC vs that wall; gains below the inter-fold σ (~0.06–0.07) are within noise and not claimed (eval C-CMP).
