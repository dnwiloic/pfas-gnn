"""V1 — INDUCTIVE heterogeneous GraphSAGE encoder vs HGT, head-to-head on T1a.

Motivation
----------
`experiments/hgt_fusion_stacking_t1` showed HGT-standalone spatial AUC (0.6537 OOF global /
0.624 per-fold-mean) UNDER the in-run tabular XGB wall and with a large random-minus-spatial
Δ (+0.149) — a signature of SPATIAL OVERFITTING of the attention encoder. V1 asks a single
sharp question: does replacing the HGT encoder by an INDUCTIVE heterogeneous GraphSAGE
(HeteroConv{relation: SAGEConv}) WITH explicit regularisation (neighbour sampling, DropEdge,
GraphNorm) reduce that overfitting and/or change the verdict vs the wall?

Design (head-to-head, ONE protocol)
-----------------------------------
Both encoders are driven through the SAME, eval-validated fold trainer
`gnn_hetero_t1.train_eval_fold` on the SAME multi-relational well-well graph (two REAL edge
types: `near` k-NN cap 1.5 km, `same_subbasin_knn` intra-sub-basin k-NN cap 2 km — NO
fabricated node type, hgt_rgcn_t1/eval_validation.md C-NODE). The only thing that changes
between the two arms is the encoder:

  * arm "hgt"            : HGTConv (the reference, exactly as in hgt_fusion_stacking_t1).
  * arm "hetero_sage_v1" : HeteroConv{near: SAGEConv(mean), same_subbasin_knn: SAGEConv(mean)},
                           aggr='sum', embedding 64-D (paper mirror), INDUCTIVE, with
                           DropEdge (per-relation, train-time), GraphNorm, and per-epoch
                           neighbour sampling (fan-out cap on incoming edges per relation).

Everything else is identical and inherited from the validated socle:
  * C-SPAT.1 spatial-block CV k=8 at the well level; grouped-random k=8 ONLY for the Δ arm.
  * C-SPAT.2/5 inter-block edges cut PER RELATION + asserted 0 (the trainer crashes otherwise).
  * C-SPAT.4 inductive: a test well aggregates only from its TRAIN neighbours of the same
    block (train-train edges at fit time; cross-block-free edges at score time).
  * C-LOC.1 61 strict context features, NO PFAS measurement, NO lat/lon.
  * C-THR F1 threshold from OOF/VAL probabilities only, never the test block.
  * C-CAL Brier + ECE + reliability curve on the aggregated OOF probas.
  * C-CMP bootstrap CI (by well) + paired Nadeau-Bengio + Wilcoxon on the 8 spatial folds.

Comparisons reported
---------------------
  * hetero-SAGE-v1 vs HGT (paired on the 8 spatial folds): the V1 question.
  * each encoder vs the IN-RUN XGB-tabular wall (same 8 folds, same 61 features, same
    frequency encoding) — the ONLY apples-to-apples wall (hgt_fusion_stacking_t1/
    eval_validation.md Point 5; the committed 0.588 scalar is NOT comparable).
  * OVERFITTING DIAGNOSTIC: per-encoder fit-vs-val AUC/F1 gap, averaged over folds — does
    the regularised inductive SAGE shrink the HGT generalisation gap?
  * Δ(random − spatial) per encoder — the spatial-leakage inflation (C-SPAT.6).

Torch / xgboost imported lazily; CPU smoke-testable (`run(smoke=True)`).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import gnn_hetero_t1 as H
from . import graph as G
from . import metrics as M
from . import splits as S
from . import targets as T
from .hybrid import (
    _bootstrap_ci_by_well,
    _corrected_resampled_ttest,
    _wilcoxon_paired,
    _ece,
    _optimal_threshold_f1,
    _make_xgb,
)

SEED = C.SEED

# WALL (committed, experiments/baseline_t1/metrics_spatial.json) — for context only; the
# eval-validated apples-to-apples comparison is the IN-RUN xgb wall computed below.
WALL_XGB_SPATIAL_AUC = 0.5877739078600925
WALL_RF_SPATIAL_AUC = 0.6009263696712559
NOISE_THRESHOLD = 0.03               # inter-fold reality bar (eval C-CMP)

# Encoders compared head-to-head.
ENCODERS = ("hgt", "hetero_sage_v1")

# V1 regularisation knobs (the explicitly-requested ones); HGT ignores them by construction.
V1_DROP_EDGE = 0.2
V1_USE_GRAPHNORM = True
V1_NEIGHBOR_FANOUT = 10

# smoke / full params
SMOKE_N_WELLS = 500
SMOKE_BLOCKS = 3
SMOKE_EPOCHS = 15
SMOKE_PATIENCE = 6
FULL_BLOCKS = C.N_SPATIAL_BLOCKS     # 8
FULL_EPOCHS = 400
FULL_PATIENCE = 50


# ============================================================= tabular wall (per-well)
def _tabular_well_matrix(df, well_ids, feature_cols, train_mask):
    """Per-well tabular matrix, FeaturePipeline FIT ON TRAIN WELLS ONLY (anti-leak).
    Frequency encoding -> no y needed. Mirrors hgt_fusion_stacking_t1._tabular_well_matrix
    so the in-run wall here is byte-for-byte the SAME wall those experiments used."""
    from . import features as F
    import warnings
    wf = G.aggregate_to_wells(df, well_ids, feature_cols)
    pipe = F.FeaturePipeline(feature_cols, encode="frequency")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit_transform(wf.iloc[train_mask], None)
        X, names = pipe.transform(wf)
    return X.astype(np.float32), list(names)


# ============================================================= one encoder, one regime
def _encoder_kw(encoder):
    """Encoder-specific kwargs for H.train_eval_fold. HGT = reference; hetero_sage_v1 adds
    the requested regularisation (DropEdge / GraphNorm / neighbour sampling)."""
    if encoder == "hetero_sage_v1":
        return dict(name="hetero_sage_v1", drop_edge=V1_DROP_EDGE,
                    use_graphnorm=V1_USE_GRAPHNORM, neighbor_fanout=V1_NEIGHBOR_FANOUT)
    return dict(name=encoder)


def run_encoder_regime(df, *, encoder, regime, feature_cols, n_blocks, hidden, layers,
                       dropout, heads, k_spatial, cap_km_spatial, k_subbasin, cap_km_subbasin,
                       max_epochs, patience, lr, weight_decay, inductive, smoke,
                       seed=SEED, verbose=False):
    """Leave-one-block-out CV for ONE encoder under ONE regime. Returns per-well OOF probas,
    per-fold row-level AUCs, the overfitting-diagnostic per fold, and the cross-block audit.
    Also returns the in-run XGB-tabular OOF probas computed on the SAME folds (the wall)."""
    well_ids, coords, well_to_node = G.well_table(df)
    subbasin = G.well_subbasin(df, well_ids)
    y_row = T.build_T1a(df).to_numpy()
    y_well = G.well_majority_target(df, y_row, well_ids)
    row_to_node = df[C.WELL_ID].map(well_to_node).to_numpy().astype(np.int64)

    if regime == "spatial":
        fold_block_row = S.spatial_block_folds(df, k=n_blocks, seed=seed)
    else:
        fold_block_row = S.group_random_folds(df, k=n_blocks, seed=seed)
    bdf = pd.DataFrame({"w": df[C.WELL_ID].to_numpy(), "b": fold_block_row})
    if int((bdf.groupby("w")["b"].nunique() > 1).sum()):
        raise AssertionError("a well straddles >1 block")
    node_block = bdf.groupby("w")["b"].agg(lambda s: int(s.iloc[0])).reindex(well_ids)\
        .to_numpy().astype(int)

    n = len(well_ids)
    gnn_proba = np.full(n, np.nan)
    wall_proba = np.full(n, np.nan)

    model_kw = dict(hidden=hidden, layers=layers, dropout=dropout, heads=heads)
    train_kw = dict(k_spatial=k_spatial, cap_km_spatial=cap_km_spatial,
                    k_subbasin=k_subbasin, cap_km_subbasin=cap_km_subbasin,
                    lr=lr, weight_decay=weight_decay, max_epochs=max_epochs,
                    patience=patience, inductive=inductive)
    enc_kw = _encoder_kw(encoder)

    blocks = sorted(set(node_block.tolist()))
    total_cross = 0
    fold_diag = []
    for b in blocks:
        test_nodes = node_block == b
        train_mask = ~test_nodes
        fr, proba_node, _emb = H.train_eval_fold(
            df, well_ids, y_well, node_block, b, feature_cols,
            coords=coords, subbasin=subbasin, y_row=y_row, seed=seed, verbose=False,
            **enc_kw, **model_kw, **train_kw)
        gnn_proba[test_nodes] = proba_node[test_nodes]
        total_cross += int(fr.audit["n_cross_block_near"] + fr.audit["n_cross_block_subbasin"])
        d = dict(fr.train_diag); d["fold"] = int(b)
        d["fold_test_auc"] = float(fr.metrics_spatial.get("roc_auc", float("nan")))
        d["best_epoch"] = int(fr.best_epoch)
        fold_diag.append(d)

        # in-run XGB-tabular wall on the SAME fold (only needed once; compute under the
        # spatial regime where the wall comparison lives, and for random for the Δ symmetry)
        import warnings
        X_tab, _ = _tabular_well_matrix(df, well_ids, feature_cols, train_mask)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xgb = _make_xgb(smoke=smoke, prevalence=float(y_well[train_mask].mean()))
            xgb.fit(X_tab[train_mask], y_well[train_mask])
            wall_proba[test_nodes] = xgb.predict_proba(X_tab[test_nodes])[:, 1]

        if verbose:
            a = fr.audit
            print(f"[{encoder}/{regime}] block {b}: AUC(fold)={d['fold_test_auc']:.4f} "
                  f"fit_auc={d.get('fit_auc'):.3f} val_auc={d.get('val_auc'):.3f} "
                  f"xblock(near={a['n_cross_block_near']},sub={a['n_cross_block_subbasin']})")

    return {
        "encoder": encoder, "regime": regime, "n_wells": n, "n_blocks": len(blocks),
        "node_block": node_block, "row_to_node": row_to_node, "y_well": y_well,
        "gnn_proba": gnn_proba, "wall_proba": wall_proba, "y_row": y_row,
        "fold_diag": fold_diag, "n_cross_block_total": int(total_cross),
        "well_ids_row": df[C.WELL_ID].to_numpy(),
    }


# ============================================================= row-level scoring
def _row_metrics(reg, proba_well, *, thr=None):
    """Broadcast per-well OOF proba to rows, threshold from OOF wells (C-THR), return the
    full row-level metric set + by-well bootstrap CI."""
    valid = ~np.isnan(proba_well)
    if thr is None:
        thr = _optimal_threshold_f1(reg["y_well"][valid].astype(int), proba_well[valid])
    proba_row = proba_well[reg["row_to_node"]]
    rmask = valid[reg["row_to_node"]]
    yt = np.asarray(reg["y_row"])[rmask].astype(int)
    pt = proba_row[rmask]
    mets = M.binary_metrics(yt, pt, threshold=thr)
    mets["ece"] = _ece(yt, pt)
    wells_row = reg["well_ids_row"][rmask]
    ci = _bootstrap_ci_by_well(yt, pt, wells_row, seed=SEED)
    reliability = _reliability_curve(yt, pt)
    return mets, ci, thr, reliability


def _per_fold_aucs(reg, proba_well):
    """Per-block row-level AUC list (paired-test unit). One AUC per held-out block."""
    from sklearn.metrics import roc_auc_score
    proba_row = proba_well[reg["row_to_node"]]
    block_row = reg["node_block"][reg["row_to_node"]]
    aucs = []
    for b in sorted(set(reg["node_block"].tolist())):
        m = (block_row == b) & ~np.isnan(proba_row)
        yt = np.asarray(reg["y_row"])[m].astype(int)
        if len(np.unique(yt)) < 2:
            aucs.append(float("nan")); continue
        aucs.append(float(roc_auc_score(yt, proba_row[m])))
    return aucs


def _reliability_curve(y_true, proba, n_bins=10):
    y_true = np.asarray(y_true); proba = np.asarray(proba)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() == 0:
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": 0,
                         "mean_conf": float("nan"), "frac_pos": float("nan")})
        else:
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(mask.sum()),
                         "mean_conf": float(proba[mask].mean()),
                         "frac_pos": float(y_true[mask].mean())})
    return rows


def _agg_overfit(fold_diag):
    """Mean fit/val AUC + gap over folds (the overfitting diagnostic headline)."""
    def _m(key):
        vals = [d[key] for d in fold_diag if np.isfinite(d.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")
    return {
        "fit_auc_mean": _m("fit_auc"), "val_auc_mean": _m("val_auc"),
        "gap_auc_mean": _m("gap_auc_fit_minus_val"),
        "fit_f1_mean": _m("fit_f1"), "val_f1_mean": _m("val_f1"),
        "gap_f1_mean": _m("gap_f1_fit_minus_val"),
        "best_epoch_mean": float(np.mean([d["best_epoch"] for d in fold_diag])),
    }


# ============================================================= paired tests
def _paired(a_folds, b_folds, n_tr, n_te):
    af = [x for x in a_folds if np.isfinite(x)]
    bf = [x for x in b_folds if np.isfinite(x)]
    k = min(len(af), len(bf))
    if k < 2:
        return {"nadeau_bengio": {"p": float("nan")}, "wilcoxon": {"p": float("nan")},
                "mean_diff": float("nan")}
    nb = _corrected_resampled_ttest(af[:k], bf[:k], n_tr, n_te)
    wc = _wilcoxon_paired(af[:k], bf[:k])
    return {"nadeau_bengio": nb, "wilcoxon": wc,
            "mean_diff": float(np.nanmean(np.array(af[:k]) - np.array(bf[:k])))}


# ============================================================= main entry
def run(df=None, *, smoke=False, n_blocks=None, hidden=64, layers=2, dropout=0.3, heads=4,
        k_spatial=8, cap_km_spatial=1.5, k_subbasin=8, cap_km_subbasin=2.0,
        max_epochs=None, patience=None, lr=5e-3, weight_decay=5e-4, inductive=True,
        compute_delta=True, write=True, exp_dir=None, seed=SEED, verbose=False):
    """Head-to-head HGT vs inductive hetero-SAGE-v1 on T1a, ONE protocol.

    smoke=True  : ~500 wells, 3 blocks, 15 epochs -> CPU < ~3 min.
    smoke=False : 8 blocks, 400 epochs. Two encoders x (spatial + random) = 4 CV passes
                  + the in-run XGB wall. A single HGT 8-fold spatial pass = ~17.5 min on
                  Colab GPU; hetero-SAGE is lighter. Budget below in run() return meta.
    """
    from . import data as D
    t0 = time.time()

    if smoke:
        n_blocks = n_blocks or SMOKE_BLOCKS
        max_epochs = max_epochs or SMOKE_EPOCHS
        patience = patience or SMOKE_PATIENCE
    else:
        n_blocks = n_blocks or FULL_BLOCKS
        max_epochs = max_epochs or FULL_EPOCHS
        patience = patience or FULL_PATIENCE

    if df is None:
        df = D.load(smoke=smoke, smoke_n=SMOKE_N_WELLS if smoke else None)
    if smoke and df[C.WELL_ID].nunique() > SMOKE_N_WELLS:
        rng = np.random.RandomState(seed)
        keep = set(rng.choice(df[C.WELL_ID].unique(), size=SMOKE_N_WELLS, replace=False))
        df = df[df[C.WELL_ID].isin(keep)].reset_index(drop=True)

    feature_cols = C.feature_columns(include_location=False, cocontam="core")

    common = dict(feature_cols=feature_cols, n_blocks=n_blocks, hidden=hidden, layers=layers,
                  dropout=dropout, heads=heads, k_spatial=k_spatial,
                  cap_km_spatial=cap_km_spatial, k_subbasin=k_subbasin,
                  cap_km_subbasin=cap_km_subbasin, max_epochs=max_epochs, patience=patience,
                  lr=lr, weight_decay=weight_decay, inductive=inductive, smoke=smoke,
                  seed=seed, verbose=verbose)

    exp_dir = Path(exp_dir) if exp_dir else (C.EXPERIMENTS_DIR / "v1_inductive_sage")
    exp_dir.mkdir(parents=True, exist_ok=True)

    out = {"meta": {
        "experiment": "v1_inductive_sage", "task": "T1a", "smoke": bool(smoke),
        "seed": int(seed), "n_features": len(feature_cols), "include_location": False,
        "feature_cols": list(feature_cols), "inductive": bool(inductive),
        "k_spatial": k_spatial, "cap_km_spatial": cap_km_spatial,
        "k_subbasin": k_subbasin, "cap_km_subbasin": cap_km_subbasin,
        "hidden": hidden, "layers": layers, "dropout": dropout, "heads": heads,
        "n_blocks": n_blocks, "relations": list(H.REL_NAMES), "encoders": list(ENCODERS),
        "v1_regularisation": {"drop_edge": V1_DROP_EDGE, "use_graphnorm": V1_USE_GRAPHNORM,
                              "neighbor_fanout": V1_NEIGHBOR_FANOUT},
        "wall_xgb_spatial_auc_committed": WALL_XGB_SPATIAL_AUC,
        "noise_threshold": NOISE_THRESHOLD,
    }}

    # ---- per-encoder spatial CV (reference) ----
    enc_results = {}
    wall_spatial = None
    for enc in ENCODERS:
        reg = run_encoder_regime(df, encoder=enc, regime="spatial", **common)
        mets, ci, thr, rel = _row_metrics(reg, reg["gnn_proba"])
        folds = _per_fold_aucs(reg, reg["gnn_proba"])
        overfit = _agg_overfit(reg["fold_diag"])
        if wall_spatial is None:
            wmets, wci, _, _ = _row_metrics(reg, reg["wall_proba"])
            wall_spatial = {"per_fold_auc": _per_fold_aucs(reg, reg["wall_proba"]),
                            "metrics": wmets, "auc_ci95": wci}
        enc_results[enc] = {
            "spatial": {"metrics": mets, "auc_ci95": ci, "per_fold_auc": folds,
                        "threshold": thr, "reliability_curve": rel,
                        "overfit_diag": overfit, "fold_diag": reg["fold_diag"],
                        "n_cross_block_total": reg["n_cross_block_total"]},
        }
        # incremental checkpoint
        if write:
            _write_metrics({**out, "encoders": enc_results,
                            "in_run_xgb_wall_spatial": wall_spatial}, exp_dir)

    out["in_run_xgb_wall_spatial"] = wall_spatial

    # ---- Δ(random − spatial) per encoder ----
    if compute_delta:
        for enc in ENCODERS:
            regr = run_encoder_regime(df, encoder=enc, regime="random", **common)
            rmets, rci, _, _ = _row_metrics(regr, regr["gnn_proba"])
            rfolds = _per_fold_aucs(regr, regr["gnn_proba"])
            enc_results[enc]["random"] = {
                "metrics": rmets, "auc_ci95": rci, "per_fold_auc": rfolds}
            enc_results[enc]["delta_random_minus_spatial"] = float(
                rmets["roc_auc"] - enc_results[enc]["spatial"]["metrics"]["roc_auc"])
            if write:
                _write_metrics({**out, "encoders": enc_results}, exp_dir)

    out["encoders"] = enc_results

    # ---- paired comparisons ----
    fold_row = S.spatial_block_folds(df, k=n_blocks, seed=seed)
    blocks = sorted(set(fold_row.tolist()))
    n_te = int(np.mean([(fold_row == b).sum() for b in blocks]))
    n_tr = int(np.mean([(fold_row != b).sum() for b in blocks]))

    sage_folds = enc_results["hetero_sage_v1"]["spatial"]["per_fold_auc"]
    hgt_folds = enc_results["hgt"]["spatial"]["per_fold_auc"]
    wall_folds = wall_spatial["per_fold_auc"]
    wall_mean = float(np.nanmean(wall_folds))

    def _verdict(arch_folds):
        gain = float(np.nanmean(arch_folds) - wall_mean)
        pt = _paired(arch_folds, wall_folds, n_tr, n_te)
        p_nb = pt["nadeau_bengio"].get("p", float("nan"))
        p_wc = pt["wilcoxon"].get("p", float("nan"))
        sig = (np.isfinite(p_nb) and p_nb < 0.05) or (np.isfinite(p_wc) and p_wc < 0.05)
        return {"gain_vs_in_run_wall": gain, "paired_vs_wall": pt,
                "significant": bool(sig), "magnitude_above_noise": bool(abs(gain) > NOISE_THRESHOLD),
                "verdict": ("robust_gain" if (sig and gain > NOISE_THRESHOLD) else "no_robust_gain")}

    out["comparison"] = {
        "n_tr_mean": n_tr, "n_te_mean": n_te, "wall_in_run_auc_mean": wall_mean,
        "wall_in_run_auc_oof_global": wall_spatial["metrics"]["roc_auc"],
        "hgt_vs_wall": _verdict(hgt_folds),
        "hetero_sage_v1_vs_wall": _verdict(sage_folds),
        "hetero_sage_v1_vs_hgt": {
            "per_fold_mean_diff": float(np.nanmean(sage_folds) - np.nanmean(hgt_folds)),
            "paired": _paired(sage_folds, hgt_folds, n_tr, n_te)},
        "overfit_gap_auc": {
            "hgt": enc_results["hgt"]["spatial"]["overfit_diag"]["gap_auc_mean"],
            "hetero_sage_v1": enc_results["hetero_sage_v1"]["spatial"]["overfit_diag"]["gap_auc_mean"],
            "reduction_sage_vs_hgt": float(
                enc_results["hgt"]["spatial"]["overfit_diag"]["gap_auc_mean"]
                - enc_results["hetero_sage_v1"]["spatial"]["overfit_diag"]["gap_auc_mean"]),
        },
    }
    # ---- under-training / convergence diagnostic (§3.8) ----
    out["comparison"]["convergence"] = {
        enc: _convergence_diag(enc_results[enc]["spatial"]["fold_diag"])
        for enc in ENCODERS}

    out["meta"]["elapsed_s"] = time.time() - t0

    if write:
        _write_metrics(out, exp_dir)
        _plot_training_curves(enc_results, exp_dir)   # writes training_curves_<enc>.png
        _write_report(out, exp_dir)
        _write_config(out, exp_dir)
    return out


# ======================================================= §3.8 convergence diagnostic
def _convergence_diag(fold_diag):
    """Detect under-training / premature early-stop from the per-epoch curves.

    For each fold we check whether validation AUC was still rising over the LAST few
    epochs before stop — if so, early stopping likely fired before the real plateau
    (the P0 under-training trap). Returns aggregate flags; no figure here.
    """
    n_ran, stopped, still_rising = [], 0, 0
    for d in fold_diag:
        va = [x for x in d.get("history_val_auc", []) if np.isfinite(x)]
        n_ran.append(d.get("n_epochs_ran", len(va)))
        if d.get("early_stopped"):
            stopped += 1
        # "still rising" = best_epoch within the final 20% of the run AND val_auc at the
        # end is within 0.005 of (or above) its running max => plateau not clearly reached.
        be = d.get("best_epoch", 0); nr = d.get("n_epochs_ran", len(va))
        if va and nr > 0 and be >= 0.8 * nr:
            still_rising += 1
    nfold = max(len(fold_diag), 1)
    return {
        "n_epochs_ran_mean": float(np.mean(n_ran)) if n_ran else 0.0,
        "n_epochs_ran_min": int(np.min(n_ran)) if n_ran else 0,
        "frac_folds_early_stopped": stopped / nfold,
        "frac_folds_best_in_last_20pct": still_rising / nfold,
        "under_training_flag": bool(still_rising / nfold >= 0.5),
        "note": ("best epoch in the last 20% of the run for >=50% of folds suggests "
                 "premature early-stop / under-training; raise max_epochs or patience"),
    }


def _plot_training_curves(enc_results, exp_dir):
    """Per-encoder PNG: per-fold val-AUC and fit-AUC vs epoch + train loss, best_epoch line."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # headless / missing -> skip gracefully, run still valid
        print(f"[plot] matplotlib unavailable ({e}); skipping training-curve figures")
        return
    for enc in enc_results:
        fds = enc_results[enc]["spatial"].get("fold_diag", [])
        if not fds:
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
        for d in fds:
            ep = d.get("history_epochs", [])
            if not ep:
                continue
            ax1.plot(ep, d.get("history_val_auc", []), lw=1, alpha=0.7,
                     label=f"fold {d.get('fold')}")
            ax1.plot(ep, d.get("history_fit_auc", []), lw=0.8, ls="--", alpha=0.4)
            be = d.get("best_epoch")
            if be is not None:
                ax1.axvline(be, color="grey", lw=0.5, alpha=0.3)
            ax2.plot(ep, d.get("history_train_loss", []), lw=1, alpha=0.7)
        ax1.set(xlabel="epoch", ylabel="AUC (solid=val, dashed=fit)",
                title=f"{enc} — val/fit AUC (line at best_epoch)")
        ax2.set(xlabel="epoch", ylabel="train loss", title=f"{enc} — train loss")
        ax1.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(Path(exp_dir) / f"training_curves_{enc}.png", dpi=110)
        plt.close(fig)
    print(f"[plot] wrote training_curves_*.png to {exp_dir}")


# ============================================================= writers
def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_metrics(out, exp_dir):
    (Path(exp_dir) / "metrics.json").write_text(json.dumps(out, indent=2,
                                                           default=_json_default))


def _write_config(out, exp_dir):
    m = out["meta"]; v = m["v1_regularisation"]
    lines = [
        "# v1_inductive_sage — config (seed 42)",
        f"task: {m['task']}", f"smoke: {m['smoke']}", f"seed: {m['seed']}",
        f"n_blocks: {m['n_blocks']}", f"n_features: {m['n_features']}",
        f"include_location: {m['include_location']}", f"inductive: {m['inductive']}",
        "encoders:", "  - hgt", "  - hetero_sage_v1",
        "relations:",
        f"  - near               # spatial k-NN, cap {m['cap_km_spatial']} km, k={m['k_spatial']}",
        f"  - same_subbasin_knn   # intra-sub-basin k-NN, cap {m['cap_km_subbasin']} km, k={m['k_subbasin']}",
        "hetero_sage_v1_regularisation:",
        f"  drop_edge: {v['drop_edge']}",
        f"  use_graphnorm: {v['use_graphnorm']}",
        f"  neighbor_fanout: {v['neighbor_fanout']}",
        "encoder_common:",
        f"  hidden: {m['hidden']}", f"  layers: {m['layers']}",
        f"  dropout: {m['dropout']}", f"  heads: {m['heads']}",
        f"noise_threshold: {m['noise_threshold']}",
        f"wall_xgb_spatial_auc_committed: {m['wall_xgb_spatial_auc_committed']}",
    ]
    (Path(exp_dir) / "config.yaml").write_text("\n".join(lines) + "\n")


def _enc_row(label, sp):
    g = sp["metrics"]; ci = sp["auc_ci95"]
    return (f"| {label} | {g['roc_auc']:.4f} | [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] | "
            f"{np.nanmean(sp['per_fold_auc']):.4f} | {g['f1']:.4f} | {g['pr_auc']:.4f} | "
            f"{g['balanced_accuracy']:.4f} | {g['brier']:.4f} | {g.get('ece', float('nan')):.4f} |")


def _write_report(out, exp_dir):
    m = out["meta"]; E = out["encoders"]; cmp = out["comparison"]
    w = out["in_run_xgb_wall_spatial"]
    hgt = E["hgt"]["spatial"]; sage = E["hetero_sage_v1"]["spatial"]
    lines = [
        "# V1 — Inductive heterogeneous GraphSAGE vs HGT (T1a, head-to-head)",
        "",
        f"> smoke={m['smoke']}  seed={m['seed']}  blocks={m['n_blocks']}  "
        f"features={m['n_features']} (strict: no PFAS measurement, no lat/lon).",
        "",
        "Both encoders run through the SAME eval-validated inductive fold trainer "
        "(`gnn_hetero_t1.train_eval_fold`) on the SAME multi-relational well-well graph "
        "(`near` k-NN cap 1.5 km, `same_subbasin_knn` intra-sub-basin k-NN cap 2 km). "
        "Only the encoder changes. hetero-SAGE-v1 = HeteroConv{relation: SAGEConv(mean)}, "
        f"INDUCTIVE, with DropEdge={m['v1_regularisation']['drop_edge']}, "
        f"GraphNorm={m['v1_regularisation']['use_graphnorm']}, neighbour-sampling fan-out="
        f"{m['v1_regularisation']['neighbor_fanout']}.",
        "",
        f"**In-run XGB-tabular wall (same 8 folds, same 61 features, frequency encoding):** "
        f"OOF-global {w['metrics']['roc_auc']:.4f} / per-fold-mean {cmp['wall_in_run_auc_mean']:.4f}. "
        "This is the ONLY apples-to-apples wall (the committed 0.588 scalar uses a different "
        "feature set + target encoding + per-fold-mean and is NOT comparable; "
        "hgt_fusion_stacking_t1/eval_validation.md Point 5).",
        "",
        "## Spatial-block results (row-level OOF)",
        "",
        "| encoder | AUC OOF | AUC 95% CI | AUC per-fold-mean | F1@OOF | PR-AUC | bal.acc | Brier | ECE |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        _enc_row("HGT (reference)", hgt),
        _enc_row("hetero-SAGE-v1 (inductive)", sage),
        _enc_row("XGB-tabular (in-run wall)", w),
        "",
        "## Overfitting diagnostic (fit vs val node AUC at best epoch, mean over 8 folds)",
        "",
        "| encoder | fit AUC | val AUC | fit−val gap | fit−val F1 gap | best epoch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, sp in [("HGT", hgt), ("hetero-SAGE-v1", sage)]:
        o = sp["overfit_diag"]
        lines.append(
            f"| {label} | {o['fit_auc_mean']:.4f} | {o['val_auc_mean']:.4f} | "
            f"{o['gap_auc_mean']:+.4f} | {o['gap_f1_mean']:+.4f} | {o['best_epoch_mean']:.1f} |")
    og = cmp["overfit_gap_auc"]
    lines += [
        "",
        f"**Overfitting gap reduction (HGT gap − SAGE gap) = {og['reduction_sage_vs_hgt']:+.4f} AUC.** "
        "A positive value means the regularised inductive SAGE generalises with a smaller "
        "fit-to-val gap than HGT (the V1 motivation).",
        "",
        "## Convergence / under-training diagnostic (§3.8)",
        "",
        "Per-epoch train-loss and val/fit-AUC curves are saved as `training_curves_<encoder>.png` "
        "(grey line = best_epoch). The table flags premature early-stop (P0 under-training trap): "
        "if `best_epoch` falls in the last 20% of the run for most folds, raise max_epochs/patience.",
        "",
        "| encoder | epochs ran (mean / min) | folds early-stopped | folds best-in-last-20% | under-training |",
        "|---|---:|---:|---:|:--:|",
    ]
    conv = cmp.get("convergence", {})
    for label, key in [("HGT", "hgt"), ("hetero-SAGE-v1", "hetero_sage_v1")]:
        c = conv.get(key, {})
        lines.append(
            f"| {label} | {c.get('n_epochs_ran_mean', float('nan')):.1f} / {c.get('n_epochs_ran_min', 0)} | "
            f"{c.get('frac_folds_early_stopped', float('nan')):.0%} | "
            f"{c.get('frac_folds_best_in_last_20pct', float('nan')):.0%} | "
            f"{'⚠️ YES' if c.get('under_training_flag') else 'no'} |")
    lines += [
        "",
        "## Δ(random − spatial) — spatial-leakage inflation (C-SPAT.6)",
        "",
        "| encoder | spatial AUC | random AUC | Δ |",
        "|---|---:|---:|---:|",
    ]
    for label, key in [("HGT", "hgt"), ("hetero-SAGE-v1", "hetero_sage_v1")]:
        sp = E[key]["spatial"]["metrics"]["roc_auc"]
        rnd = E[key].get("random", {}).get("metrics", {}).get("roc_auc", float("nan"))
        d = E[key].get("delta_random_minus_spatial", float("nan"))
        lines.append(f"| {label} | {sp:.4f} | {rnd:.4f} | {d:+.4f} |")
    lines += [
        "",
        "## Paired tests vs the in-run wall and head-to-head (8 spatial folds)",
        "",
        "| comparison | per-fold mean diff | NB p | Wilcoxon p | verdict |",
        "|---|---:|---:|---:|---|",
    ]
    hw = cmp["hgt_vs_wall"]; sw = cmp["hetero_sage_v1_vs_wall"]; sh = cmp["hetero_sage_v1_vs_hgt"]
    lines += [
        f"| HGT − wall | {hw['gain_vs_in_run_wall']:+.4f} | "
        f"{hw['paired_vs_wall']['nadeau_bengio'].get('p', float('nan')):.4f} | "
        f"{hw['paired_vs_wall']['wilcoxon'].get('p', float('nan')):.4f} | {hw['verdict']} |",
        f"| hetero-SAGE-v1 − wall | {sw['gain_vs_in_run_wall']:+.4f} | "
        f"{sw['paired_vs_wall']['nadeau_bengio'].get('p', float('nan')):.4f} | "
        f"{sw['paired_vs_wall']['wilcoxon'].get('p', float('nan')):.4f} | {sw['verdict']} |",
        f"| hetero-SAGE-v1 − HGT | {sh['per_fold_mean_diff']:+.4f} | "
        f"{sh['paired']['nadeau_bengio'].get('p', float('nan')):.4f} | "
        f"{sh['paired']['wilcoxon'].get('p', float('nan')):.4f} | "
        f"{'differs' if (np.isfinite(sh['paired']['nadeau_bengio'].get('p', np.nan)) and sh['paired']['nadeau_bengio']['p'] < 0.05) else 'no_robust_diff'} |",
        "",
        "**Reality rule (eval C-CMP):** a gain over the wall is robust only if paired-",
        "significant (p<0.05) AND above the inter-fold noise bar (0.03 AUC). Cross-block edges "
        f"remaining (must be 0): HGT={hgt['n_cross_block_total']}, "
        f"hetero-SAGE-v1={sage['n_cross_block_total']}.",
    ]
    (Path(exp_dir) / "REPORT.md").write_text("\n".join(lines) + "\n")
