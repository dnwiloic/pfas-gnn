"""Three-architecture hybrid on T1a — HGT encoder + embedding fusion + stacking ensemble.

This is the SINGLE experiment mandated for `experiments/hgt_fusion_stacking_t1/`. It
reproduces the reference paper's three-architecture design ADAPTED to the node/edge types
that are legitimate for THIS dataset (eval-validated, experiments/hgt_rgcn_t1/
eval_validation.md): a MULTI-RELATIONAL WELL-WELL graph with two real edge types
(`near` spatial k-NN cap 1.5 km, `same_subbasin_knn` intra-sub-basin k-NN cap 2 km). HGT
is used PURELY as a relational encoder over these typed EDGES. No fabricated source/
compound node type is created (C-NODE.1/2 — REJECTED by the prior audit).

The three architectures, under ONE protocol
--------------------------------------------
1. HGT STANDALONE   — multi-relational encoder over the well-well graph -> per-well
                      embedding + a classification head. Reuses gnn_hetero_t1 verbatim.
2. EMBEDDING FUSION — XGBoost on [tabular 96-col strict features (NO PFAS measurement)
                      (+) PCA-reduced HGT embeddings to 95% variance]. We report how many
                      components 95% variance is HERE.
3. STACKING         — meta-XGBoost over base out-of-fold predictions
                      {HGT, XGBoost-tabular, LightGBM-tabular} + agreement / entropy
                      meta-features. Meta-features fit on OOF TRAIN predictions only.

Why a SHARED OOF backbone (and how it stays leak-free)
------------------------------------------------------
Architectures 2 and 3 consume HGT outputs and base-model outputs. To make them leak-free
under the spatial-block protocol we run ONE leave-one-block-out (LOBO) over the 8 KMeans
spatial blocks. For each held-out block b:

  * HGT is trained on the train blocks, then scored on block b -> the test-block wells get
    an OOF HGT probability AND an OOF HGT embedding (the message passing for a block-b well
    uses ONLY its TRAIN neighbours, since cross-block edges are cut per relation, C-SPAT.2/5,
    asserted 0).
  * XGBoost-tabular and LightGBM-tabular are trained on the train-block wells (tabular
    pipeline FIT on train wells only) and scored on block b -> OOF base probabilities.

After the loop every well carries OOF arrays {hgt_proba, hgt_emb[H], xgb_proba, lgbm_proba}
none of which ever saw that well's block at training time. The fusion / stacking
META-learners are then evaluated with a SECOND, nested LOBO over the SAME 8 blocks on these
OOF arrays: for held-out block b the meta-learner (and its PCA, for fusion) is FIT on the
OOF rows of the OTHER 7 blocks and predicts block b. So nothing test-block ever touches a
fit step — neither the HGT, nor the bases, nor the PCA, nor the meta-learner. This mirrors
the paper's PCA-to-95%-variance + stacking design while honouring the project's anti-leak law.

Everything is reported at the SAMPLING (row) level (well proba broadcast to each sampling),
exactly like the non-graph WALL (XGB spatial AUC 0.588, RF 0.601 in
experiments/baseline_t1/metrics_spatial.json), so the comparison is apples-to-apples.

Protocol guarantees (eval_validation.md conditions)
---------------------------------------------------
  C-SPAT.1  spatial-block CV at the WELL level (splits.spatial_block_folds, k=8) reported
            BESIDE grouped-random CV (the Δ arm).
  C-SPAT.2/5 cross-block edges cut per relation, asserted 0 (inherited from gnn_hetero_t1).
  C-SPAT.4  inductive HGT (test well aggregates only from TRAIN neighbours).
  C-LOC.1   lat/lon never a node/tabular feature.
  C-THR     every F1 threshold from OOF/VAL probabilities only, never from test.
  C-CAL     Brier + ECE reported for each architecture.
  C-CMP     paired Nadeau-Bengio + Wilcoxon on the 8 spatial folds vs the WALL, bootstrap CI.
  Seed fixed (config.SEED=42) everywhere.

Torch / xgboost / lightgbm are imported lazily so the module imports on a bare box; it is
CPU smoke-testable (`run(smoke=True)`, ~a few hundred wells, 3 blocks, few epochs, < ~3 min).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import gnn_hetero_t1 as H
from . import graph as G
from . import metrics as M
from . import splits as S
from . import targets as T

# reuse the FROZEN paired-test / CI helpers from the hybrid socle (do not re-derive stats)
from .hybrid import (
    _bootstrap_ci_by_well,
    _corrected_resampled_ttest,
    _wilcoxon_paired,
    _ece,
    _optimal_threshold_f1,
    _make_xgb,
)

SEED = C.SEED

# WALL (experiments/baseline_t1/metrics_spatial.json, full run smoke=false, k=8)
WALL_XGB_SPATIAL_AUC = 0.5877739078600925
WALL_RF_SPATIAL_AUC = 0.6009263696712559
NOISE_THRESHOLD = 0.03   # inter-fold reality bar (eval C-CMP)


# ----------------------------------------------------------------- smoke / full params
SMOKE_N_WELLS = 500
SMOKE_BLOCKS = 3
SMOKE_EPOCHS = 15
SMOKE_PATIENCE = 6

FULL_BLOCKS = C.N_SPATIAL_BLOCKS     # 8
FULL_EPOCHS = 400
FULL_PATIENCE = 50


# ============================================================= LightGBM factory
def _make_lgbm(smoke: bool = False, seed: int = SEED):
    """LightGBM tabular base; falls back to sklearn HistGradientBoosting if lgbm absent.

    A SECOND, architecturally-different tabular base (vs XGBoost) is what makes stacking
    meaningful — the meta-learner exploits disagreement between two boosters and the GNN.
    """
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=50 if smoke else 300,
            num_leaves=31, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
            n_jobs=-1, verbose=-1,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=50 if smoke else 300, random_state=seed, class_weight="balanced")


# ============================================================= tabular features (well level)
def _tabular_well_matrix(df, well_ids, feature_cols, train_mask):
    """Per-well tabular matrix, FeaturePipeline FIT ON TRAIN WELLS ONLY (anti-leak).

    `train_mask` is a per-well boolean (well order). Returns X[n_wells, d], names.
    Frequency encoding -> no y needed -> trivially leak-free.
    """
    from . import features as F
    wf = G.aggregate_to_wells(df, well_ids, feature_cols)
    pipe = F.FeaturePipeline(feature_cols, encode="frequency")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit_transform(wf.iloc[train_mask], None)
        X, names = pipe.transform(wf)
    return X.astype(np.float32), list(names)


# ============================================================= OOF backbone (one LOBO)
@dataclass
class OOFArrays:
    """Per-well OOF arrays after the shared leave-one-block-out backbone.

    Every entry was produced by a model that NEVER saw that well's spatial block. `*_proba`
    are well-level OOF probabilities; `hgt_emb` is the well-level OOF HGT embedding.
    """
    well_ids: np.ndarray
    node_block: np.ndarray                 # per-well spatial block id
    y_well: np.ndarray                     # well-level (majority) T1a
    hgt_proba: np.ndarray
    hgt_emb: np.ndarray                    # [n_wells, H]
    xgb_proba: np.ndarray
    lgbm_proba: np.ndarray
    tabular: np.ndarray                    # [n_wells, d] (per-well, freq-encoded; OK to reuse: fit per outer-fold below)
    row_to_node: np.ndarray
    audit: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def build_oof_backbone(df, *, feature_cols, n_blocks, regime, hidden, layers, dropout,
                       heads, k_spatial, cap_km_spatial, k_subbasin, cap_km_subbasin,
                       max_epochs, patience, lr, weight_decay, inductive, smoke,
                       seed=SEED, verbose=False):
    """Run ONE leave-one-block-out over `regime` blocks producing per-well OOF arrays for
    all three architectures. Returns an `OOFArrays`.

    `regime`: 'spatial' (KMeans blocks, reference) or 'random' (grouped-random, Δ arm).
    """
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
    hgt_proba = np.full(n, np.nan)
    hgt_emb = np.full((n, hidden), np.nan, dtype=np.float32)
    xgb_proba = np.full(n, np.nan)
    lgbm_proba = np.full(n, np.nan)

    model_kw = dict(hidden=hidden, layers=layers, dropout=dropout, heads=heads)
    train_kw = dict(k_spatial=k_spatial, cap_km_spatial=cap_km_spatial,
                    k_subbasin=k_subbasin, cap_km_subbasin=cap_km_subbasin,
                    lr=lr, weight_decay=weight_decay, max_epochs=max_epochs,
                    patience=patience, inductive=inductive)

    blocks = sorted(set(node_block.tolist()))
    total_cross = 0
    for b in blocks:
        test_nodes = node_block == b
        train_mask = ~test_nodes
        # --- HGT base (reuses the eval-validated multi-relational fold trainer) ---
        fr, proba_node, emb_node = H.train_eval_fold(
            df, well_ids, y_well, node_block, b, feature_cols,
            name="hgt", coords=coords, subbasin=subbasin, y_row=y_row,
            seed=seed, verbose=False, **model_kw, **train_kw)
        hgt_proba[test_nodes] = proba_node[test_nodes]
        hgt_emb[test_nodes] = emb_node[test_nodes]
        total_cross += int(fr.audit["n_cross_block_near"] + fr.audit["n_cross_block_subbasin"])

        # --- tabular bases (XGB + LightGBM), pipeline fit on TRAIN wells only ---
        X_tab, _ = _tabular_well_matrix(df, well_ids, feature_cols, train_mask)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xgb = _make_xgb(smoke=smoke, prevalence=float(y_well[train_mask].mean()))
            xgb.fit(X_tab[train_mask], y_well[train_mask])
            xgb_proba[test_nodes] = xgb.predict_proba(X_tab[test_nodes])[:, 1]
            lgbm = _make_lgbm(smoke=smoke, seed=seed)
            lgbm.fit(X_tab[train_mask], y_well[train_mask])
            lp = lgbm.predict_proba(X_tab[test_nodes])
            lgbm_proba[test_nodes] = lp[:, 1] if lp.ndim == 2 else lp

        if verbose:
            a = fr.audit
            print(f"[{regime}] block {b}: HGT_AUC(fold)={fr.metrics_spatial['roc_auc']:.4f} "
                  f"xblock(near={a['n_cross_block_near']},sub={a['n_cross_block_subbasin']})")

    # tabular for the fusion arm: a full-data freq-encoding (PCA + final XGB are still re-fit
    # per OUTER fold on train OOF rows, so this matrix is only a column schema, not a fit-leak).
    X_tab_full, _ = _tabular_well_matrix(df, well_ids, feature_cols,
                                         np.ones(n, dtype=bool))

    return OOFArrays(
        well_ids=well_ids, node_block=node_block, y_well=y_well,
        hgt_proba=hgt_proba, hgt_emb=hgt_emb, xgb_proba=xgb_proba,
        lgbm_proba=lgbm_proba, tabular=X_tab_full, row_to_node=row_to_node,
        audit={"n_cross_block_total": int(total_cross)},
        meta={"n_wells": n, "hidden": hidden, "regime": regime,
              "n_blocks": len(blocks), "n_tabular_features": int(X_tab_full.shape[1])},
    )


# ============================================================= V2 accessor (non-breaking)
def oof_embeddings_and_tabular(oof):
    """Reusable accessor for V2 fusion variants (does NOT alter the V0 backbone).

    Returns a dict aligning, BY WELL ROW, the leak-free OOF arrays a downstream fusion
    head / XGBoost needs:

        well_ids   : np.ndarray[str]   shape (n_wells,)   well identity, row order
        node_block : np.ndarray[int]   shape (n_wells,)   spatial block id (k=8) per well
        y_well     : np.ndarray[int]   shape (n_wells,)   well-level majority T1a label
        hgt_emb    : np.ndarray[f32]   shape (n_wells, H) leak-free OOF HGT embedding
        tabular    : np.ndarray[f32]   shape (n_wells, d) per-well freq-encoded features
        row_to_node: np.ndarray[int]   maps each sampling ROW -> well index (broadcast)
        valid_emb  : np.ndarray[bool]  rows whose hgt_emb is finite (every well in a full run)

    LEAK GUARANTEE (inherited from build_oof_backbone): row i of `hgt_emb` was produced by
    an HGT trained on the OTHER 7 spatial blocks and scored on well i's block via the
    cross-block-free edge set, so the embedding of a held-out-block well aggregates ONLY
    from its TRAIN neighbours (C-SPAT.4). The fusion head must STILL be fit nested-LOBO on
    these rows (never fit on the held-out block) — see v2_fusion_gating.train_gating_oof.

    Row order is identical across all returned arrays: index i is always well_ids[i].
    """
    valid_emb = ~np.isnan(oof.hgt_emb).any(axis=1)
    return {
        "well_ids": oof.well_ids,
        "node_block": oof.node_block,
        "y_well": oof.y_well.astype(int),
        "hgt_emb": oof.hgt_emb,
        "tabular": oof.tabular,
        "row_to_node": oof.row_to_node,
        "valid_emb": valid_emb,
        "hidden": int(oof.meta["hidden"]),
        "n_tabular_features": int(oof.meta["n_tabular_features"]),
    }


# ============================================================= row-level scoring helper
def _row_metrics(oof, proba_well, y_row, df, *, thr_source_well=None):
    """Broadcast a per-well OOF probability to sampling rows, threshold from OOF wells,
    return the full row-level metric set (comparable to the WALL)."""
    valid = ~np.isnan(proba_well)
    # OOF F1 threshold from the WELL-level OOF probas (C-THR)
    if thr_source_well is None:
        thr = _optimal_threshold_f1(oof.y_well[valid].astype(int), proba_well[valid])
    else:
        thr = thr_source_well
    proba_row = proba_well[oof.row_to_node]
    rmask = valid[oof.row_to_node]
    yt = np.asarray(y_row)[rmask].astype(int)
    pt = proba_row[rmask]
    mets = M.binary_metrics(yt, pt, threshold=thr)
    mets["ece"] = _ece(yt, pt)
    wells_row = df[C.WELL_ID].to_numpy()[rmask]
    ci = _bootstrap_ci_by_well(yt, pt, wells_row, seed=SEED)
    return mets, ci, thr, yt, pt, wells_row


def _per_fold_aucs(oof, proba_well, y_row, df):
    """Per-spatial-block row-level AUC list (for paired tests). One AUC per held-out block."""
    from sklearn.metrics import roc_auc_score
    proba_row = proba_well[oof.row_to_node]
    block_row = oof.node_block[oof.row_to_node]
    aucs = []
    for b in sorted(set(oof.node_block.tolist())):
        m = (block_row == b) & ~np.isnan(proba_row)
        yt = np.asarray(y_row)[m].astype(int)
        if len(np.unique(yt)) < 2:
            aucs.append(float("nan")); continue
        aucs.append(float(roc_auc_score(yt, proba_row[m])))
    return aucs


# ============================================================= ARCH 2: embedding fusion
def fusion_oof_proba(oof, *, pca_var=0.95, smoke=False, seed=SEED):
    """Nested-LOBO fusion: for each held-out block, FIT PCA(95% var) + XGB on the OTHER
    blocks' OOF rows ([tabular (+) PCA(HGT emb)]) and predict the held-out block.

    Returns (fusion_proba_well[n], n_components_used_list, mean_n_components).
    """
    from sklearn.decomposition import PCA
    import warnings

    n = len(oof.well_ids)
    fusion = np.full(n, np.nan)
    blocks = sorted(set(oof.node_block.tolist()))
    n_comp_used = []
    valid_emb = ~np.isnan(oof.hgt_emb).any(axis=1)
    for b in blocks:
        tr = (oof.node_block != b) & valid_emb
        te = (oof.node_block == b) & valid_emb
        if tr.sum() < 10 or te.sum() < 1 or len(np.unique(oof.y_well[tr])) < 2:
            continue
        emb_tr = oof.hgt_emb[tr]
        # PCA fit on TRAIN OOF embeddings only; keep enough comps for `pca_var` variance
        max_c = min(emb_tr.shape[0] - 1, emb_tr.shape[1])
        pca = PCA(n_components=max_c, random_state=seed).fit(emb_tr)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cum, pca_var) + 1)
        k = max(1, min(k, max_c))
        n_comp_used.append(k)
        emb_tr_p = pca.transform(emb_tr)[:, :k]
        emb_te_p = pca.transform(oof.hgt_emb[te])[:, :k]
        X_tr = np.hstack([oof.tabular[tr], emb_tr_p]).astype(np.float32)
        X_te = np.hstack([oof.tabular[te], emb_te_p]).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = _make_xgb(smoke=smoke, prevalence=float(oof.y_well[tr].mean()))
            clf.fit(X_tr, oof.y_well[tr])
            fusion[te] = clf.predict_proba(X_te)[:, 1]
    mean_k = float(np.mean(n_comp_used)) if n_comp_used else float("nan")
    return fusion, n_comp_used, mean_k


# ============================================================= ARCH 3: stacking
def _entropy(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def stacking_oof_proba(oof, *, smoke=False, seed=SEED):
    """Nested-LOBO stacking: meta-XGB over base OOF probas {HGT, XGB-tab, LGBM-tab} plus
    agreement / entropy meta-features. Meta fit on TRAIN OOF rows only (never test).

    Returns (stack_proba_well[n], meta_feature_names).
    """
    import warnings
    n = len(oof.well_ids)
    stack = np.full(n, np.nan)
    base = np.vstack([oof.hgt_proba, oof.xgb_proba, oof.lgbm_proba]).T   # [n,3]
    # meta-features: mean, std (disagreement), pairwise abs-diff agreement, mean entropy
    mean_p = np.nanmean(base, axis=1)
    std_p = np.nanstd(base, axis=1)
    agree_hx = np.abs(oof.hgt_proba - oof.xgb_proba)
    agree_hl = np.abs(oof.hgt_proba - oof.lgbm_proba)
    agree_xl = np.abs(oof.xgb_proba - oof.lgbm_proba)
    ent = np.nanmean(np.vstack([_entropy(oof.hgt_proba), _entropy(oof.xgb_proba),
                                _entropy(oof.lgbm_proba)]).T, axis=1)
    feats = np.column_stack([oof.hgt_proba, oof.xgb_proba, oof.lgbm_proba,
                             mean_p, std_p, agree_hx, agree_hl, agree_xl, ent])
    names = ["hgt_p", "xgb_p", "lgbm_p", "mean_p", "std_p",
             "agree_hgt_xgb", "agree_hgt_lgbm", "agree_xgb_lgbm", "mean_entropy"]
    valid = ~np.isnan(feats).any(axis=1)
    blocks = sorted(set(oof.node_block.tolist()))
    for b in blocks:
        tr = (oof.node_block != b) & valid
        te = (oof.node_block == b) & valid
        if tr.sum() < 10 or te.sum() < 1 or len(np.unique(oof.y_well[tr])) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meta = _make_xgb(smoke=smoke, prevalence=float(oof.y_well[tr].mean()),
                             max_depth=3, n_estimators=50 if smoke else 200)
            meta.fit(feats[tr], oof.y_well[tr])
            stack[te] = meta.predict_proba(feats[te])[:, 1]
    return stack, names


# ============================================================= one regime (3 archs)
def run_regime(df, *, regime, feature_cols, n_blocks, hidden, layers, dropout, heads,
               k_spatial, cap_km_spatial, k_subbasin, cap_km_subbasin, max_epochs,
               patience, lr, weight_decay, inductive, pca_var, smoke, seed=SEED,
               verbose=False):
    """Build the OOF backbone for `regime` then evaluate all THREE architectures at row
    level. Returns a dict with per-architecture metrics, CIs, per-fold AUC lists."""
    y_row = T.build_T1a(df).to_numpy()
    oof = build_oof_backbone(
        df, feature_cols=feature_cols, n_blocks=n_blocks, regime=regime, hidden=hidden,
        layers=layers, dropout=dropout, heads=heads, k_spatial=k_spatial,
        cap_km_spatial=cap_km_spatial, k_subbasin=k_subbasin, cap_km_subbasin=cap_km_subbasin,
        max_epochs=max_epochs, patience=patience, lr=lr, weight_decay=weight_decay,
        inductive=inductive, smoke=smoke, seed=seed, verbose=verbose)

    # ARCH 1 — HGT standalone
    hgt_mets, hgt_ci, hgt_thr, *_ = _row_metrics(oof, oof.hgt_proba, y_row, df)
    hgt_folds = _per_fold_aucs(oof, oof.hgt_proba, y_row, df)

    # base tabular references (for context / stacking inputs)
    xgb_mets, xgb_ci, *_ = _row_metrics(oof, oof.xgb_proba, y_row, df)
    xgb_folds = _per_fold_aucs(oof, oof.xgb_proba, y_row, df)
    lgbm_mets, lgbm_ci, *_ = _row_metrics(oof, oof.lgbm_proba, y_row, df)

    # ARCH 2 — embedding fusion
    fus_proba, n_comp_list, mean_k = fusion_oof_proba(oof, pca_var=pca_var, smoke=smoke,
                                                      seed=seed)
    fus_mets, fus_ci, fus_thr, *_ = _row_metrics(oof, fus_proba, y_row, df)
    fus_folds = _per_fold_aucs(oof, fus_proba, y_row, df)

    # ARCH 3 — stacking
    stk_proba, meta_names = stacking_oof_proba(oof, smoke=smoke, seed=seed)
    stk_mets, stk_ci, stk_thr, *_ = _row_metrics(oof, stk_proba, y_row, df)
    stk_folds = _per_fold_aucs(oof, stk_proba, y_row, df)

    return {
        "regime": regime,
        "n_cross_block_total": oof.audit["n_cross_block_total"],
        "n_wells": oof.meta["n_wells"],
        "n_blocks": oof.meta["n_blocks"],
        "n_tabular_features": oof.meta["n_tabular_features"],
        "hidden": hidden,
        "architectures": {
            "hgt_standalone": {"metrics": hgt_mets, "auc_ci95": hgt_ci,
                               "per_fold_auc": hgt_folds},
            "embedding_fusion": {"metrics": fus_mets, "auc_ci95": fus_ci,
                                 "per_fold_auc": fus_folds,
                                 "pca_n_components_per_fold": n_comp_list,
                                 "pca_n_components_mean": mean_k,
                                 "pca_variance_target": pca_var},
            "stacking": {"metrics": stk_mets, "auc_ci95": stk_ci,
                         "per_fold_auc": stk_folds, "meta_features": meta_names},
        },
        "base_references": {
            "xgb_tabular": {"metrics": xgb_mets, "auc_ci95": xgb_ci,
                            "per_fold_auc": xgb_folds},
            "lgbm_tabular": {"metrics": lgbm_mets, "auc_ci95": lgbm_ci},
        },
    }


# ============================================================= paired tests vs WALL
def _paired_vs_wall(arch_folds, n_tr_mean, n_te_mean):
    """Paired Nadeau-Bengio + Wilcoxon of one architecture's 8 spatial-fold AUCs vs the
    XGB-tabular WALL folds. The WALL arm here is the XGB-tabular base computed ON THE SAME
    8 folds in THIS run (true paired ablation, not the committed scalar)."""
    a = np.asarray(arch_folds, float)
    return a


def build_comparison(spatial_res, *, df, n_blocks):
    """Paired tests + reality verdict for each architecture vs the in-run XGB-tabular wall
    (same 8 folds) and vs the committed wall scalar (0.588)."""
    archs = spatial_res["architectures"]
    wall_folds = spatial_res["base_references"]["xgb_tabular"]["per_fold_auc"]
    wall_mean = spatial_res["base_references"]["xgb_tabular"]["metrics"]["roc_auc"]

    # rough train/test sizes for the Nadeau-Bengio correction (row counts per block)
    fold_row = S.spatial_block_folds(df, k=n_blocks, seed=SEED)
    blocks = sorted(set(fold_row.tolist()))
    n_te_mean = int(np.mean([(fold_row == b).sum() for b in blocks]))
    n_tr_mean = int(np.mean([(fold_row != b).sum() for b in blocks]))

    out = {"in_run_xgb_wall_auc_mean": float(wall_mean),
           "committed_xgb_wall_auc": WALL_XGB_SPATIAL_AUC,
           "committed_rf_wall_auc": WALL_RF_SPATIAL_AUC,
           "noise_threshold": NOISE_THRESHOLD,
           "n_tr_mean": n_tr_mean, "n_te_mean": n_te_mean, "by_architecture": {}}

    for name, a in archs.items():
        af = [x for x in a["per_fold_auc"] if np.isfinite(x)]
        wf = [x for x in wall_folds if np.isfinite(x)]
        k = min(len(af), len(wf))
        rec = {"auc_mean": float(np.nanmean(a["per_fold_auc"])),
               "auc_oof_global": a["metrics"]["roc_auc"],
               "gain_vs_in_run_wall": float(np.nanmean(a["per_fold_auc"]) - wall_mean),
               "gain_vs_committed_wall": float(a["metrics"]["roc_auc"] - WALL_XGB_SPATIAL_AUC)}
        if k >= 2:
            nb = _corrected_resampled_ttest(af[:k], wf[:k], n_tr_mean, n_te_mean)
            wc = _wilcoxon_paired(af[:k], wf[:k])
            rec["nadeau_bengio"] = nb
            rec["wilcoxon"] = wc
            p_nb = nb.get("p", float("nan"))
            p_wc = wc.get("p", float("nan"))
            sig = (np.isfinite(p_nb) and p_nb < 0.05) or (np.isfinite(p_wc) and p_wc < 0.05)
            above = abs(rec["gain_vs_in_run_wall"]) > NOISE_THRESHOLD
            rec["significant"] = bool(sig)
            rec["above_noise"] = bool(above)
            rec["verdict"] = ("robust_gain" if (sig and above and rec["gain_vs_in_run_wall"] > 0)
                              else "no_robust_gain")
        out["by_architecture"][name] = rec
    return out


# ============================================================= main entry point
def run(df=None, *, smoke=False, n_blocks=None, hidden=64, layers=2, dropout=0.3, heads=4,
        k_spatial=8, cap_km_spatial=1.5, k_subbasin=8, cap_km_subbasin=2.0,
        max_epochs=None, patience=None, lr=5e-3, weight_decay=5e-4, inductive=True,
        pca_var=0.95, compute_delta=True, write=True, exp_dir=None, seed=SEED,
        verbose=False):
    """End-to-end three-architecture T1a run.

    smoke=True : ~500 wells, 3 blocks, 15 epochs, small hidden -> CPU < ~3 min.
    smoke=False: COLAB / local CPU full run (8 blocks, 400 epochs). CPU-feasible
                 (gnn_hybrid_t1 ran ~68 min locally); this is heavier (3 bases x 8 folds
                 + nested meta-CV) -> budget ~2-4 h on CPU. Checkpoints per regime.

    Returns the full results dict; writes metrics.json + REPORT.md + config.yaml when write.
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

    common = dict(feature_cols=feature_cols, n_blocks=n_blocks, hidden=hidden,
                  layers=layers, dropout=dropout, heads=heads, k_spatial=k_spatial,
                  cap_km_spatial=cap_km_spatial, k_subbasin=k_subbasin,
                  cap_km_subbasin=cap_km_subbasin, max_epochs=max_epochs,
                  patience=patience, lr=lr, weight_decay=weight_decay,
                  inductive=inductive, pca_var=pca_var, smoke=smoke, seed=seed,
                  verbose=verbose)

    exp_dir = Path(exp_dir) if exp_dir else (C.EXPERIMENTS_DIR / "hgt_fusion_stacking_t1")
    exp_dir.mkdir(parents=True, exist_ok=True)

    out = {"meta": {
        "task": "T1a", "smoke": bool(smoke), "seed": int(seed),
        "n_features": len(feature_cols), "include_location": False,
        "feature_cols": list(feature_cols), "inductive": bool(inductive),
        "k_spatial": k_spatial, "cap_km_spatial": cap_km_spatial,
        "k_subbasin": k_subbasin, "cap_km_subbasin": cap_km_subbasin,
        "hidden": hidden, "layers": layers, "dropout": dropout, "heads": heads,
        "pca_variance_target": pca_var, "n_blocks": n_blocks,
        "relations": list(H.REL_NAMES),
        "wall_xgb_spatial_auc": WALL_XGB_SPATIAL_AUC,
        "wall_rf_spatial_auc": WALL_RF_SPATIAL_AUC,
    }}

    spatial = run_regime(df, regime="spatial", **common)
    out["spatial"] = spatial
    # incremental checkpoint after the (expensive) spatial arm
    _write_metrics(out, exp_dir)

    if compute_delta:
        rnd = run_regime(df, regime="random", **common)
        out["random"] = rnd
        for arch in spatial["architectures"]:
            d = (rnd["architectures"][arch]["metrics"]["roc_auc"]
                 - spatial["architectures"][arch]["metrics"]["roc_auc"])
            out.setdefault("delta_random_minus_spatial", {})[arch] = float(d)

    out["comparison"] = build_comparison(spatial, df=df, n_blocks=n_blocks)
    out["meta"]["elapsed_s"] = time.time() - t0

    if write:
        _write_metrics(out, exp_dir)
        _write_report(out, exp_dir)
        _write_config(out, exp_dir)
    return out


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
    m = out["meta"]
    lines = [
        "# hgt_fusion_stacking_t1 — config (seed 42)",
        f"task: {m['task']}",
        f"smoke: {m['smoke']}",
        f"seed: {m['seed']}",
        f"n_blocks: {m['n_blocks']}",
        f"n_features: {m['n_features']}",
        f"include_location: {m['include_location']}",
        f"inductive: {m['inductive']}",
        "relations:",
        f"  - near        # spatial k-NN, cap {m['cap_km_spatial']} km, k={m['k_spatial']}",
        f"  - same_subbasin_knn  # intra-sub-basin k-NN, cap {m['cap_km_subbasin']} km, k={m['k_subbasin']}",
        "hgt:",
        f"  hidden: {m['hidden']}",
        f"  layers: {m['layers']}",
        f"  dropout: {m['dropout']}",
        f"  heads: {m['heads']}",
        f"pca_variance_target: {m['pca_variance_target']}",
        f"wall_xgb_spatial_auc: {m['wall_xgb_spatial_auc']}",
        f"wall_rf_spatial_auc: {m['wall_rf_spatial_auc']}",
    ]
    (Path(exp_dir) / "config.yaml").write_text("\n".join(lines) + "\n")


def _arch_row(name, a):
    g = a["metrics"]; ci = a["auc_ci95"]
    return (f"| {name} | {g['roc_auc']:.4f} | "
            f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] | {g['f1']:.4f} | "
            f"{g['pr_auc']:.4f} | {g['balanced_accuracy']:.4f} | {g['brier']:.4f} | "
            f"{g.get('ece', float('nan')):.4f} |")


def _write_report(out, exp_dir):
    m = out["meta"]; sp = out["spatial"]; A = sp["architectures"]
    cmp = out.get("comparison", {})
    lines = [
        "# HGT encoder + embedding fusion + stacking ensemble — T1a (ONE experiment)",
        "",
        f"> smoke={m['smoke']}  seed={m['seed']}  blocks={m['n_blocks']}  "
        f"features={m['n_features']} (strict, no PFAS measurement, no lat/lon).",
        "",
        "Three architectures under ONE spatial-block protocol on a MULTI-RELATIONAL "
        "well-well graph (`near` spatial k-NN cap 1.5 km, `same_subbasin_knn` intra-sub-basin "
        "k-NN cap 2 km). HGT is a relational encoder over typed EDGES — no fabricated source/"
        "compound node type (C-NODE.1/2, eval_validation.md). Evaluation is row-level, "
        "comparable to the non-graph WALL.",
        "",
        f"**WALL (committed, full run k=8):** XGB spatial AUC = {WALL_XGB_SPATIAL_AUC:.4f}, "
        f"RF = {WALL_RF_SPATIAL_AUC:.4f} (experiments/baseline_t1/metrics_spatial.json).",
        f"**In-run XGB-tabular wall (same 8 folds):** "
        f"{sp['base_references']['xgb_tabular']['metrics']['roc_auc']:.4f}.",
        "",
        "## Spatial-block results (row-level OOF)",
        "",
        "| architecture | AUC OOF | AUC 95% CI | F1@OOF | PR-AUC | bal.acc | Brier | ECE |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
        _arch_row("HGT standalone", A["hgt_standalone"]),
        _arch_row("Embedding fusion (XGB + PCA-HGT)", A["embedding_fusion"]),
        _arch_row("Stacking (HGT+XGB+LGBM meta)", A["stacking"]),
        _arch_row("XGB-tabular (in-run wall)", sp["base_references"]["xgb_tabular"]),
        _arch_row("LGBM-tabular (ref)", sp["base_references"]["lgbm_tabular"]),
    ]
    fus = A["embedding_fusion"]
    lines += [
        "",
        f"**PCA-to-{int(fus['pca_variance_target']*100)}%-variance** kept "
        f"{fus['pca_n_components_mean']:.1f} components on average "
        f"(per fold: {fus['pca_n_components_per_fold']}) out of {m['hidden']} HGT-embed dims.",
        "",
        f"Cross-block edges remaining (must be 0): {sp['n_cross_block_total']}.",
        "",
    ]
    if "random" in out:
        rA = out["random"]["architectures"]
        d = out.get("delta_random_minus_spatial", {})
        lines += [
            "## Δ(random − spatial) — spatial-leakage inflation (C-SPAT.6)",
            "",
            "| architecture | spatial AUC | random AUC | Δ |",
            "|---|---:|---:|---:|",
        ]
        for key, lab in [("hgt_standalone", "HGT standalone"),
                         ("embedding_fusion", "Embedding fusion"),
                         ("stacking", "Stacking")]:
            lines.append(
                f"| {lab} | {A[key]['metrics']['roc_auc']:.4f} | "
                f"{rA[key]['metrics']['roc_auc']:.4f} | {d.get(key, float('nan')):+.4f} |")
        lines.append("")
    if cmp:
        lines += [
            "## Paired tests vs the WALL (8 spatial folds; Nadeau-Bengio + Wilcoxon)",
            "",
            "| architecture | gain vs in-run wall | gain vs committed 0.588 | NB p | Wilcoxon p | verdict |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for name, rec in cmp["by_architecture"].items():
            nb = rec.get("nadeau_bengio", {}).get("p", float("nan"))
            wc = rec.get("wilcoxon", {}).get("p", float("nan"))
            lines.append(
                f"| {name} | {rec.get('gain_vs_in_run_wall', float('nan')):+.4f} | "
                f"{rec.get('gain_vs_committed_wall', float('nan')):+.4f} | "
                f"{nb:.4f} | {wc:.4f} | {rec.get('verdict', 'n/a')} |")
        lines.append("")
        lines += [
            "**Reality rule (eval C-CMP):** a gain counts as robust only if it is paired-",
            "significant (p<0.05) AND exceeds the inter-fold noise bar (0.03 AUC). The honest "
            "question — does graph context beat 0.588 spatial robustly — is answered in the "
            "table above; ~0.60 spatial with no robust gain is the expected, reportable outcome.",
        ]
    (Path(exp_dir) / "REPORT.md").write_text("\n".join(lines) + "\n")
