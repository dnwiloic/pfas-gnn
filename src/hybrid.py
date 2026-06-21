"""Nested-OOF GNN-embedding + XGBoost fusion for T1a (contract §3, EVAL_PROTOCOL_HYBRID.md).

Architecture: for each OUTER spatial block f (leave-one-block-out), we need training-set
embeddings that are ANTI-LEAK (the GNN never saw the label of the row it is embedding) and
test-set embeddings where the test block's labels/edges were never used. This requires a
NESTED GNN training loop (§3.2–§3.3).

Key invariant (must be assertable in the smoke test, §3.4):
  - Every train-row embedding comes from a GNN that saw NEITHER its label NOR a cross-block
    edge connecting to it (inner-OOF, C4 per relation, per inner fold).
  - Every test-row embedding comes from a GNN trained on `train_f` with ALL cross-block
    edges to `test_f` removed (outer C4).
  - Threshold / calibration from inner OOF probas only — never from the test block.

Embedding-alignment caveat (documented, §Watch-outs):
  Each inner GNN (and the test GNN) is initialized independently (same seed=42, but the
  network weights converge differently because the training data changes per fold). The
  pre-head embedding AXES are therefore not shared across folds: one axis in fold j may
  correspond to a completely different latent direction in fold k. XGBoost treats these
  columns as noisy features rather than a shared representation — empirically this is a
  well-known limitation of OOF neural stacking, and it tends to UNDERESTIMATE the true
  benefit of a jointly-trained hybrid (the full-run XGB will partially overcome this via
  bagging). We document but do NOT silently ignore it; the report discusses the gap.
  Mitigation applied here: same seed for all GNN initialisations (deterministic init order
  reduces axis rotation variance). The fixed `hidden` ensures a stable XGB feature schema.

Cost (§C.9 / full-run estimate):
  K outer blocks × J inner blocks = GNN trainings for train embeddings + K GNN trainings
  for test embeddings = K×(J+1) total GNN runs. With K=8 outer (LOBO) and J=4 inner,
  that is 8×5 = 40 GNN trainings for the spatial arm, plus 8 for the random arm = 48.
  Each full-data GNN training takes ~10–20 min on Colab GPU → full hybrid ≈ 8–16 h.
  The spatial arm alone: 40 × 10–20 min = 7–13 h. Use SMOKE_TEST=True for the CPU check.

Usage:
    from src.hybrid import run_hybrid_t1
    results = run_hybrid_t1(df, smoke=True)   # CPU, < 3 min
    results = run_hybrid_t1(df, smoke=False)  # Colab GPU only

Graine fixée 42 partout. Groupé par gm_well_id.
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score,
    precision_score, f1_score, accuracy_score, balanced_accuracy_score,
    brier_score_loss,
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from . import config as C
from . import data as D
from . import features as F
from . import gnn
from . import graph as G
from . import metrics as M
from . import splits as S
from . import targets as T

logger = logging.getLogger(__name__)

SEED = C.SEED

# ----------------------------------------------------------------- smoke params
SMOKE_N_WELLS     = 500      # ~500 wells — keeps inner GNN < 1 min on CPU
SMOKE_OUTER_K     = 3        # 3 outer spatial blocks
SMOKE_INNER_K     = 2        # 2 inner micro-blocks for OOF embeddings
SMOKE_GNN_EPOCHS  = 15
SMOKE_GNN_PATIENCE= 6

# ----------------------------------------------------------------- full params
FULL_OUTER_K      = C.N_SPATIAL_BLOCKS   # 8
FULL_INNER_K      = 4
FULL_GNN_EPOCHS   = 400
FULL_GNN_PATIENCE = 50


# ============================================================= helpers

def _optimal_threshold_f1(y_true: np.ndarray, oof_proba: np.ndarray) -> float:
    """F1-optimal threshold on OOF probas — NEVER called on test data."""
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (oof_proba >= t).astype(int)
        s = f1_score(y_true, pred, zero_division=0)
        if s > best_score:
            best_score, best_t = s, float(t)
    return best_t


def _ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (uniform-width bins)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() == 0:
            continue
        frac_pos = float(y_true[mask].mean())
        mean_conf = float(proba[mask].mean())
        ece += mask.sum() / n * abs(frac_pos - mean_conf)
    return float(ece)


def _cumulative_gain(y_true: np.ndarray, proba: np.ndarray, k_pct: int = 20) -> float:
    """% positifs capturés dans les k% meilleurs scores."""
    n = len(y_true)
    k = max(1, int(n * k_pct / 100))
    idx = np.argsort(proba)[::-1][:k]
    return float(y_true[idx].sum() / max(y_true.sum(), 1))


def _lift_at_k(y_true: np.ndarray, proba: np.ndarray, k_pct: int = 20) -> float:
    """Lift @ k% = (% positifs capturés dans top k%) / (prévalence globale)."""
    gain = _cumulative_gain(y_true, proba, k_pct)
    prev = float(y_true.mean())
    if prev <= 0:
        return float("nan")
    return gain / (k_pct / 100.0) / prev


def _full_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float,
                  calibrated_proba: np.ndarray | None = None) -> dict:
    """§4.3 complete metric set for one fold."""
    pred = (proba >= threshold).astype(int)
    two = len(np.unique(y_true)) > 1
    p_cal = calibrated_proba if calibrated_proba is not None else proba
    out = {
        "roc_auc":           float(roc_auc_score(y_true, proba)) if two else float("nan"),
        "pr_auc":            float(average_precision_score(y_true, proba)) if two else float("nan"),
        "recall":            float(recall_score(y_true, pred, zero_division=0)),
        "precision":         float(precision_score(y_true, pred, zero_division=0)),
        "f1":                float(f1_score(y_true, pred, zero_division=0)),
        "accuracy":          float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier":             float(brier_score_loss(y_true, p_cal)),
        "ece":               _ece(y_true, p_cal),
        "gain_top20pct":     _cumulative_gain(y_true, proba, k_pct=20),
        "lift_top20pct":     _lift_at_k(y_true, proba, k_pct=20),
        "gain_top10pct":     _cumulative_gain(y_true, proba, k_pct=10),
        "threshold_used":    float(threshold),
    }
    return out


def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Mean ± std over folds for each numeric metric."""
    if not fold_metrics:
        return {}
    out = {}
    keys = [k for k in fold_metrics[0] if k != "threshold_used"]
    for k in keys:
        vals = [m[k] for m in fold_metrics if np.isfinite(m.get(k, float("nan")))]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{k}_std"]  = float(np.std(vals)) if vals else float("nan")
    thrs = [m["threshold_used"] for m in fold_metrics]
    out["threshold_mean"] = float(np.mean(thrs))
    return out


def _corrected_resampled_ttest(scores_a: np.ndarray, scores_b: np.ndarray,
                                n_train: int, n_test: int) -> dict:
    """Nadeau-Bengio corrected resampled t-test (2003)."""
    diff = np.asarray(scores_a, float) - np.asarray(scores_b, float)
    k = len(diff)
    mean_diff = float(np.mean(diff))
    var_raw = float(np.var(diff, ddof=1)) if k > 1 else 0.0
    correction = 1.0 / k + n_test / max(n_train, 1)
    var = correction * var_raw
    if var <= 0 or k < 2:
        return {"t": float("nan"), "p": float("nan"), "mean_diff": mean_diff}
    t = mean_diff / np.sqrt(var)
    p = float(2 * (1 - scipy_stats.t.cdf(abs(t), df=k - 1)))
    return {"t": float(t), "p": p, "mean_diff": mean_diff}


def _wilcoxon_paired(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    diff = np.asarray(scores_a, float) - np.asarray(scores_b, float)
    if len(diff) < 4 or np.all(diff == 0):
        return {"w": float("nan"), "p": float("nan")}
    try:
        res = scipy_stats.wilcoxon(diff, alternative="two-sided")
        return {"w": float(res.statistic), "p": float(res.pvalue)}
    except Exception:
        return {"w": float("nan"), "p": float("nan")}


def _bootstrap_ci_by_well(y_true: np.ndarray, proba: np.ndarray,
                           well_ids_row: np.ndarray, n_boot: int = 1000,
                           alpha: float = 0.05, seed: int = SEED) -> dict:
    """IC95% bootstrap PAR GROUPE (well) sur l'AUC OOF concaténée (§4.5).

    Resample at the WELL level (not row level) so pseudo-replicates within a well stay
    together — avoids falsely tight intervals caused by repeated samplings of one well.
    """
    rng = np.random.RandomState(seed)
    unique_wells = np.unique(well_ids_row)
    n_wells = len(unique_wells)
    # build per-well index list
    well_idx = {w: np.where(well_ids_row == w)[0] for w in unique_wells}

    boot_aucs = []
    for _ in range(n_boot):
        sampled_wells = rng.choice(unique_wells, size=n_wells, replace=True)
        row_idxs = np.concatenate([well_idx[w] for w in sampled_wells])
        y_b = y_true[row_idxs]
        p_b = proba[row_idxs]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            boot_aucs.append(float(roc_auc_score(y_b, p_b)))
        except Exception:
            pass

    if not boot_aucs:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    lo = float(np.percentile(boot_aucs, 100 * alpha / 2))
    hi = float(np.percentile(boot_aucs, 100 * (1 - alpha / 2)))
    return {"ci_low": lo, "ci_high": hi, "n_boot": len(boot_aucs)}


# ============================================================= XGBoost factory

def _make_xgb(smoke: bool = False, prevalence: float = 0.445, **kw) -> Any:
    """XGBoost balanced for T1a. T1a is quasi-balanced (44.5%) so scale_pos_weight~1."""
    n_est = kw.pop("n_estimators", 50 if smoke else 300)
    default_spw = 1.0 if prevalence > 0.35 else (1 - prevalence) / max(prevalence, 1e-6)
    spw = kw.pop("scale_pos_weight", default_spw)
    if XGBOOST_AVAILABLE:
        return xgb.XGBClassifier(
            n_estimators=n_est,
            max_depth=kw.pop("max_depth", 6),
            learning_rate=kw.pop("learning_rate", 0.1),
            subsample=kw.pop("subsample", 0.8),
            colsample_bytree=kw.pop("colsample_bytree", 0.8),
            reg_lambda=kw.pop("reg_lambda", 1.0),
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=SEED, verbosity=0,
            **C.xgb_device_params(), **kw,
        )
    # sklearn HGB fallback (no XGBoost installed)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=n_est, random_state=SEED, class_weight="balanced"
    )


# ============================================================= tabular feature matrix

def _tabular_X(df: pd.DataFrame, well_ids: np.ndarray, feature_cols: list[str],
               pipe: F.FeaturePipeline | None = None,
               y: np.ndarray | None = None) -> tuple[np.ndarray, F.FeaturePipeline, list[str]]:
    """Aggregate df to well level, run FeaturePipeline.

    If pipe is None: fits a new pipeline (train call). Otherwise transforms only.
    Returns (X_well[n_wells, d], pipe, feature_names).
    Note: we aggregate to wells first (matching graph node-level), then let XGBoost
    score at the row level by broadcasting (row_to_node), consistent with §4.2.
    """
    wf = G.aggregate_to_wells(df, well_ids, feature_cols)
    if pipe is None:
        # y_node not needed for frequency encoding (no-leak)
        pipe = F.FeaturePipeline(feature_cols, encode="frequency")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X, names = pipe.fit_transform(wf, y)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X, names = pipe.transform(wf)
    return X.astype(np.float32), pipe, list(names)


def _fuse(X_tab: np.ndarray, emb: np.ndarray) -> np.ndarray:
    """Concatenate tabular features and GNN embedding column-wise.
    Both must already be aligned node-to-node (same well order)."""
    return np.hstack([X_tab, emb]).astype(np.float32)


# ============================================================= inner spatial block assignment

def _inner_block_folds(df_train: pd.DataFrame, J: int, seed: int = SEED) -> np.ndarray:
    """Spatial KMeans block assignment on the train-only DataFrame, J inner blocks.

    Returns a per-ROW integer block array (values 0..J-1), grouped by well.
    Used for the J inner plis that generate OOF embeddings for the train rows.
    """
    return S.spatial_block_folds(df_train, k=J, seed=seed)


# ============================================================= core nested-OOF loop

def run_one_outer_fold(
    df: pd.DataFrame,
    y_row: np.ndarray,
    feature_cols: list[str],
    fold_block: np.ndarray,       # outer spatial block assignment (per ROW)
    test_block: int,              # the block held out as TEST in this outer fold
    *,
    inner_k: int = FULL_INNER_K,
    relation: str = "subbasin_knn",
    hidden: int = 64,
    gnn_max_epochs: int = FULL_GNN_EPOCHS,
    gnn_patience: int = FULL_GNN_PATIENCE,
    smoke: bool = False,
    prevalence: float = 0.445,
    seed: int = SEED,
    verbose: bool = False,
) -> dict:
    """Execute ONE outer fold of the nested-OOF hybrid loop (§3.2–§3.3).

    Steps:
      A. Split outer train / outer test.
      B. Build J inner spatial blocks within outer train.
      C. For each inner block j: call train_gnn_and_embed with fit=all-inner-except-j,
         embed=j. Collect OOF embeddings for all outer-train rows.
      D. Call train_gnn_and_embed with fit=train_f, embed=test_block for test embeddings.
      E. Fuse [tabular ⊕ embedding] for train and test.
      F. Train XGBoost on fused train features. Get inner-OOF probas for threshold.
      G. Score test rows at row level (§4.2). Return per-fold metrics dict.

    Anti-leak guards verified inside this function (§3.4):
      - fit_blocks ∩ embed_blocks == ∅ for every GNN call (asserted in train_gnn_and_embed).
      - n_cross_block_remaining == 0 for every GNN call (asserted there).
      - Tabular FeaturePipeline fit on TRAIN wells only.
      - Threshold from inner-OOF only (steps C probas).
    """
    t0 = time.time()

    # A. Outer split -------------------------------------------------------
    train_mask = fold_block != test_block
    test_mask  = fold_block == test_block

    df_train = df[train_mask].reset_index(drop=True)
    y_train  = y_row[train_mask]
    df_test  = df[test_mask].reset_index(drop=True)
    y_test   = y_row[test_mask]

    if len(np.unique(y_test)) < 2 or len(y_test) < 5:
        logger.warning(f"[hybrid fold {test_block}] degenerate test fold — skip")
        return {}

    # Row-to-well maps (aligned to the position in the full df)
    train_row_indices = np.where(train_mask)[0]
    test_row_indices  = np.where(test_mask)[0]

    well_ids_train = df_train[C.WELL_ID].unique()
    well_ids_test  = df_test[C.WELL_ID].unique()

    if verbose:
        logger.info(f"[hybrid fold {test_block}] train={len(df_train)} rows / "
                    f"{len(well_ids_train)} wells; test={len(df_test)} rows / "
                    f"{len(well_ids_test)} wells")

    # B. Inner spatial block assignment (on train-only df) -----------------
    # We pass the train sub-df to spatial_block_folds; this gives per-ROW inner blocks.
    J = min(inner_k, max(2, df_train[C.WELL_ID].nunique() // 3))
    inner_fold_block = _inner_block_folds(df_train, J=J, seed=seed)
    inner_blocks = sorted(set(inner_fold_block.tolist()))

    if verbose:
        logger.info(f"[hybrid fold {test_block}] inner J={J} blocks={inner_blocks}")

    # Well-level node order for train (will be used to assemble OOF embedding matrix)
    _, _, well_to_node_train = G.well_table(df_train)
    n_wells_train = len(well_to_node_train)

    # Pre-allocate OOF embedding storage (well-level, filled per inner fold)
    oof_emb_train = np.full((n_wells_train, hidden), fill_value=float("nan"), dtype=np.float32)
    oof_proba_train_well = np.full(n_wells_train, fill_value=float("nan"), dtype=np.float64)

    # Placeholder XGBoost to accumulate inner-OOF probas for threshold
    inner_oof_proba_rows   = np.full(len(df_train), fill_value=float("nan"), dtype=np.float64)
    inner_oof_y_rows       = y_train.copy().astype(int)

    # C. Inner-OOF loop: produce OOF embeddings for all train rows -----------
    info_inner_list: list = []
    for j in inner_blocks:
        fit_blocks_j   = [b for b in inner_blocks if b != j]
        embed_blocks_j = [j]

        if verbose:
            logger.info(f"  [inner j={j}] fit_blocks={fit_blocks_j} embed_blocks=[{j}]")

        try:
            emb_j, info_j = gnn.train_gnn_and_embed(
                df_train, y_train, feature_cols, inner_fold_block,
                fit_blocks=fit_blocks_j,
                embed_blocks=embed_blocks_j,
                relation=relation,
                hidden=hidden,
                max_epochs=gnn_max_epochs,
                patience=gnn_patience,
                seed=seed,
                verbose=verbose,
            )
        except AssertionError as e:
            logger.error(f"  [inner j={j}] anti-leak assertion FAILED: {e}")
            raise

        # Verify guards (§3.4): 0 cross-block edges must already be guaranteed by the
        # primitive, but we surface them here for the smoke-test audit trail.
        assert info_j.n_cross_block_remaining == 0, (
            f"inner fold j={j}: {info_j.n_cross_block_remaining} cross-block edges remain "
            "(C4 violated in embed graph)"
        )

        # Fill OOF embeddings for the wells in embed_blocks_j
        # info_j.embed_well_ids: array of well_ids whose embedding is in emb_j
        # We need to map those to their position in the TRAIN-ONLY well table.
        for local_i, wid in enumerate(info_j.embed_well_ids):
            node_in_train = well_to_node_train.get(wid)
            if node_in_train is not None:
                oof_emb_train[node_in_train] = emb_j[local_i]

        info_inner_list.append({
            "j": j,
            "n_fit_nodes": info_j.n_fit_nodes,
            "n_embed_nodes": info_j.n_embed_nodes,
            "n_removed_cross_block": info_j.n_removed_cross_block,
            "n_cross_block_remaining": info_j.n_cross_block_remaining,
            "best_epoch": info_j.best_epoch,
            "final_loss": info_j.final_loss,
        })

        # Build inner-OOF XGBoost probas for threshold estimation -------
        # For the embed block j rows, fit XGB on the other inner folds and predict
        inner_val_mask   = inner_fold_block == j
        inner_train_mask = ~inner_val_mask

        # Tabular features for XGB inner train/val (well-level, pipe fit on inner train)
        well_ids_itr = df_train[inner_train_mask][C.WELL_ID].unique()
        wf_agg_inner = G.aggregate_to_wells(df_train, well_ids_itr, feature_cols)
        pipe_inner = F.FeaturePipeline(feature_cols, encode="frequency")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_tab_itr, feat_names = pipe_inner.fit_transform(wf_agg_inner, None)

        # Transform val wells
        well_ids_ival = df_train[inner_val_mask][C.WELL_ID].unique()
        wf_ival = G.aggregate_to_wells(df_train, well_ids_ival, feature_cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_tab_ival, _ = pipe_inner.transform(wf_ival)

        # Embedding for inner train wells (from the OOF fills completed so far, may be
        # partial since we fill as we go — use zeros as fallback for not-yet-filled)
        node_itr = np.array([well_to_node_train[w] for w in well_ids_itr
                              if w in well_to_node_train])
        emb_itr_raw = oof_emb_train[node_itr]
        # NaN slots: prior inner folds may not have embedded these wells yet; use zeros.
        nan_mask_itr = np.isnan(emb_itr_raw).any(axis=1)
        emb_itr_raw[nan_mask_itr] = 0.0

        # Embedding for inner val wells (just filled in emb_j above)
        node_ival = np.array([well_to_node_train[w] for w in well_ids_ival
                               if w in well_to_node_train])
        emb_ival_raw = oof_emb_train[node_ival]
        nan_mask_ival = np.isnan(emb_ival_raw).any(axis=1)
        emb_ival_raw[nan_mask_ival] = 0.0

        X_fuse_itr = _fuse(X_tab_itr, emb_itr_raw)
        X_fuse_ival = _fuse(X_tab_ival, emb_ival_raw)

        y_itr = G.well_majority_target(df_train[inner_train_mask].reset_index(drop=True),
                                       y_train[inner_train_mask], well_ids_itr)
        y_ival_well = G.well_majority_target(df_train[inner_val_mask].reset_index(drop=True),
                                             y_train[inner_val_mask], well_ids_ival)

        if len(np.unique(y_itr)) < 2 or len(y_ival_well) < 2:
            continue

        clf_inner = _make_xgb(smoke=smoke, prevalence=prevalence)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf_inner.fit(X_fuse_itr, y_itr)

        # Broadcast well-level proba to rows in the val mask
        # We score the ROWS to match the test scoring (§4.2)
        proba_ival_well = clf_inner.predict_proba(X_fuse_ival)[:, 1]
        # Build well->proba map
        w2proba_ival = {w: float(proba_ival_well[i])
                        for i, w in enumerate(well_ids_ival) if i < len(proba_ival_well)}
        row_probas_ival = np.array([
            w2proba_ival.get(w, float("nan"))
            for w in df_train[inner_val_mask][C.WELL_ID].to_numpy()
        ])
        inner_oof_proba_rows[inner_val_mask] = row_probas_ival

    # Fill any remaining NaN in OOF embeddings with zeros (degenerate inner folds)
    nan_mask_global = np.isnan(oof_emb_train).any(axis=1)
    if nan_mask_global.any():
        oof_emb_train[nan_mask_global] = 0.0
        logger.debug(f"[hybrid fold {test_block}] {nan_mask_global.sum()} wells with NaN "
                     "embedding (filled with 0)")

    # D. Test embeddings (§3.3): train_f -> embed test block -----------------
    # Outer fold_block values: test_block is the hold-out.
    # fold_block for the FULL df, not train-only: the test wells have block == test_block,
    # train wells have other blocks.  We pass the FULL df + outer fold_block so the
    # primitive can distinguish fit vs embed nodes using block membership.
    fit_blocks_ext  = sorted(set(fold_block.tolist()) - {test_block})
    embed_blocks_ext = [test_block]

    if verbose:
        logger.info(f"[hybrid fold {test_block}] test GNN: "
                    f"fit_blocks={fit_blocks_ext} embed_blocks=[{test_block}]")

    try:
        emb_test, info_test = gnn.train_gnn_and_embed(
            df, y_row, feature_cols, fold_block,
            fit_blocks=fit_blocks_ext,
            embed_blocks=embed_blocks_ext,
            relation=relation,
            hidden=hidden,
            max_epochs=gnn_max_epochs,
            patience=gnn_patience,
            seed=seed,
            verbose=verbose,
        )
    except AssertionError as e:
        logger.error(f"[hybrid fold {test_block}] test embed assertion FAILED: {e}")
        raise

    assert info_test.n_cross_block_remaining == 0, (
        f"outer fold {test_block}: {info_test.n_cross_block_remaining} cross-block edges "
        "remain in test embed graph (C4 violated)"
    )
    assert test_block not in info_test.fit_block_ids, (
        f"test block {test_block} is in fit_block_ids — label leak into test GNN"
    )

    if verbose:
        logger.info(f"  test embed: {emb_test.shape} removed_xblock={info_test.n_removed_cross_block} "
                    f"cross_remaining={info_test.n_cross_block_remaining} best_ep={info_test.best_epoch}")

    # E. Build fused feature matrices (§3.2/§3.3) ---------------------------
    # Train: tabular well-level features (pipe fit on train wells only)
    well_ids_train_ordered, _, _ = G.well_table(df_train)
    X_tab_train, pipe_outer, feat_names = _tabular_X(df_train, well_ids_train_ordered,
                                                      feature_cols)
    # OOF embeddings are already in oof_emb_train aligned to well_ids_train_ordered
    X_fuse_train = _fuse(X_tab_train, oof_emb_train)  # [n_wells_train, d_tab + hidden]

    # Well-level target for XGBoost training
    y_train_well = G.well_majority_target(df_train, y_train, well_ids_train_ordered)

    # Test: tabular features transformed with pipe_outer
    well_ids_test_ordered = info_test.embed_well_ids  # wells in test block
    wf_test = G.aggregate_to_wells(df_test, well_ids_test_ordered, feature_cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_tab_test, _ = pipe_outer.transform(wf_test)
    X_fuse_test = _fuse(X_tab_test, emb_test)  # [n_wells_test, d_tab + hidden]

    # F. Train XGBoost on fused train, threshold from inner OOF (§4.4) ------
    clf_outer = _make_xgb(smoke=smoke, prevalence=prevalence)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_outer.fit(X_fuse_train, y_train_well)

    # Threshold from inner-OOF row probas (C4.4: never from test)
    valid_oof_mask = np.isfinite(inner_oof_proba_rows)
    if valid_oof_mask.sum() >= 10 and len(np.unique(y_train[valid_oof_mask])) >= 2:
        tau = _optimal_threshold_f1(y_train[valid_oof_mask],
                                    inner_oof_proba_rows[valid_oof_mask])
    else:
        tau = 0.5

    # Optional Platt calibration fitted on inner-OOF proba (§4.4)
    calibrated_proba_test = None
    try:
        if valid_oof_mask.sum() >= 20 and len(np.unique(y_train[valid_oof_mask])) >= 2:
            from sklearn.linear_model import LogisticRegression
            platt = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
            oof_p_col = inner_oof_proba_rows[valid_oof_mask].reshape(-1, 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                platt.fit(oof_p_col, y_train[valid_oof_mask].astype(int))
    except Exception as e:
        logger.debug(f"[hybrid fold {test_block}] Platt calibration skipped: {e}")
        platt = None

    # G. Score test rows (§4.2): well proba broadcast to sampling rows -------
    proba_test_well = clf_outer.predict_proba(X_fuse_test)[:, 1]
    # Map: well_id -> proba
    w2proba_test = {w: float(proba_test_well[i])
                    for i, w in enumerate(well_ids_test_ordered)
                    if i < len(proba_test_well)}
    proba_test_row = np.array([
        w2proba_test.get(w, float("nan"))
        for w in df_test[C.WELL_ID].to_numpy()
    ])

    # Calibrated proba for Brier/ECE
    if platt is not None and calibrated_proba_test is None:
        valid_t = np.isfinite(proba_test_row)
        p_cal = np.full_like(proba_test_row, float("nan"))
        if valid_t.any():
            p_cal[valid_t] = platt.predict_proba(
                proba_test_row[valid_t].reshape(-1, 1))[:, 1]
        calibrated_proba_test = p_cal

    # Fill any NaN (wells not in the embed set -> very rare)
    nan_rows = np.isnan(proba_test_row)
    if nan_rows.any():
        proba_test_row[nan_rows] = float(np.nanmean(proba_test_row))

    mets = _full_metrics(y_test.astype(int), proba_test_row, tau, calibrated_proba_test)

    elapsed = time.time() - t0
    if verbose:
        logger.info(f"[hybrid fold {test_block}] AUC={mets['roc_auc']:.4f} "
                    f"recall={mets['recall']:.3f} tau={tau:.2f} elapsed={elapsed:.1f}s")

    return {
        "fold": int(test_block),
        "metrics": mets,
        "info_inner": info_inner_list,
        "info_test": {
            "n_fit_nodes": info_test.n_fit_nodes,
            "n_embed_nodes": info_test.n_embed_nodes,
            "n_removed_cross_block": info_test.n_removed_cross_block,
            "n_cross_block_remaining": info_test.n_cross_block_remaining,
            "best_epoch": info_test.best_epoch,
            "final_loss": info_test.final_loss,
            "relation": info_test.relation,
        },
        "threshold": tau,
        "n_wells_train": len(well_ids_train),
        "n_wells_test": len(well_ids_test),
        "n_rows_test": int(len(y_test)),
        "embed_dim": hidden,
        "n_tab_features": int(X_tab_train.shape[1]),
        "n_fused_features": int(X_fuse_train.shape[1]),
        "elapsed_s": elapsed,
        # keep probas and labels for OOF AUC + bootstrap CI
        "_proba_test_row": proba_test_row.tolist(),
        "_y_test_row": y_test.tolist(),
        "_well_ids_test": df_test[C.WELL_ID].to_numpy().tolist(),
    }


# ============================================================= full three-way CV

def run_hybrid_cv(
    df: pd.DataFrame,
    y_row: np.ndarray,
    feature_cols: list[str],
    fold_block: np.ndarray,
    *,
    inner_k: int = FULL_INNER_K,
    relation: str = "subbasin_knn",
    hidden: int = 64,
    gnn_max_epochs: int = FULL_GNN_EPOCHS,
    gnn_patience: int = FULL_GNN_PATIENCE,
    smoke: bool = False,
    prevalence: float = 0.445,
    seed: int = SEED,
    verbose: bool = False,
    out_dir: Path | None = None,
) -> dict:
    """Run the full nested-OOF hybrid CV (outer = leave-one-block-out or LOBO over
    `fold_block`) and return per-fold metrics, aggregated metrics, and diagnostics.

    Checkpoints each outer fold result to `out_dir/metrics_incremental.json` (§C.8).
    """
    blocks = sorted(set(fold_block.tolist()))
    fold_results = []
    all_proba, all_y, all_well_ids = [], [], []

    prevalence = float(y_row.mean())
    n_tr_mean = int(np.mean([(fold_block != b).sum() for b in blocks]))
    n_te_mean = int(np.mean([(fold_block == b).sum() for b in blocks]))

    for b in blocks:
        logger.info(f"[hybrid/{relation}] outer block {b} / {blocks[-1]} ...")
        res = run_one_outer_fold(
            df, y_row, feature_cols, fold_block, test_block=b,
            inner_k=inner_k, relation=relation, hidden=hidden,
            gnn_max_epochs=gnn_max_epochs, gnn_patience=gnn_patience,
            smoke=smoke, prevalence=prevalence, seed=seed, verbose=verbose,
        )
        if not res:
            continue
        fold_results.append(res)

        # collect for OOF AUC
        all_proba.extend(res["_proba_test_row"])
        all_y.extend(res["_y_test_row"])
        all_well_ids.extend(res["_well_ids_test"])

        # Checkpointing (§C.8)
        if out_dir is not None:
            inc_path = out_dir / "metrics_incremental.json"
            inc_data = {
                "relation": relation,
                "completed_blocks": [r["fold"] for r in fold_results],
                "per_fold": [
                    {"fold": r["fold"], **r["metrics"], "elapsed_s": r["elapsed_s"]}
                    for r in fold_results
                ],
            }
            with open(inc_path, "w") as fh:
                json.dump(inc_data, fh, indent=2, default=str)

    # Aggregate
    fold_metrics = [r["metrics"] for r in fold_results]
    agg = _aggregate_fold_metrics(fold_metrics)

    # Global OOF AUC + bootstrap CI by well (§4.5)
    global_oof_auc = float("nan")
    ci = {"ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    if all_proba:
        yy = np.array(all_y, dtype=int)
        pp = np.array(all_proba, dtype=float)
        ww = np.array(all_well_ids)
        if len(np.unique(yy)) > 1:
            global_oof_auc = float(roc_auc_score(yy, pp))
            ci = _bootstrap_ci_by_well(yy, pp, ww, seed=seed)

    per_fold_aucs = [r["metrics"].get("roc_auc", float("nan")) for r in fold_results]

    return {
        "relation": relation,
        "n_blocks": len(fold_results),
        "aggregated": agg,
        "per_fold": [
            {"fold": r["fold"], **r["metrics"], "elapsed_s": r["elapsed_s"],
             "embed_dim": r["embed_dim"], "n_fused_features": r["n_fused_features"],
             "threshold": r["threshold"]}
            for r in fold_results
        ],
        "global_oof_auc": global_oof_auc,
        "bootstrap_ci_by_well": ci,
        "per_fold_aucs": per_fold_aucs,
        "n_tr_mean": n_tr_mean,
        "n_te_mean": n_te_mean,
    }


# ============================================================= three-way comparison

def run_three_way_comparison(
    spatial_cv: dict,
    random_cv: dict,
    gnn_spatial: dict,
    gnn_random: dict,
    xgb_spatial_auc_mean: float,
    xgb_spatial_aucs: list[float],
    noise_threshold: float = 0.03,
) -> dict:
    """Build the three-way comparison table (§4.3–§4.5).

    Given:
      spatial_cv / random_cv  : hybrid arm outputs (run_hybrid_cv)
      gnn_spatial / gnn_random: GNN-alone arm (from gnn.run_t1_cv summary dict)
      xgb_spatial_auc_mean    : XGB-alone spatial AUC mean (from baselines_t1)
      xgb_spatial_aucs        : XGB-alone per-fold AUC list

    Returns comparison dict with:
      - triplets (random, spatial, delta) per arm
      - paired significance tests (Nadeau-Bengio + Wilcoxon) hybrid vs XGB-alone
      - reality rule verdict (§4.5)
    """
    hyb_sp  = spatial_cv["aggregated"].get("roc_auc_mean", float("nan"))
    hyb_rd  = random_cv["aggregated"].get("roc_auc_mean", float("nan"))
    hyb_sp_aucs = [p["roc_auc"] for p in spatial_cv["per_fold"]
                   if np.isfinite(p.get("roc_auc", float("nan")))]
    gnn_sp  = gnn_spatial.get("auc_mean", float("nan"))
    gnn_rd  = gnn_random.get("auc_mean", float("nan"))
    gnn_sp_aucs = gnn_spatial.get("per_fold_auc", [])

    n_tr = spatial_cv.get("n_tr_mean", 1)
    n_te = spatial_cv.get("n_te_mean", 1)

    # Paired test: hybrid vs XGB alone (spatial arm)
    k = min(len(hyb_sp_aucs), len(xgb_spatial_aucs))
    paired_hyb_xgb = {}
    if k >= 2:
        nb = _corrected_resampled_ttest(hyb_sp_aucs[:k], xgb_spatial_aucs[:k], n_tr, n_te)
        wc = _wilcoxon_paired(hyb_sp_aucs[:k], xgb_spatial_aucs[:k])
        paired_hyb_xgb = {"nadeau_bengio": nb, "wilcoxon": wc,
                          "k_folds": k, "hybrid_aucs": hyb_sp_aucs[:k],
                          "xgb_aucs": xgb_spatial_aucs[:k]}

    # Paired test: hybrid vs GNN alone (spatial arm)
    k2 = min(len(hyb_sp_aucs), len(gnn_sp_aucs))
    paired_hyb_gnn = {}
    if k2 >= 2:
        nb2 = _corrected_resampled_ttest(hyb_sp_aucs[:k2], gnn_sp_aucs[:k2], n_tr, n_te)
        wc2 = _wilcoxon_paired(hyb_sp_aucs[:k2], gnn_sp_aucs[:k2])
        paired_hyb_gnn = {"nadeau_bengio": nb2, "wilcoxon": wc2, "k_folds": k2}

    # Reality rule (§4.5): gain over XGB wall must be significant AND > noise_threshold
    hyb_gain = hyb_sp - xgb_spatial_auc_mean
    p_nb = paired_hyb_xgb.get("nadeau_bengio", {}).get("p", float("nan"))
    p_wc = paired_hyb_xgb.get("wilcoxon", {}).get("p", float("nan"))
    significant = (np.isfinite(p_nb) and p_nb < 0.05) or (np.isfinite(p_wc) and p_wc < 0.05)
    above_noise = abs(hyb_gain) > noise_threshold
    verdict = "real" if (significant and above_noise and hyb_gain > 0) else (
              "spurious (not significant)" if not significant else
              "within_noise" if not above_noise else "negative")

    return {
        "triplets": {
            "hybrid":   {"spatial": hyb_sp, "random": hyb_rd, "delta": hyb_rd - hyb_sp},
            "gnn_alone":{"spatial": gnn_sp, "random": gnn_rd, "delta": gnn_rd - gnn_sp},
            "xgb_alone":{"spatial": xgb_spatial_auc_mean, "random": float("nan"),
                         "delta": float("nan")},
        },
        "paired_hybrid_vs_xgb": paired_hyb_xgb,
        "paired_hybrid_vs_gnn": paired_hyb_gnn,
        "reality_rule": {
            "hybrid_gain_over_xgb_wall": round(float(hyb_gain), 4),
            "significant": bool(significant),
            "above_noise_threshold": bool(above_noise),
            "noise_threshold": noise_threshold,
            "verdict": verdict,
        },
        "bootstrap_ci_spatial_oof": spatial_cv.get("bootstrap_ci_by_well", {}),
    }


# ============================================================= main entry point

def run_hybrid_t1(
    df: pd.DataFrame | None = None,
    *,
    smoke: bool = False,
    relation: str = "subbasin_knn",
    hidden: int = 64,
    inner_k: int | None = None,
    gnn_max_epochs: int | None = None,
    gnn_patience: int | None = None,
    outer_k: int | None = None,
    seed: int = SEED,
    verbose: bool = False,
    out_dir: Path | None = None,
) -> dict:
    """End-to-end hybrid T1a run (both spatial and random arms for the Δ).

    smoke=True:  < 3 min CPU (tiny subsample, 1 inner fold, few epochs).
    smoke=False: COLAB GPU ONLY (see CLAUDE.md §4/§5, full nested-OOF).

    Returns dict with keys: hybrid_spatial, hybrid_random, config, elapsed_s.
    Writes metrics_incremental.json per outer fold to out_dir.
    """
    t_global = time.time()

    # Parameters -------------------------------------------------------
    if smoke:
        _outer_k    = outer_k or SMOKE_OUTER_K
        _inner_k    = inner_k or SMOKE_INNER_K
        _epochs     = gnn_max_epochs or SMOKE_GNN_EPOCHS
        _patience   = gnn_patience or SMOKE_GNN_PATIENCE
        _smoke_n    = SMOKE_N_WELLS
    else:
        _outer_k    = outer_k or FULL_OUTER_K
        _inner_k    = inner_k or FULL_INNER_K
        _epochs     = gnn_max_epochs or FULL_GNN_EPOCHS
        _patience   = gnn_patience or FULL_GNN_PATIENCE
        _smoke_n    = None  # full data

    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Load + target -------------------------------------------------------
    if df is None:
        df = D.load(smoke=smoke, smoke_n=_smoke_n)
    logger.info(f"[hybrid] data: {df.shape}  wells={df[C.WELL_ID].nunique()}")

    y_row = T.build_T1a(df).to_numpy()
    prevalence = float(y_row.mean())
    logger.info(f"[hybrid] T1a prevalence={prevalence:.3f}  n_pos={int(y_row.sum())}/{len(y_row)}")

    feature_cols = C.feature_columns(include_location=False, cocontam="core")
    logger.info(f"[hybrid] feature cols: {len(feature_cols)} (core, no lat/lon)")

    # Splits (§4.1) -------------------------------------------------------
    fold_block_spatial = S.spatial_block_folds(df, k=_outer_k, seed=seed)
    fold_block_random  = S.group_random_folds(df, k=_outer_k, seed=seed)
    S.assert_no_group_leak(df, fold_block_spatial)
    S.assert_no_group_leak(df, fold_block_random)

    # Spatial arm (reference) -------------------------------------------
    logger.info(f"[hybrid] SPATIAL arm — K={_outer_k} outer, J={_inner_k} inner, "
                f"relation={relation}, hidden={hidden}, epochs={_epochs}")
    spatial_dir = out_dir / "spatial" if out_dir else None
    if spatial_dir:
        spatial_dir.mkdir(exist_ok=True)

    spatial_result = run_hybrid_cv(
        df, y_row, feature_cols, fold_block_spatial,
        inner_k=_inner_k, relation=relation, hidden=hidden,
        gnn_max_epochs=_epochs, gnn_patience=_patience,
        smoke=smoke, prevalence=prevalence, seed=seed, verbose=verbose,
        out_dir=spatial_dir,
    )

    # Random arm (Δ only) -----------------------------------------------
    logger.info(f"[hybrid] RANDOM arm — K={_outer_k} outer (Δ measurement only)")
    random_dir = out_dir / "random" if out_dir else None
    if random_dir:
        random_dir.mkdir(exist_ok=True)

    random_result = run_hybrid_cv(
        df, y_row, feature_cols, fold_block_random,
        inner_k=_inner_k, relation=relation, hidden=hidden,
        gnn_max_epochs=_epochs, gnn_patience=_patience,
        smoke=smoke, prevalence=prevalence, seed=seed, verbose=verbose,
        out_dir=random_dir,
    )

    elapsed = time.time() - t_global
    logger.info(f"[hybrid] DONE  spatial_AUC={spatial_result['aggregated'].get('roc_auc_mean', 'n/a'):.3f}  "
                f"random_AUC={random_result['aggregated'].get('roc_auc_mean', 'n/a'):.3f}  "
                f"elapsed={elapsed:.1f}s")

    cfg = {
        "smoke": smoke,
        "outer_k": _outer_k,
        "inner_k": _inner_k,
        "relation": relation,
        "hidden": hidden,
        "gnn_max_epochs": _epochs,
        "gnn_patience": _patience,
        "feature_cols_count": len(feature_cols),
        "prevalence": float(prevalence),
        "seed": seed,
        "xgboost_available": XGBOOST_AVAILABLE,
        "elapsed_s": elapsed,
    }

    return {
        "hybrid_spatial": spatial_result,
        "hybrid_random":  random_result,
        "config": cfg,
        "elapsed_s": elapsed,
    }
