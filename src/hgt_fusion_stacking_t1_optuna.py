"""Optuna HPO for the four components of the HGT-fusion-stacking pipeline (T1a, v2).

Four independent studies, each optimising the OOF spatial AUC on the k=8 LOBO folds:

  1. HGT standalone   — hyperparams: hidden, layers, dropout, heads, lr, weight_decay,
                        k_spatial, cap_km_spatial.
  2. XGBoost tabular  — hyperparams: n_estimators, max_depth, learning_rate, subsample,
                        colsample_bytree, reg_lambda, min_child_weight (or scale_pos_weight).
  3. RF tabular       — hyperparams: n_estimators, max_depth, max_features,
                        min_samples_leaf, class_weight.
  4. Stacking meta    — hyperparams: max_depth, n_estimators, learning_rate of meta-XGB;
                        pca_var for the fusion arm (used in stacking inputs indirectly via
                        the meta-features, which do NOT depend on PCA — so pca_var here
                        governs the fusion head's best-params choice that feeds the stacking).

OBJECTIVE (all four studies):
  Mean per-fold spatial AUC over the k LOBO folds evaluated on GLOBAL OOF rows
  (the same `_per_fold_aucs` function used in the main run). This is consistent with
  how the main run optimises via early-stopping (val AUC per fold). Using PER-FOLD MEAN
  rather than GLOBAL OOF AUC is the correct choice here because:
    - it averages out spatial heterogeneity between blocks, giving more stable signal;
    - it naturally handles blocks with different positive rates (global OOF weights by
      block size, which is dominated by large blocks);
    - it mirrors the paired-test statistic (Nadeau-Bengio) used in the main comparison.

ANTI-LEAK:
  All objectives call the same LOBO backbone as the main experiment — the test block
  data is NEVER used to fit hyperparameters. The LOBO folds for HPO and for the final
  evaluation are the SAME split (seeded K-Means blocks), so best HPs are those that
  maximise MEAN OOF fold AUC.

SMOKE mode:
  n_trials=OPTUNA_TRIALS_SMOKE (3-5), n_blocks=SMOKE_BLOCKS (3), 15 epochs, 500 wells,
  small XGB/RF. Runs in < 3 min CPU total for all four studies.

Usage (module level, CPU smoke test):
    from src.hgt_fusion_stacking_t1_optuna import run_all_studies
    results = run_all_studies(smoke=True)

Style follows src/baselines_t1.py: TPESampler(seed=SEED), optuna.logging.WARNING,
n_trials reduced in smoke, create_study per model.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config as C
from . import hgt_fusion_stacking_t1 as HFS
from .hybrid import _make_xgb  # reuse the factory with correct XGB device params

SEED = C.SEED

# ---- Optuna trial counts (mirrors baselines_t1.py pattern) ----
OPTUNA_TRIALS_SMOKE = 3
OPTUNA_TRIALS_FULL  = 30   # per study; 4 × 30 = 120 total trials on Colab

# ---- Smoke sub-run params ----
SMOKE_N_WELLS  = 500
SMOKE_BLOCKS   = 3
SMOKE_EPOCHS   = 15
SMOKE_PATIENCE = 6

# ---- Full run params ----
FULL_BLOCKS   = C.N_SPATIAL_BLOCKS  # 8
FULL_EPOCHS   = 400
FULL_PATIENCE = 50


# ============================================================= lazy imports
def _optuna():
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna
    except ImportError as e:
        raise ImportError("optuna is required for HPO. "
                          "Install with: pip install optuna>=3.0") from e


def _make_lgbm_factory(smoke: bool, seed: int = SEED):
    """Return a callable that builds a LightGBM (or HistGB fallback) classifier."""
    def factory(**kw):
        try:
            import lightgbm as lgb
            n_est = kw.get("n_estimators", 50 if smoke else 300)
            return lgb.LGBMClassifier(
                n_estimators=n_est,
                num_leaves=kw.get("num_leaves", 31),
                learning_rate=kw.get("learning_rate", 0.05),
                subsample=kw.get("subsample", 0.8),
                colsample_bytree=kw.get("colsample_bytree", 0.8),
                reg_lambda=kw.get("reg_lambda", 1.0),
                random_state=seed,
                n_jobs=-1, verbose=-1,
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_iter=kw.get("n_estimators", 50 if smoke else 300),
                random_state=seed, class_weight="balanced")
    return factory


# ============================================================= shared OOF-AUC objective helper
def _mean_spatial_fold_auc(proba_well, oof, y_row, df) -> float:
    """Compute mean per-fold spatial AUC from a per-well OOF probability vector.

    Mirrors HFS._per_fold_aucs but returns the mean (objective for Optuna).
    NaN folds (degenerate block — fewer than 2 classes) are excluded from the mean.
    Returns 0.5 (random) if no valid fold.
    """
    from sklearn.metrics import roc_auc_score
    proba_row = proba_well[oof.row_to_node]
    block_row = oof.node_block[oof.row_to_node]
    aucs = []
    for b in sorted(set(oof.node_block.tolist())):
        m = (block_row == b) & ~np.isnan(proba_row)
        yt = np.asarray(y_row)[m].astype(int)
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(yt, proba_row[m])))
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else 0.5


# ============================================================= Study 1 — HGT standalone
def optimize_hgt(df, *, feature_cols, n_blocks, smoke, seed=SEED, n_trials=None,
                 verbose=False) -> dict:
    """Optuna study for HGT standalone. Objective = mean per-fold spatial AUC (OOF).

    The backbone (build_oof_backbone) is called inside each trial with the proposed HGT
    HPs. XGB and LGBM base probabilities are computed with DEFAULT params (we only optimise
    the HGT part here). The XGB/LGBM params are irrelevant for the HGT AUC.

    Space:
        hidden       : {32, 64, 96, 128}
        layers       : {1, 2, 3}
        dropout      : [0.1, 0.5]
        heads        : {2, 4, 8} (must divide hidden)
        lr           : [1e-3, 1e-2] log
        weight_decay : [1e-5, 1e-3] log
        k_spatial    : {4, 6, 8, 12}
        cap_km_spatial: [0.5, 3.0]
    """
    optuna = _optuna()
    n_trials = n_trials or (OPTUNA_TRIALS_SMOKE if smoke else OPTUNA_TRIALS_FULL)
    max_epochs = SMOKE_EPOCHS if smoke else FULL_EPOCHS
    patience   = SMOKE_PATIENCE if smoke else FULL_PATIENCE
    y_row = None  # lazy

    # Subsample df once for the study (same sample across trials, like smoke)
    study_df = df
    if smoke and study_df[C.WELL_ID].nunique() > SMOKE_N_WELLS:
        rng = np.random.RandomState(seed)
        keep = set(rng.choice(study_df[C.WELL_ID].unique(), size=SMOKE_N_WELLS, replace=False))
        study_df = study_df[study_df[C.WELL_ID].isin(keep)].reset_index(drop=True)

    from . import targets as T
    y_row = T.build_T1a(study_df).to_numpy()

    def objective(trial):
        hidden_choice = trial.suggest_categorical("hidden", [32, 64, 96, 128])
        # heads must divide hidden
        valid_heads = [h for h in [2, 4, 8] if hidden_choice % h == 0]
        if not valid_heads:
            valid_heads = [1]
        heads = trial.suggest_categorical("heads", valid_heads)
        params = dict(
            hidden=hidden_choice,
            layers=trial.suggest_int("layers", 1, 3 if not smoke else 2),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            heads=heads,
            lr=trial.suggest_float("lr", 1e-3, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            k_spatial=trial.suggest_categorical("k_spatial", [4, 6, 8] if smoke else [4, 6, 8, 12]),
            cap_km_spatial=trial.suggest_float("cap_km_spatial", 0.5, 3.0),
        )
        try:
            oof = HFS.build_oof_backbone(
                study_df,
                feature_cols=feature_cols,
                n_blocks=n_blocks,
                regime="spatial",
                hidden=params["hidden"],
                layers=params["layers"],
                dropout=params["dropout"],
                heads=params["heads"],
                k_spatial=params["k_spatial"],
                cap_km_spatial=params["cap_km_spatial"],
                k_subbasin=8,
                cap_km_subbasin=2.0,
                max_epochs=max_epochs,
                patience=patience,
                lr=params["lr"],
                weight_decay=params["weight_decay"],
                inductive=True,
                smoke=smoke,
                seed=seed,
                verbose=False,
            )
            return _mean_spatial_fold_auc(oof.hgt_proba, oof, y_row, study_df)
        except Exception as e:
            if verbose:
                print(f"[HGT trial {trial.number}] failed: {e}")
            return 0.5

    study = optuna.create_study(
        direction="maximize",
        study_name="hgt_standalone",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
        "study": study,
    }


# ============================================================= Study 2 — XGBoost tabular
def optimize_xgb(df, *, feature_cols, n_blocks, smoke, seed=SEED, n_trials=None,
                 verbose=False) -> dict:
    """Optuna study for XGBoost tabular base. Objective = mean per-fold spatial AUC.

    Uses the same LOBO backbone but fixes HGT params to defaults, then optimises
    only the XGB sub-step inside build_oof_backbone via a direct inner evaluation
    that avoids re-running HGT each trial (HGT is expensive). Instead, we build
    a lightweight standalone LOBO loop for the XGB tabular model only.

    Space (mirrors baselines_t1._tune_xgb):
        n_estimators    : [50, 400]
        max_depth       : [3, 8]
        learning_rate   : [0.05, 0.3] log
        subsample       : [0.6, 1.0]
        colsample_bytree: [0.6, 1.0]
        reg_lambda      : [0.5, 5.0]
        min_child_weight: [1, 10]
    """
    optuna = _optuna()
    n_trials = n_trials or (OPTUNA_TRIALS_SMOKE if smoke else OPTUNA_TRIALS_FULL)

    study_df = df
    if smoke and study_df[C.WELL_ID].nunique() > SMOKE_N_WELLS:
        rng = np.random.RandomState(seed)
        keep = set(rng.choice(study_df[C.WELL_ID].unique(), size=SMOKE_N_WELLS, replace=False))
        study_df = study_df[study_df[C.WELL_ID].isin(keep)].reset_index(drop=True)

    from . import targets as T, splits as S, graph as G
    y_row = T.build_T1a(study_df).to_numpy()
    well_ids, _, well_to_node = G.well_table(study_df)
    y_well = G.well_majority_target(study_df, y_row, well_ids)
    row_to_node = study_df[C.WELL_ID].map(well_to_node).to_numpy().astype(np.int64)

    fold_block_row = S.spatial_block_folds(study_df, k=n_blocks, seed=seed)
    bdf = pd.DataFrame({"w": study_df[C.WELL_ID].to_numpy(), "b": fold_block_row})
    node_block = bdf.groupby("w")["b"].agg(lambda s: int(s.iloc[0])).reindex(well_ids).to_numpy().astype(int)
    blocks = sorted(set(node_block.tolist()))

    prevalence = float(y_well.mean())

    def _xgb_lobo_auc(params):
        """Lightweight XGB LOBO without HGT (fast, for HPO)."""
        from sklearn.metrics import roc_auc_score
        proba_well = np.full(len(well_ids), np.nan)
        for b in blocks:
            test_nodes = node_block == b
            train_mask = ~test_nodes
            X_tab, _ = HFS._tabular_well_matrix(study_df, well_ids, feature_cols, train_mask)
            n_est = params.get("n_estimators", 50 if smoke else 300)
            try:
                clf = _make_xgb(
                    smoke=smoke, prevalence=prevalence,
                    n_estimators=n_est,
                    max_depth=params.get("max_depth", 6),
                    learning_rate=params.get("learning_rate", 0.1),
                    subsample=params.get("subsample", 0.8),
                    colsample_bytree=params.get("colsample_bytree", 0.8),
                    reg_lambda=params.get("reg_lambda", 1.0),
                    min_child_weight=params.get("min_child_weight", 1),
                )
            except TypeError:
                # fallback if _make_xgb doesn't accept min_child_weight in HGB fallback
                clf = _make_xgb(smoke=smoke, prevalence=prevalence, n_estimators=n_est)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_tab[train_mask], y_well[train_mask])
                proba_well[test_nodes] = clf.predict_proba(X_tab[test_nodes])[:, 1]

        proba_row = proba_well[row_to_node]
        block_row = node_block[row_to_node]
        aucs = []
        for b in blocks:
            m = (block_row == b) & ~np.isnan(proba_row)
            yt = y_row[m].astype(int)
            if len(np.unique(yt)) < 2:
                continue
            try:
                aucs.append(float(roc_auc_score(yt, proba_row[m])))
            except Exception:
                continue
        return float(np.mean(aucs)) if aucs else 0.5

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators",
                                              50 if smoke else 100,
                                              100 if smoke else 500,
                                              step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 6 if smoke else 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        try:
            return _xgb_lobo_auc(params)
        except Exception as e:
            if verbose:
                print(f"[XGB trial {trial.number}] failed: {e}")
            return 0.5

    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_tabular",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
        "study": study,
    }


# ============================================================= Study 3 — RF tabular
def optimize_rf(df, *, feature_cols, n_blocks, smoke, seed=SEED, n_trials=None,
                verbose=False) -> dict:
    """Optuna study for Random Forest tabular base. Objective = mean per-fold spatial AUC.

    Space:
        n_estimators   : [50, 500] step 50
        max_depth      : None or [4, 30]
        max_features   : {sqrt, log2, 0.5}
        min_samples_leaf: [1, 20]
        class_weight   : {None, balanced}
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    optuna = _optuna()
    n_trials = n_trials or (OPTUNA_TRIALS_SMOKE if smoke else OPTUNA_TRIALS_FULL)

    study_df = df
    if smoke and study_df[C.WELL_ID].nunique() > SMOKE_N_WELLS:
        rng = np.random.RandomState(seed)
        keep = set(rng.choice(study_df[C.WELL_ID].unique(), size=SMOKE_N_WELLS, replace=False))
        study_df = study_df[study_df[C.WELL_ID].isin(keep)].reset_index(drop=True)

    from . import targets as T, splits as S, graph as G
    y_row = T.build_T1a(study_df).to_numpy()
    well_ids, _, well_to_node = G.well_table(study_df)
    y_well = G.well_majority_target(study_df, y_row, well_ids)
    row_to_node = study_df[C.WELL_ID].map(well_to_node).to_numpy().astype(np.int64)

    fold_block_row = S.spatial_block_folds(study_df, k=n_blocks, seed=seed)
    bdf = pd.DataFrame({"w": study_df[C.WELL_ID].to_numpy(), "b": fold_block_row})
    node_block = bdf.groupby("w")["b"].agg(lambda s: int(s.iloc[0])).reindex(well_ids).to_numpy().astype(int)
    blocks = sorted(set(node_block.tolist()))

    def _rf_lobo_auc(params):
        proba_well = np.full(len(well_ids), np.nan)
        n_est = params.get("n_estimators", 50 if smoke else 200)
        for b in blocks:
            test_nodes = node_block == b
            train_mask = ~test_nodes
            X_tab, _ = HFS._tabular_well_matrix(study_df, well_ids, feature_cols, train_mask)

            max_d = params.get("max_depth", None)
            clf = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                min_samples_leaf=params.get("min_samples_leaf", 5),
                max_features=params.get("max_features", "sqrt"),
                class_weight=params.get("class_weight", None),
                n_jobs=-1, random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_tab[train_mask], y_well[train_mask])
                proba_well[test_nodes] = clf.predict_proba(X_tab[test_nodes])[:, 1]

        proba_row = proba_well[row_to_node]
        block_row = node_block[row_to_node]
        aucs = []
        for b in blocks:
            m = (block_row == b) & ~np.isnan(proba_row)
            yt = y_row[m].astype(int)
            if len(np.unique(yt)) < 2:
                continue
            try:
                aucs.append(float(roc_auc_score(yt, proba_row[m])))
            except Exception:
                continue
        return float(np.mean(aucs)) if aucs else 0.5

    def objective(trial):
        # max_depth: None or an integer
        use_max_depth = trial.suggest_categorical("use_max_depth", [False, True])
        max_depth = (trial.suggest_int("max_depth", 4, 30)
                     if use_max_depth else None)
        params = {
            "n_estimators": trial.suggest_int("n_estimators",
                                              50 if smoke else 100,
                                              100 if smoke else 600,
                                              step=50),
            "max_depth": max_depth,
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features",
                                                       ["sqrt", "log2", 0.5]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        try:
            return _rf_lobo_auc(params)
        except Exception as e:
            if verbose:
                print(f"[RF trial {trial.number}] failed: {e}")
            return 0.5

    study = optuna.create_study(
        direction="maximize",
        study_name="rf_tabular",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
        "study": study,
    }


# ============================================================= Study 4 — Stacking meta-learner
def optimize_stacking(df, *, feature_cols, n_blocks, smoke, seed=SEED, n_trials=None,
                      hgt_params=None, xgb_params=None, verbose=False) -> dict:
    """Optuna study for the stacking meta-learner (meta-XGB on OOF base probas).

    Approach: re-runs `build_oof_backbone` once with the best (or default) HGT/XGB
    params to get the OOF arrays, then optimises the meta-XGB inside `stacking_oof_proba`
    by varying its max_depth, n_estimators, and learning_rate.

    What is being optimised:
        max_depth    : [2, 6]  — meta-learner on 9 meta-features, shallow is preferred
        n_estimators : [50, 300]
        learning_rate: [0.01, 0.3] log

    The base OOF arrays are computed ONCE (expensive: HGT + XGB + LGBM for each block)
    and cached across all trials for efficiency.

    Anti-leak: the meta-learner is trained in a nested LOBO loop inside `stacking_oof_proba`
    — so the meta-learner is never trained on the test block's OOF rows.
    """
    optuna = _optuna()
    n_trials = n_trials or (OPTUNA_TRIALS_SMOKE if smoke else OPTUNA_TRIALS_FULL)
    max_epochs = SMOKE_EPOCHS if smoke else FULL_EPOCHS
    patience   = SMOKE_PATIENCE if smoke else FULL_PATIENCE

    study_df = df
    if smoke and study_df[C.WELL_ID].nunique() > SMOKE_N_WELLS:
        rng = np.random.RandomState(seed)
        keep = set(rng.choice(study_df[C.WELL_ID].unique(), size=SMOKE_N_WELLS, replace=False))
        study_df = study_df[study_df[C.WELL_ID].isin(keep)].reset_index(drop=True)

    from . import targets as T
    y_row = T.build_T1a(study_df).to_numpy()

    # Use best HGT params if provided (or defaults from main run)
    hgt_kw = dict(
        hidden=int((hgt_params or {}).get("hidden", 64)),
        layers=int((hgt_params or {}).get("layers", 2)),
        dropout=float((hgt_params or {}).get("dropout", 0.3)),
        heads=int((hgt_params or {}).get("heads", 4)),
        lr=float((hgt_params or {}).get("lr", 5e-3)),
        weight_decay=float((hgt_params or {}).get("weight_decay", 5e-4)),
        k_spatial=int((hgt_params or {}).get("k_spatial", 8)),
        cap_km_spatial=float((hgt_params or {}).get("cap_km_spatial", 1.5)),
    )

    print(f"  [stacking HPO] building shared OOF backbone (1×, then cached for {n_trials} trials)...")
    t0 = time.time()
    oof = HFS.build_oof_backbone(
        study_df,
        feature_cols=feature_cols,
        n_blocks=n_blocks,
        regime="spatial",
        k_subbasin=8,
        cap_km_subbasin=2.0,
        max_epochs=max_epochs,
        patience=patience,
        inductive=True,
        smoke=smoke,
        seed=seed,
        verbose=False,
        **hgt_kw,
    )
    print(f"  [stacking HPO] backbone ready in {time.time()-t0:.0f}s. Starting meta HPO...")

    from sklearn.metrics import roc_auc_score

    def _meta_lobo_auc(meta_max_depth, meta_n_estimators, meta_lr):
        """Nested-LOBO stacking with custom meta-XGB params. Anti-leak: same as stacking_oof_proba."""
        n = len(oof.well_ids)
        stack = np.full(n, np.nan)
        base = np.vstack([oof.hgt_proba, oof.xgb_proba, oof.lgbm_proba]).T

        # meta-features (same as HFS.stacking_oof_proba)
        mean_p = np.nanmean(base, axis=1)
        std_p = np.nanstd(base, axis=1)
        agree_hx = np.abs(oof.hgt_proba - oof.xgb_proba)
        agree_hl = np.abs(oof.hgt_proba - oof.lgbm_proba)
        agree_xl = np.abs(oof.xgb_proba - oof.lgbm_proba)

        def _ent(p):
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return -(p * np.log(p) + (1 - p) * np.log(1 - p))

        ent = np.nanmean(np.vstack([_ent(oof.hgt_proba), _ent(oof.xgb_proba),
                                    _ent(oof.lgbm_proba)]).T, axis=1)
        feats = np.column_stack([oof.hgt_proba, oof.xgb_proba, oof.lgbm_proba,
                                  mean_p, std_p, agree_hx, agree_hl, agree_xl, ent])
        valid = ~np.isnan(feats).any(axis=1)
        blocks = sorted(set(oof.node_block.tolist()))
        prevalence = float(oof.y_well.mean())

        for b in blocks:
            tr = (oof.node_block != b) & valid
            te = (oof.node_block == b) & valid
            if tr.sum() < 10 or te.sum() < 1 or len(np.unique(oof.y_well[tr])) < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta_clf = _make_xgb(
                    smoke=smoke, prevalence=prevalence,
                    max_depth=meta_max_depth,
                    n_estimators=meta_n_estimators,
                )
                # Override lr via separate step
                meta_clf.set_params(learning_rate=meta_lr)
                meta_clf.fit(feats[tr], oof.y_well[tr])
                stack[te] = meta_clf.predict_proba(feats[te])[:, 1]

        proba_row = stack[oof.row_to_node]
        block_row = oof.node_block[oof.row_to_node]
        aucs = []
        for b in blocks:
            m = (block_row == b) & ~np.isnan(proba_row)
            yt = y_row[m].astype(int)
            if len(np.unique(yt)) < 2:
                continue
            try:
                aucs.append(float(roc_auc_score(yt, proba_row[m])))
            except Exception:
                continue
        return float(np.mean(aucs)) if aucs else 0.5

    def objective(trial):
        meta_max_depth = trial.suggest_int("meta_max_depth", 2, 6 if not smoke else 4)
        meta_n_estimators = trial.suggest_int("meta_n_estimators",
                                               50 if smoke else 50,
                                               100 if smoke else 300,
                                               step=50)
        meta_lr = trial.suggest_float("meta_learning_rate", 0.01, 0.3, log=True)
        try:
            return _meta_lobo_auc(meta_max_depth, meta_n_estimators, meta_lr)
        except Exception as e:
            if verbose:
                print(f"[Stacking trial {trial.number}] failed: {e}")
            return 0.5

    study = optuna.create_study(
        direction="maximize",
        study_name="stacking_meta",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
        "study": study,
        "oof_cached": oof,  # reuse for final run if desired
    }


# ============================================================= run all four studies
def run_all_studies(df=None, *, smoke=False, feature_cols=None, n_blocks=None,
                    n_trials_hgt=None, n_trials_xgb=None, n_trials_rf=None,
                    n_trials_stack=None, seed=SEED, exp_dir=None, write=True,
                    verbose=False):
    """Run all 4 Optuna studies sequentially. Returns a dict with all results + best params.

    In smoke mode: 500 wells, 3 blocks, 3-5 trials per study, CPU < ~3 min.
    In full mode:  all wells, 8 blocks, 30 trials per study, ~2-4 h on Colab GPU.

    Anti-leak guarantee inherited from the per-study objectives:
    - All objectives use the same seeded K-Means LOBO splits.
    - HGT and tabular models are trained only on train-block data; test blocks score is OOF.
    - Meta-learner is optimised in the nested LOBO loop (never trained on test block rows).
    - No PFAS measurement columns in feature_cols (contract from run_v2.py FEATURE_COLS).
    """
    from . import data as D

    t0 = time.time()

    if df is None:
        df = D.load(smoke=smoke, smoke_n=SMOKE_N_WELLS if smoke else None)
    if smoke and df[C.WELL_ID].nunique() > SMOKE_N_WELLS:
        rng = np.random.RandomState(seed)
        keep = set(rng.choice(df[C.WELL_ID].unique(), size=SMOKE_N_WELLS, replace=False))
        df = df[df[C.WELL_ID].isin(keep)].reset_index(drop=True)

    if feature_cols is None:
        feature_cols = [c for c in C.feature_columns(include_location=False,
                                                      cocontam="all", include_air=True)
                        if c not in C.ADMIN_GEO_CAT]

    if n_blocks is None:
        n_blocks = SMOKE_BLOCKS if smoke else FULL_BLOCKS

    print(f"\n{'='*60}")
    print(f"HPO run_all_studies smoke={smoke} n_blocks={n_blocks} "
          f"n_features={len(feature_cols)} seed={seed}")
    print(f"{'='*60}")

    results = {
        "meta": {
            "smoke": bool(smoke), "seed": int(seed), "n_blocks": int(n_blocks),
            "n_features": int(len(feature_cols)),
            "n_trials_per_study": {
                "hgt": n_trials_hgt or (OPTUNA_TRIALS_SMOKE if smoke else OPTUNA_TRIALS_FULL),
                "xgb": n_trials_xgb or (OPTUNA_TRIALS_SMOKE if smoke else OPTUNA_TRIALS_FULL),
                "rf":  n_trials_rf  or (OPTUNA_TRIALS_SMOKE if smoke else OPTUNA_TRIALS_FULL),
                "stacking": n_trials_stack or (OPTUNA_TRIALS_SMOKE if smoke else OPTUNA_TRIALS_FULL),
            },
        },
        "studies": {},
    }

    # ---- Study 1: HGT ----
    print(f"\n[1/4] HGT standalone HPO ...")
    t1 = time.time()
    hgt_res = optimize_hgt(df, feature_cols=feature_cols, n_blocks=n_blocks,
                           smoke=smoke, seed=seed,
                           n_trials=n_trials_hgt, verbose=verbose)
    hgt_res.pop("study")  # not JSON-serialisable; kept in memory for visualisation
    print(f"  best AUC (mean-pfm): {hgt_res['best_value']:.4f}  "
          f"params: {hgt_res['best_params']}  ({time.time()-t1:.0f}s)")
    results["studies"]["hgt_standalone"] = hgt_res

    # ---- Study 2: XGBoost ----
    print(f"\n[2/4] XGBoost tabular HPO ...")
    t2 = time.time()
    xgb_res = optimize_xgb(df, feature_cols=feature_cols, n_blocks=n_blocks,
                           smoke=smoke, seed=seed,
                           n_trials=n_trials_xgb, verbose=verbose)
    xgb_res.pop("study")
    print(f"  best AUC (mean-pfm): {xgb_res['best_value']:.4f}  "
          f"params: {xgb_res['best_params']}  ({time.time()-t2:.0f}s)")
    results["studies"]["xgb_tabular"] = xgb_res

    # ---- Study 3: RF ----
    print(f"\n[3/4] RandomForest tabular HPO ...")
    t3 = time.time()
    rf_res = optimize_rf(df, feature_cols=feature_cols, n_blocks=n_blocks,
                         smoke=smoke, seed=seed,
                         n_trials=n_trials_rf, verbose=verbose)
    rf_res.pop("study")
    print(f"  best AUC (mean-pfm): {rf_res['best_value']:.4f}  "
          f"params: {rf_res['best_params']}  ({time.time()-t3:.0f}s)")
    results["studies"]["rf_tabular"] = rf_res

    # ---- Study 4: Stacking meta ----
    print(f"\n[4/4] Stacking meta-learner HPO ...")
    t4 = time.time()
    stk_res = optimize_stacking(
        df, feature_cols=feature_cols, n_blocks=n_blocks,
        smoke=smoke, seed=seed,
        n_trials=n_trials_stack,
        hgt_params=hgt_res["best_params"],
        xgb_params=xgb_res["best_params"],
        verbose=verbose,
    )
    stk_res.pop("study")
    stk_res.pop("oof_cached", None)  # numpy arrays not JSON-serialisable
    print(f"  best AUC (mean-pfm): {stk_res['best_value']:.4f}  "
          f"params: {stk_res['best_params']}  ({time.time()-t4:.0f}s)")
    results["studies"]["stacking_meta"] = stk_res

    results["meta"]["elapsed_s"] = float(time.time() - t0)
    print(f"\nAll 4 studies done in {results['meta']['elapsed_s']:.0f}s "
          f"({results['meta']['elapsed_s']/60:.1f} min)")

    if write and exp_dir is not None:
        exp_path = Path(exp_dir)
        exp_path.mkdir(parents=True, exist_ok=True)
        out_path = exp_path / "optuna_best_params.json"

        def _default(o):
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if o is None:
                return None
            return str(o)

        out_path.write_text(json.dumps(results, indent=2, default=_default))
        print(f"Best params written to: {out_path}")

    return results
