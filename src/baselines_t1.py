"""Baseline non-graphe pour la tâche T1 (dépassement réglementaire binaire, PFAS CA).

Modèles : LogisticRegression (plancher), RandomForest, XGBoost (ou repli
HistGradientBoosting sklearn si xgboost absent).

Protocole (eval C1-C6, EVAL_PROTOCOL.md) :
  - Cible T1a avec garde-fou détection (C1) ; T1b secondaire.
  - Tous les splits groupés par gm_well_id (C2).
  - Double évaluation : CV spatiale par blocs (référence) + CV aléatoire groupée
    (pour mesurer Δ = artefact spatial).
  - Optimisation de seuil UNIQUEMENT sur probabilités OOF du train (jamais sur test).
  - HP tuning via Optuna en CV interne (groupée+spatiale sur le train).
  - SHAP / importance par permutation sur le meilleur modèle, CV spatiale.

SMOKE_TEST mode :
  - Sous-échantillon ~500 puits, 2 plis externes, 2 plis internes.
  - Exécutable sur CPU en < ~3 min.
  - Estimé la durée du run complet.

Usage :
    from src.baselines_t1 import run_baselines
    results = run_baselines(smoke=False)          # run complet
    results = run_baselines(smoke=True)           # smoke test rapide

Smoke-test (le module utilise des imports relatifs -> lancer comme module/test,
pas comme script isolé) :
    python -m src.baselines_t1
    python tests/test_baselines_t1.py
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
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score,
    precision_score, f1_score, accuracy_score, balanced_accuracy_score,
    brier_score_loss,
)
from sklearn.model_selection import GroupKFold

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from . import config as C
from . import data as D
from . import targets as T
from . import splits as S
from . import features as F

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------- constants
EXPERIMENTS_DIR = C.EXPERIMENTS_DIR / "baseline_t1"
SEED = C.SEED

# Full-run parameters
OUTER_SPATIAL_K   = C.N_SPATIAL_BLOCKS   # 8 spatial outer folds
OUTER_RANDOM_K    = C.N_RANDOM_FOLDS     # 8 random outer folds
INNER_FOLDS_FULL  = 4
OPTUNA_TRIALS_FULL = 20

# Smoke parameters
SMOKE_N_WELLS     = 500      # small but enough for grouped splits
SMOKE_OUTER_K     = 3        # 3 outer spatial folds (still ≥2 to estimate Δ)
SMOKE_INNER_K     = 2
OPTUNA_TRIALS_SMOKE = 3

# ----------------------------------------------------------------- imbalance note
# T1a prevalence ~44.5% (ratio 1:1.2, quasi-équilibré).
# -> Les arbres (RF/XGB) sans pondération sont suffisants.
# -> class_weight="balanced" est proposé dans l'espace Optuna (auto-sélection).
# T1b prevalence ~24.8% (ratio 1:3).
# -> scale_pos_weight=3 par défaut pour XGBoost.
# La pondération est CONDITIONNELLE à la cible et DOCUMENTÉE ici.

# ============================================================= helpers

def _optimal_threshold(y_true: np.ndarray, oof_proba: np.ndarray,
                       metric: str = "f1") -> float:
    """Threshold maximisant `metric` sur probabilités OOF (jamais le test)."""
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        pred = (oof_proba >= t).astype(int)
        if metric == "f1":
            s = f1_score(y_true, pred, zero_division=0)
        elif metric == "balanced_accuracy":
            s = balanced_accuracy_score(y_true, pred)
        else:
            s = f1_score(y_true, pred, zero_division=0)
        if s > best_score:
            best_score, best_t = s, float(t)
    return best_t


def _metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray,
                           threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    n_classes = len(np.unique(y_true))
    return {
        # the 5 required headline metrics
        "roc_auc":           float(roc_auc_score(y_true, proba)) if n_classes > 1 else float("nan"),
        "f1":                float(f1_score(y_true, pred, zero_division=0)),
        "accuracy":          float(accuracy_score(y_true, pred)),
        "recall":            float(recall_score(y_true, pred, zero_division=0)),
        "precision":         float(precision_score(y_true, pred, zero_division=0)),
        # decision-oriented extras
        "pr_auc":            float(average_precision_score(y_true, proba)) if n_classes > 1 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier":             float(brier_score_loss(y_true, proba)),
        "threshold_used":    float(threshold),
    }


def _cumulative_gain(y_true: np.ndarray, proba: np.ndarray, k_pct: int = 20) -> float:
    """% positifs capturés dans le top k% des puits classés."""
    n = len(y_true)
    k = max(1, int(n * k_pct / 100))
    idx = np.argsort(proba)[::-1][:k]
    return float(y_true[idx].sum() / max(y_true.sum(), 1))


def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Mean ± std over folds for each numeric metric."""
    if not fold_metrics:
        return {}
    out = {}
    keys = [k for k in fold_metrics[0] if k != "threshold_used"]
    for k in keys:
        vals = [m[k] for m in fold_metrics
                if not np.isnan(m.get(k, float("nan")))]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{k}_std"]  = float(np.std(vals))  if vals else float("nan")
    thrs = [m["threshold_used"] for m in fold_metrics]
    out["threshold_mean"] = float(np.mean(thrs))
    return out


def _corrected_resampled_ttest(scores_a: np.ndarray, scores_b: np.ndarray,
                                n_train: int, n_test: int) -> dict:
    """Corrected resampled t-test (Nadeau & Bengio 2003).

    Corrects for the train/test correlation inherent to resampled CV.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
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
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    if len(diff) < 4 or np.all(diff == 0):
        return {"w": float("nan"), "p": float("nan")}
    try:
        res = scipy_stats.wilcoxon(diff, alternative="two-sided")
        return {"w": float(res.statistic), "p": float(res.pvalue)}
    except Exception:
        return {"w": float("nan"), "p": float("nan")}


# ============================================================= model factories

def _make_lr(prevalence: float = 0.445) -> LogisticRegression:
    """Logistic regression baseline. T1a quasi-équilibré -> pas de pondération."""
    cw = None if prevalence > 0.35 else "balanced"
    return LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                              class_weight=cw, random_state=SEED)


def _make_rf(prevalence: float = 0.445, smoke: bool = False,
             **kw) -> RandomForestClassifier:
    n_est = kw.get("n_estimators", 50 if smoke else 300)
    return RandomForestClassifier(
        n_estimators=n_est,
        max_depth=kw.get("max_depth", None),
        min_samples_leaf=kw.get("min_samples_leaf", 5),
        max_features=kw.get("max_features", "sqrt"),
        class_weight=kw.get("class_weight", None),  # T1a: no weighting needed
        n_jobs=-1, random_state=SEED,
    )


def _make_xgb(prevalence: float = 0.445, smoke: bool = False, **kw) -> Any:
    n_est = kw.get("n_estimators", 50 if smoke else 300)
    # T1a quasi-équilibré -> scale_pos_weight≈1 ; T1b (prév 25%) -> ~3
    default_spw = 1.0 if prevalence > 0.35 else (1 - prevalence) / max(prevalence, 1e-6)
    spw = kw.get("scale_pos_weight", default_spw)

    if XGBOOST_AVAILABLE:
        return xgb.XGBClassifier(
            n_estimators=n_est,
            max_depth=kw.get("max_depth", 6),
            learning_rate=kw.get("learning_rate", 0.1),
            subsample=kw.get("subsample", 0.8),
            colsample_bytree=kw.get("colsample_bytree", 0.8),
            reg_lambda=kw.get("reg_lambda", 1.0),
            scale_pos_weight=spw,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=SEED, tree_method="hist", verbosity=0,
        )
    # Sklearn HGB fallback
    cw_val = kw.get("class_weight", None)
    return HistGradientBoostingClassifier(
        max_iter=n_est,
        learning_rate=kw.get("learning_rate", 0.1),
        max_depth=kw.get("max_depth", 6),
        l2_regularization=kw.get("reg_lambda", 1.0),
        random_state=SEED, class_weight=cw_val,
    )


def _build_model(model_name: str, params: dict,
                 prevalence: float = 0.445, smoke: bool = False) -> Any:
    if model_name == "LR":
        return _make_lr(prevalence)
    if model_name == "RF":
        return _make_rf(prevalence, smoke, **params)
    if model_name == "XGB":
        return _make_xgb(prevalence, smoke, **params)
    raise ValueError(f"Unknown model: {model_name}")


# ============================================================= inner CV helpers

def _inner_cv_auc(clf_factory, df_tr: pd.DataFrame, y_tr: np.ndarray,
                  fold_inner: np.ndarray, feature_cols: list[str]) -> float:
    """Inner CV AUC for HP evaluation."""
    aucs = []
    for _, tr_j, va_j in S.iter_folds(fold_inner):
        if tr_j.sum() < 15 or va_j.sum() < 5 or len(np.unique(y_tr[va_j])) < 2:
            continue
        pipe = F.FeaturePipeline(feature_cols, encode="target")
        Xtr_j, _ = pipe.fit_transform(df_tr[tr_j], y_tr[tr_j])
        Xva_j, _ = pipe.transform(df_tr[va_j])
        clf = clf_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xtr_j, y_tr[tr_j])
        proba = clf.predict_proba(Xva_j)[:, 1]
        aucs.append(float(roc_auc_score(y_tr[va_j], proba)))
    return float(np.mean(aucs)) if aucs else 0.5


def _oof_proba_inner(clf_factory, df_tr: pd.DataFrame, y_tr: np.ndarray,
                     fold_inner: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    """OOF probabilities on train set (for threshold selection, never touches test)."""
    oof = np.full(len(y_tr), float("nan"))
    for _, tr_j, va_j in S.iter_folds(fold_inner):
        if tr_j.sum() < 15 or va_j.sum() < 5:
            continue
        pipe = F.FeaturePipeline(feature_cols, encode="target")
        Xtr_j, _ = pipe.fit_transform(df_tr[tr_j], y_tr[tr_j])
        Xva_j, _ = pipe.transform(df_tr[va_j])
        clf = clf_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xtr_j, y_tr[tr_j])
        oof[va_j] = clf.predict_proba(Xva_j)[:, 1]
    # fill any unfilled slots with global mean
    nan_mask = np.isnan(oof)
    if nan_mask.any():
        oof[nan_mask] = float(np.nanmean(oof)) if not np.all(nan_mask) else 0.5
    return oof


# ============================================================= Optuna HP search

def _tune_rf(df_tr: pd.DataFrame, y_tr: np.ndarray, fold_inner: np.ndarray,
             feature_cols: list[str], n_trials: int,
             prevalence: float = 0.445, smoke: bool = False) -> dict:
    if not OPTUNA_AVAILABLE or n_trials == 0:
        return {}

    def objective(trial):
        params = {
            "n_estimators":   trial.suggest_int("n_estimators", 50 if smoke else 100,
                                                100 if smoke else 500, step=50),
            "max_depth":      trial.suggest_categorical("max_depth", [None, 8, 16]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            "max_features":   trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight":   trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        return _inner_cv_auc(
            lambda: _make_rf(prevalence, smoke, **params),
            df_tr, y_tr, fold_inner, feature_cols,
        )

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _tune_xgb(df_tr: pd.DataFrame, y_tr: np.ndarray, fold_inner: np.ndarray,
              feature_cols: list[str], n_trials: int,
              prevalence: float = 0.445, smoke: bool = False) -> dict:
    if not OPTUNA_AVAILABLE or n_trials == 0:
        return {}

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 50 if smoke else 100,
                                                 100 if smoke else 400, step=50),
            "max_depth":       trial.suggest_int("max_depth", 3, 6 if smoke else 8),
            "learning_rate":   trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda":      trial.suggest_float("reg_lambda", 0.5, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 3.0),
        }
        return _inner_cv_auc(
            lambda: _make_xgb(prevalence, smoke, **params),
            df_tr, y_tr, fold_inner, feature_cols,
        )

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ============================================================= outer CV runner

def _run_model_cv(
    model_name: str,
    df: pd.DataFrame,
    y: np.ndarray,
    fold_spatial: np.ndarray,
    fold_random: np.ndarray,
    feature_cols: list[str],
    prevalence: float = 0.445,
    smoke: bool = False,
    n_optuna_trials: int = OPTUNA_TRIALS_FULL,
    inner_k: int = INNER_FOLDS_FULL,
) -> dict:
    """Outer double CV (spatial + random) pour un modèle.

    Protocole par pli externe :
      1. Inner CV sur train -> Optuna HP + OOF probas -> seuil optimal.
      2. Ré-entraînement sur tout le train avec best HP.
      3. Évaluation unique sur test externe (seuil figé).
    """
    logger.info(f"[{model_name}] outer CV (spatial+random) ...")

    def _run_one_scheme(outer_fold: np.ndarray, scheme: str) -> list[dict]:
        fold_metrics = []
        for f, tr_mask, te_mask in S.iter_folds(outer_fold):
            df_tr = df[tr_mask].reset_index(drop=True)
            y_tr  = y[tr_mask]
            df_te = df[te_mask].reset_index(drop=True)
            y_te  = y[te_mask]

            if len(np.unique(y_te)) < 2 or len(y_te) < 10:
                logger.warning(f"[{model_name}/{scheme}] fold {f}: degenerate, skip")
                continue

            # inner folds (spatial-grouped)
            k_in = min(inner_k, max(2, df_tr[C.WELL_ID].nunique() // 3))
            if scheme == "spatial":
                fold_inner = S.spatial_block_folds(df_tr, k=k_in)
            else:
                fold_inner = S.group_random_folds(df_tr, k=k_in)

            # HP tuning
            if model_name == "RF":
                best_params = _tune_rf(df_tr, y_tr, fold_inner, feature_cols,
                                       n_optuna_trials, prevalence, smoke)
            elif model_name == "XGB":
                best_params = _tune_xgb(df_tr, y_tr, fold_inner, feature_cols,
                                        n_optuna_trials, prevalence, smoke)
            else:
                best_params = {}

            # OOF probas on train -> threshold (never test data)
            oof = _oof_proba_inner(
                lambda p=best_params: _build_model(model_name, p, prevalence, smoke),
                df_tr, y_tr, fold_inner, feature_cols,
            )
            tau = _optimal_threshold(y_tr, oof, metric="f1")

            # re-train on full outer train
            pipe = F.FeaturePipeline(feature_cols, encode="target")
            Xtr, _ = pipe.fit_transform(df_tr, y_tr)
            Xte, _ = pipe.transform(df_te)

            assert np.isfinite(Xtr).all(), "non-finite train features"
            assert np.isfinite(Xte).all(), "non-finite test features"

            clf = _build_model(model_name, best_params, prevalence, smoke)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(Xtr, y_tr)

            proba_te = clf.predict_proba(Xte)[:, 1]
            m = _metrics_at_threshold(y_te, proba_te, tau)
            m["gain_top20pct"] = _cumulative_gain(y_te, proba_te, k_pct=20)
            fold_metrics.append(m)

            logger.info(
                f"  [{model_name}/{scheme}] fold {f}: "
                f"AUC={m['roc_auc']:.3f}  recall={m['recall']:.3f}  τ={tau:.2f}"
            )
        return fold_metrics

    # spatial CV (référence)
    spatial_metrics = _run_one_scheme(fold_spatial, "spatial")
    # random CV (pour Δ)
    random_metrics  = _run_one_scheme(fold_random, "random")

    agg_sp = _aggregate_fold_metrics(spatial_metrics)
    agg_rd = _aggregate_fold_metrics(random_metrics)

    delta = {
        k.replace("_mean", ""): round(
            agg_rd.get(k, float("nan")) - agg_sp.get(k, float("nan")), 4
        )
        for k in agg_sp if k.endswith("_mean") and k != "threshold_mean"
    }

    # global OOF AUC (concaténation plis spatiaux)
    # rebuild by re-running predict over all spatial test folds (no data from test leaked,
    # just reusing trained models for aggregation; note: best_params varies per fold)
    global_auc = float("nan")
    try:
        all_p, all_y = [], []
        for f, tr_mask, te_mask in S.iter_folds(fold_spatial):
            df_tr = df[tr_mask].reset_index(drop=True)
            y_tr  = y[tr_mask]
            df_te = df[te_mask].reset_index(drop=True)
            y_te  = y[te_mask]
            if len(np.unique(y_te)) < 2:
                continue
            pipe = F.FeaturePipeline(feature_cols, encode="target")
            Xtr, _ = pipe.fit_transform(df_tr, y_tr)
            Xte, _ = pipe.transform(df_te)
            clf = _build_model(model_name, {}, prevalence, smoke)  # default params
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(Xtr, y_tr)
            all_p.append(clf.predict_proba(Xte)[:, 1])
            all_y.append(y_te)
        if all_p:
            global_auc = float(roc_auc_score(
                np.concatenate(all_y), np.concatenate(all_p)
            ))
    except Exception as e:
        logger.warning(f"[{model_name}] global OOF AUC failed: {e}")

    return {
        "model": model_name,
        "spatial": agg_sp,
        "random":  agg_rd,
        "delta":   delta,
        "global_auc_spatial_oof": global_auc,
        "fold_metrics_spatial": spatial_metrics,
        "fold_metrics_random":  random_metrics,
        "xgboost_used": XGBOOST_AVAILABLE if model_name == "XGB" else None,
    }


# ============================================================= SHAP / importance

def _compute_importances(clf, feature_names: list[str],
                         Xtr: np.ndarray, Xte: np.ndarray,
                         y_te: np.ndarray) -> pd.DataFrame:
    """SHAP tree explainer si dispo, sinon permutation importance."""
    imp, method = None, "unknown"
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(Xte)
            if isinstance(sv, list):
                sv = sv[1]
            imp = np.abs(sv).mean(axis=0)
            method = "SHAP_TreeExplainer"
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")

    if imp is None:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(clf, Xte, y_te, n_repeats=10,
                                   random_state=SEED, scoring="roc_auc")
        imp = r.importances_mean
        method = "permutation_importance"

    df = pd.DataFrame({"feature": feature_names, "importance": imp, "method": method})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def run_shap_analysis(
    model_name: str,
    df: pd.DataFrame,
    y: np.ndarray,
    fold_spatial: np.ndarray,
    feature_cols: list[str],
    prevalence: float = 0.445,
    smoke: bool = False,
) -> pd.DataFrame:
    """SHAP / importance sur le premier pli spatial non-dégénéré."""
    for f, tr_mask, te_mask in S.iter_folds(fold_spatial):
        y_te = y[te_mask]
        if len(np.unique(y_te)) < 2 or te_mask.sum() < 20:
            continue
        df_tr = df[tr_mask].reset_index(drop=True)
        y_tr  = y[tr_mask]
        df_te = df[te_mask].reset_index(drop=True)

        pipe = F.FeaturePipeline(feature_cols, encode="target")
        Xtr, names = pipe.fit_transform(df_tr, y_tr)
        Xte, _     = pipe.transform(df_te)

        clf = _build_model(model_name, {}, prevalence, smoke)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xtr, y_tr)

        imp = _compute_importances(clf, names, Xtr, Xte, y_te)
        imp["fold"] = f
        return imp

    return pd.DataFrame()


# ============================================================= ablations

def run_ablations(
    model_name: str,
    df: pd.DataFrame,
    y: np.ndarray,
    fold_spatial: np.ndarray,
    fold_random: np.ndarray,
    prevalence: float = 0.445,
    smoke: bool = False,
    inner_k: int = INNER_FOLDS_FULL,
) -> dict:
    """4 ablations de configuration de features : loc, cocontam, air.

    (a) no_loc_all   — baseline (reference)
    (b) with_loc_all — lat/lon inclus
    (c) no_loc_core  — cocontam core seulement
    (d) no_loc_none  — sans cocontam
    """
    n_trials = 0  # no Optuna for ablations (use defaults for speed)
    cfgs = {
        "no_loc_all":   dict(include_location=False, cocontam="all",  include_air=True),
        "with_loc_all": dict(include_location=True,  cocontam="all",  include_air=True),
        "no_loc_core":  dict(include_location=False, cocontam="core", include_air=True),
        "no_loc_none":  dict(include_location=False, cocontam="none", include_air=True),
    }
    results = {}
    for key, cfg in cfgs.items():
        cols = C.feature_columns(**cfg)
        logger.info(f"  Ablation {key}: {len(cols)} cols")
        res = _run_model_cv(
            model_name, df, y, fold_spatial, fold_random, cols,
            prevalence=prevalence, smoke=smoke,
            n_optuna_trials=n_trials, inner_k=inner_k,
        )
        results[key] = {
            "n_features":       len(cols),
            "spatial_roc_auc":  res["spatial"].get("roc_auc_mean", float("nan")),
            "random_roc_auc":   res["random"].get("roc_auc_mean", float("nan")),
            "delta_roc_auc":    res["delta"].get("roc_auc", float("nan")),
            "spatial_recall":   res["spatial"].get("recall_mean", float("nan")),
        }
    return results


# ============================================================= main entry point

def run_baselines(
    smoke: bool = False,
    target: str = "T1a",
    run_ablations_flag: bool = True,
    run_shap_flag: bool = True,
    n_optuna_trials: int | None = None,
    save_dir: Path | None = None,
) -> dict:
    """Train LR / RF / XGB on T1a (or T1b) with the full double-CV protocol.

    Args:
        smoke:             Use tiny subsample; fast CPU path (< 3 min).
        target:            "T1a" or "T1b".
        run_ablations_flag: Run feature ablations (RF only).
        run_shap_flag:     Compute SHAP / permutation importances.
        n_optuna_trials:   Override Optuna trial count.
        save_dir:          Where to write artefacts.
    Returns:
        dict with model_results, comparison, importance, ablations, config, block_prevalence.
    """
    t_global = time.time()
    save_dir = Path(save_dir) if save_dir else EXPERIMENTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # smoke-mode parameters
    outer_k     = SMOKE_OUTER_K       if smoke else OUTER_SPATIAL_K
    inner_k     = SMOKE_INNER_K       if smoke else INNER_FOLDS_FULL
    n_trials    = n_optuna_trials or (OPTUNA_TRIALS_SMOKE if smoke else OPTUNA_TRIALS_FULL)
    smoke_n     = SMOKE_N_WELLS

    logger.info(f"run_baselines: target={target} smoke={smoke} outer_k={outer_k} "
                f"inner_k={inner_k} n_trials={n_trials}")

    # 1. Load
    df = D.load(smoke=smoke, smoke_n=smoke_n)
    logger.info(f"Data: {df.shape}  wells={df[C.WELL_ID].nunique()}")

    # 2. Target
    y_series = T.build_T1a(df) if target == "T1a" else T.build_T1b(df)
    y = y_series.to_numpy()
    prevalence = float(y.mean())
    logger.info(f"Target {target}: prevalence={prevalence:.3f}  n_pos={int(y.sum())}/{len(y)}")

    # 3. Splits
    fold_spatial = S.spatial_block_folds(df, k=outer_k)
    fold_random  = S.group_random_folds(df, k=outer_k)
    S.assert_no_group_leak(df, fold_spatial)
    S.assert_no_group_leak(df, fold_random)
    bp = S.block_prevalence(y, fold_spatial)
    logger.info(f"Spatial blocks: {len(bp)}  prev [{bp.prevalence.min():.2f},{bp.prevalence.max():.2f}]")

    # 4. Features
    feature_cols = C.feature_columns(include_location=False, cocontam="all", include_air=True)
    logger.info(f"Feature candidates: {len(feature_cols)}")

    # 5. Model loop
    model_results: dict[str, dict] = {}
    for mname in ("LR", "RF", "XGB"):
        t_m = time.time()
        trials_m = n_trials if mname in ("RF", "XGB") else 0
        res = _run_model_cv(
            mname, df, y, fold_spatial, fold_random, feature_cols,
            prevalence=prevalence, smoke=smoke,
            n_optuna_trials=trials_m, inner_k=inner_k,
        )
        res["elapsed_s"] = time.time() - t_m
        model_results[mname] = res
        logger.info(
            f"[{mname}] {res['elapsed_s']:.1f}s  "
            f"spatial AUC={res['spatial'].get('roc_auc_mean', float('nan')):.3f}  "
            f"random AUC={res['random'].get('roc_auc_mean', float('nan')):.3f}  "
            f"Δ={res['delta'].get('roc_auc', float('nan')):+.3f}"
        )

    # 6. Paired comparisons
    comparison: dict[str, dict] = {}
    n_tr_mean = int(np.mean([tr.sum() for _, tr, _ in S.iter_folds(fold_spatial)]))
    n_te_mean = int(np.mean([te.sum() for _, _, te in S.iter_folds(fold_spatial)]))
    for a, b in [("RF", "LR"), ("XGB", "LR"), ("RF", "XGB")]:
        aucs_a = [m["roc_auc"] for m in model_results[a]["fold_metrics_spatial"]
                  if np.isfinite(m.get("roc_auc", float("nan")))]
        aucs_b = [m["roc_auc"] for m in model_results[b]["fold_metrics_spatial"]
                  if np.isfinite(m.get("roc_auc", float("nan")))]
        k = min(len(aucs_a), len(aucs_b))
        if k < 2:
            continue
        nb = _corrected_resampled_ttest(aucs_a[:k], aucs_b[:k], n_tr_mean, n_te_mean)
        wc = _wilcoxon_paired(aucs_a[:k], aucs_b[:k])
        comparison[f"{a}_vs_{b}"] = {
            "nadeau_bengio": nb,
            "wilcoxon": wc,
            "scores_a": aucs_a[:k],
            "scores_b": aucs_b[:k],
            "noise_threshold_auc": 0.03,
        }

    # 7. SHAP
    importance_df = pd.DataFrame()
    if run_shap_flag:
        best_m = max(model_results,
                     key=lambda m: model_results[m]["spatial"].get("roc_auc_mean", 0))
        logger.info(f"SHAP on {best_m} ...")
        importance_df = run_shap_analysis(
            best_m, df, y, fold_spatial, feature_cols,
            prevalence=prevalence, smoke=smoke,
        )

    # 8. Ablations
    ablation_results: dict[str, dict] = {}
    if run_ablations_flag:
        logger.info("Ablations (RF, no Optuna) ...")
        ablation_results = run_ablations(
            "RF", df, y, fold_spatial, fold_random,
            prevalence=prevalence, smoke=smoke, inner_k=inner_k,
        )

    # 9. Save artefacts
    elapsed_total = time.time() - t_global

    cfg_out = {
        "target": target,
        "smoke": smoke,
        "outer_k": outer_k,
        "inner_k": inner_k,
        "n_optuna_trials": n_trials,
        "feature_cols_count": len(feature_cols),
        "prevalence": float(prevalence),
        "seed": int(SEED),
        "xgboost_available": XGBOOST_AVAILABLE,
        "optuna_available": OPTUNA_AVAILABLE,
        "shap_available": SHAP_AVAILABLE,
        "elapsed_s": elapsed_total,
    }
    with open(save_dir / "config.yaml", "w") as fh:
        for k, v in cfg_out.items():
            fh.write(f"{k}: {v}\n")

    metrics_out = {
        "models": {
            mn: {
                "spatial": res["spatial"],
                "random":  res["random"],
                "delta":   res["delta"],
                "global_auc_spatial_oof": res["global_auc_spatial_oof"],
                "elapsed_s": res.get("elapsed_s", 0),
                "xgboost_used": res.get("xgboost_used"),
            }
            for mn, res in model_results.items()
        },
        "comparisons": comparison,
        "ablations": ablation_results,
        "block_prevalence": bp.to_dict(orient="records"),
        "config": cfg_out,
    }
    with open(save_dir / "metrics.json", "w") as fh:
        json.dump(metrics_out, fh, indent=2, default=str)

    if not importance_df.empty:
        importance_df.to_csv(save_dir / "feature_importance.csv", index=False)

    logger.info(f"Artefacts -> {save_dir}")
    logger.info(f"Total: {elapsed_total:.1f}s")

    return {
        "model_results": model_results,
        "comparison": comparison,
        "importance": importance_df,
        "ablations": ablation_results,
        "config": cfg_out,
        "block_prevalence": bp,
        "save_dir": save_dir,
    }


# ============================================================= __main__ (smoke)

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)

    print("=" * 60)
    print("SMOKE TEST  baselines_t1.py")
    print(f"smoke_n={SMOKE_N_WELLS} wells, outer_k={SMOKE_OUTER_K}, "
          f"inner_k={SMOKE_INNER_K}, n_trials={OPTUNA_TRIALS_SMOKE}")
    print("=" * 60)

    t0 = time.time()
    results = run_baselines(
        smoke=True, target="T1a",
        run_ablations_flag=True, run_shap_flag=True,
        n_optuna_trials=OPTUNA_TRIALS_SMOKE,
    )
    elapsed = time.time() - t0

    print("\n--- Scores spatial: les 5 métriques (+ Δ AUC random−spatial) ---")
    print(f"{'Model':<6}  {'AUC':>6}  {'F1':>6}  {'Acc':>6}  {'Recall':>6}  "
          f"{'Prec':>6}  | {'AUC_rd':>7}  {'Δ_AUC':>7}")
    for mn, res in results["model_results"].items():
        sp, rd, dlt = res["spatial"], res["random"], res["delta"]
        g = lambda d, k: d.get(f"{k}_mean", float("nan"))
        print(
            f"  {mn:<4}  {g(sp,'roc_auc'):>6.3f}  {g(sp,'f1'):>6.3f}"
            f"  {g(sp,'accuracy'):>6.3f}  {g(sp,'recall'):>6.3f}  {g(sp,'precision'):>6.3f}"
            f"  | {g(rd,'roc_auc'):>7.3f}  {dlt.get('roc_auc', float('nan')):>+7.3f}"
        )

    print("\n--- Paired comparisons (Nadeau-Bengio corrected t-test) ---")
    for pair, comp in results["comparison"].items():
        nb = comp["nadeau_bengio"]
        wc = comp["wilcoxon"]
        print(f"  {pair}: Δ={nb['mean_diff']:+.3f}  NB-t p={nb['p']:.3f}  Wilcoxon p={wc['p']:.3f}")

    if not results["importance"].empty:
        print("\n--- Top 10 features ---")
        top = results["importance"].head(10)
        for _, row in top.iterrows():
            print(f"  {row['feature']:<40} {row['importance']:.4f}  [{row['method']}]")

    if results["ablations"]:
        print("\n--- Ablations (RF) ---")
        print(f"  {'config':<20}  {'AUC_sp':>8}  {'AUC_rd':>8}  {'Δ':>7}")
        for key, abl in results["ablations"].items():
            print(f"  {key:<20}  {abl['spatial_roc_auc']:>8.3f}  "
                  f"{abl['random_roc_auc']:>8.3f}  {abl['delta_roc_auc']:>+7.3f}")

    n_full = 11333
    est = elapsed * n_full / SMOKE_N_WELLS
    print(f"\nSmoke elapsed: {elapsed:.1f}s")
    print(f"Estimated full run: {est/60:.0f} min (×{n_full/SMOKE_N_WELLS:.0f} wells, "
          f"×{OUTER_SPATIAL_K/SMOKE_OUTER_K:.0f} folds, "
          f"×{OPTUNA_TRIALS_FULL/OPTUNA_TRIALS_SMOKE:.0f} Optuna trials)")
    print("SMOKE TEST DONE.")
