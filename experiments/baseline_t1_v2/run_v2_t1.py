"""Tabular WALL for T1a on the v2 dataset — strict spatial protocol + SHAP + training curves.

What this does (one run):
  1. RF + XGBoost, CV spatiale par blocs (reference) + CV aleatoire groupee (Delta),
     grouped by gm_well_id, strict blocklist (no PFAS measurement, no lat/lon,
     gm_dataset_name excluded), detection-guarded T1a (C1), OOF F1 threshold.
     -> the (random, spatial, Delta) triplet, the honest spatial wall.
  2. XGBoost training curves per spatial fold: train vs held-out-block logloss AND AUC
     over boosting rounds (CLAUDE.md 3.8). NB: the "val" curve is the held-out spatial
     block, used ONLY as a diagnostic (no early-stop, fixed n_estimators) -> it shows
     real geographic generalization. RF is non-iterative -> no boosting curve (justified).
  3. SHAP (TreeExplainer) on a full-data XGBoost with the strict feature set: global
     feature-importance bar + beeswarm, top-20 saved to JSON. Sanity check vs the hydro
     expectation (geotracker > depth/screen > soil retention > land use).

    python experiments/baseline_t1_v2/run_v2_t1.py            # full (k=8), ~30 min CPU
    SMOKE_TEST=1 python experiments/baseline_t1_v2/run_v2_t1.py   # quick CPU check
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config as C
from src import data as D
from src import targets as T
from src import splits as S
from src import features as F
from src import metrics as MM
from src import progress as P

import xgboost as xgb

SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
# ABLATION="pure_mech" drops the whole administrative block (county/dwr/sgma/regional_board)
# to measure the wall WITHOUT spatial-prevalence memorization (SHAP showed dwr_basin__enc +
# sgma_subbasin_name__enc dominating). Keeps every genuine-mechanism feature.
ABLATION = os.environ.get("ABLATION", "").strip()
K = 2 if SMOKE else C.N_SPATIAL_BLOCKS
INNER_K = 2 if SMOKE else 3
N_EST = 120 if SMOKE else 400
OUT = Path(__file__).resolve().parent
SUFFIX = f"_{ABLATION}" if ABLATION else ""
FIG = OUT / (f"figures_{ABLATION}" if ABLATION else "figures")
FIG.mkdir(parents=True, exist_ok=True)
FEATURE_COLS = C.feature_columns(include_location=False, cocontam="all", include_air=True)
if ABLATION == "pure_mech":
    FEATURE_COLS = [c for c in FEATURE_COLS if c not in C.ADMIN_GEO_CAT]


def make_rf():
    return RandomForestClassifier(
        n_estimators=120 if SMOKE else 300, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=C.SEED)


def make_xgb(eval_metric="logloss"):
    return xgb.XGBClassifier(
        n_estimators=N_EST, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric=eval_metric, verbosity=0, random_state=C.SEED,
        **C.xgb_device_params())


def _proba(m, X):
    return m.predict_proba(X)[:, 1]


def oof_threshold(df_tr, y_tr, make_model):
    """F1-optimal threshold from a grouped+spatial inner CV on TRAIN only (no test leak)."""
    fold = S.spatial_block_folds(df_tr, k=INNER_K)
    oof = np.full(len(y_tr), np.nan)
    for _, tr, va in S.iter_folds(fold):
        if tr.sum() < 20 or va.sum() < 5 or len(np.unique(y_tr[tr])) < 2:
            continue
        pipe = F.FeaturePipeline(FEATURE_COLS, encode="target")
        Xtr, _ = pipe.fit_transform(df_tr[tr], y_tr[tr])
        Xva, _ = pipe.transform(df_tr[va])
        m = make_model(); m.fit(Xtr, y_tr[tr])
        oof[va] = _proba(m, Xva)
    oof = np.where(np.isnan(oof), np.nanmean(oof) if np.isfinite(np.nanmean(oof)) else 0.5, oof)
    grid = np.linspace(0.1, 0.9, 33)
    f1s = [f1_score(y_tr, (oof >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def run_scheme(df, y, fold, make_model, scheme, name, capture_curves=False):
    """Run one CV scheme; optionally capture per-fold XGB train/val curves."""
    rows, curves = [], []
    folds = list(S.iter_folds(fold))
    dev = "GPU" if (name == "XGB" and C.gpu_available()) else "CPU"
    for f, tr, te in P.track(folds, total=len(folds), desc=f"{name}/{scheme} [{dev}]"):
        if len(np.unique(y[te])) < 2 or te.sum() < 10:
            continue
        df_tr, df_te = df[tr].reset_index(drop=True), df[te].reset_index(drop=True)
        ytr, yte = y[tr], y[te]
        tau = oof_threshold(df_tr, ytr, make_model)
        pipe = F.FeaturePipeline(FEATURE_COLS, encode="target")
        Xtr, names = pipe.fit_transform(df_tr, ytr)
        Xte, _ = pipe.transform(df_te)

        if capture_curves and name == "XGB":
            # Monitor train vs held-out spatial block over boosting rounds (diagnostic only).
            m = xgb.XGBClassifier(
                n_estimators=N_EST, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                eval_metric=["logloss", "auc"], verbosity=0, random_state=C.SEED,
                **C.xgb_device_params())
            m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xte, yte)], verbose=False)
            ev = m.evals_result_
            curves.append({
                "fold": int(f),
                "train_logloss": ev["validation_0"]["logloss"],
                "val_logloss": ev["validation_1"]["logloss"],
                "train_auc": ev["validation_0"]["auc"],
                "val_auc": ev["validation_1"]["auc"],
            })
        else:
            m = make_model(); m.fit(Xtr, ytr)
        rows.append(MM.binary_metrics(yte, _proba(m, Xte), tau))

    keys = ["roc_auc", "f1", "accuracy", "recall", "precision", "pr_auc", "balanced_accuracy"]
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    agg.update({f"{k}_std": float(np.nanstd([r[k] for r in rows]))
                for k in ["roc_auc", "f1", "accuracy", "recall", "precision"]})
    agg["n_folds"] = len(rows)
    return agg, rows, curves


# --------------------------------------------------------------------- plotting
def plot_curves(curves, kind, fname, title):
    """Plot per-fold train/val curves (logloss or auc) + mean band."""
    if not curves:
        return
    tr_key, va_key = f"train_{kind}", f"val_{kind}"
    fig, ax = plt.subplots(figsize=(8, 5))
    min_len = min(len(c[tr_key]) for c in curves)
    tr = np.array([c[tr_key][:min_len] for c in curves])
    va = np.array([c[va_key][:min_len] for c in curves])
    x = np.arange(1, min_len + 1)
    for i, c in enumerate(curves):
        ax.plot(x, c[tr_key][:min_len], color="tab:blue", alpha=0.18, lw=0.8)
        ax.plot(x, c[va_key][:min_len], color="tab:red", alpha=0.18, lw=0.8)
    ax.plot(x, tr.mean(0), color="tab:blue", lw=2.2, label="train (mean over folds)")
    ax.plot(x, va.mean(0), color="tab:red", lw=2.2, label="val = held-out spatial block (mean)")
    ax.fill_between(x, va.mean(0) - va.std(0), va.mean(0) + va.std(0),
                    color="tab:red", alpha=0.12)
    ax.set_xlabel("boosting round")
    ax.set_ylabel(kind)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)
    print(f"[fig] {FIG/fname}")


def run_shap(df, y):
    """Global SHAP on a full-data XGB with strict features (interpretability sanity)."""
    import shap
    pipe = F.FeaturePipeline(FEATURE_COLS, encode="target")
    X, names = pipe.fit_transform(df, y)
    m = make_xgb(); m.fit(X, y)
    rng = np.random.default_rng(C.SEED)
    n_bg = min(2000, len(X))
    idx = rng.choice(len(X), n_bg, replace=False)
    expl = shap.TreeExplainer(m)
    sv = expl.shap_values(X[idx])
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(mean_abs)[::-1]
    top = [{"feature": names[i], "mean_abs_shap": float(mean_abs[i])} for i in order[:25]]

    # bar
    fig, ax = plt.subplots(figsize=(8, 8))
    top20 = top[:20][::-1]
    ax.barh([t["feature"] for t in top20], [t["mean_abs_shap"] for t in top20],
            color="tab:purple")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("T1a v2 — global feature importance (SHAP, full-data XGB)")
    fig.tight_layout(); fig.savefig(FIG / "shap_summary_bar.png", dpi=130); plt.close(fig)
    print(f"[fig] {FIG/'shap_summary_bar.png'}")

    # beeswarm
    try:
        shap.summary_plot(sv, X[idx], feature_names=names, max_display=20, show=False)
        plt.tight_layout(); plt.savefig(FIG / "shap_beeswarm.png", dpi=130); plt.close()
        print(f"[fig] {FIG/'shap_beeswarm.png'}")
    except Exception as e:
        print(f"[shap] beeswarm skipped: {e}")
    return top


def main():
    t0 = time.time()
    df = D.load(smoke=SMOKE, smoke_n=800)
    y = T.build_T1a(df).to_numpy()
    print(f"[data] v2 rows={len(df)} wells={df[C.WELL_ID].nunique()} prev={y.mean():.3f} "
          f"feats={len(FEATURE_COLS)} GPU={C.gpu_available()} SMOKE={SMOKE}")
    spatial = S.spatial_block_folds(df, k=K)
    random = S.group_random_folds(df, k=K)
    S.assert_no_group_leak(df, spatial); S.assert_no_group_leak(df, random)

    out = {"smoke": SMOKE, "seed": C.SEED, "k": int(K), "n_features": len(FEATURE_COLS),
           "dataset": "v2", "ablation": ABLATION or "none",
           "target": "T1a", "prevalence": float(y.mean()), "models": {}}
    all_curves = {}
    for name, mk in [("RF", make_rf), ("XGB", make_xgb)]:
        ts = time.time()
        sp, _, curves = run_scheme(df, y, spatial, mk, "spatial", name,
                                   capture_curves=(name == "XGB"))
        rd, _, _ = run_scheme(df, y, random, mk, "random", name)
        delta = {k: round(rd[k] - sp[k], 4) for k in
                 ["roc_auc", "f1", "accuracy", "recall", "precision"]}
        out["models"][name] = {"spatial": sp, "random": rd, "delta": delta,
                               "elapsed_s": round(time.time() - ts, 1)}
        if curves:
            all_curves[name] = curves
        print(f"[{name}] sp AUC={sp['roc_auc']:.3f} F1={sp['f1']:.3f} rec={sp['recall']:.3f} "
              f"prec={sp['precision']:.3f} | rd AUC={rd['roc_auc']:.3f} "
              f"| dAUC={delta['roc_auc']:+.3f}  ({time.time()-ts:.0f}s)")

    # training curves (XGB)
    if "XGB" in all_curves:
        (OUT / f"history_xgb{SUFFIX}.json").write_text(json.dumps(all_curves["XGB"], indent=2))
        plot_curves(all_curves["XGB"], "logloss", "loss_curves.png",
                    "T1a v2 — XGBoost logloss (train vs held-out spatial block)")
        plot_curves(all_curves["XGB"], "auc", "metric_curves.png",
                    "T1a v2 — XGBoost AUC (train vs held-out spatial block)")

    # SHAP
    print("[shap] computing global SHAP on full-data XGB ...")
    out["shap_top25"] = run_shap(df, y)

    out["wall_s"] = round(time.time() - t0, 1)
    (OUT / f"metrics_v2_t1{SUFFIX}.json").write_text(json.dumps(out, indent=2))
    print(f"\nDONE in {out['wall_s']:.0f}s -> {OUT/('metrics_v2_t1'+SUFFIX+'.json')}")
    print("[shap] top-8:", [t["feature"] for t in out["shap_top25"][:8]])


if __name__ == "__main__":
    main()
