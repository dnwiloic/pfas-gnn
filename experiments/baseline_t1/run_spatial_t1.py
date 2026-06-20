"""Lean SPATIAL reference run for T1a — our strict protocol, NO Optuna (fast on CPU).

Gives the honest spatial-CV number that the prior random-split baseline (~0.97 AUC)
cannot: CV spatiale par blocs (reference) + CV aleatoire groupee (Delta), grouped by
gm_well_id, our blocklist (gm_dataset_name excluded, C6), detection guard on T1a (C1),
OOF threshold (no test leak). Reuses the smoke-tested socle. RF + XGBoost, 5 metrics.

    python experiments/baseline_t1/run_spatial_t1.py            # full (k=8)
    SMOKE_TEST=1 python experiments/baseline_t1/run_spatial_t1.py   # quick check
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
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

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
K = 2 if SMOKE else C.N_SPATIAL_BLOCKS
INNER_K = 2 if SMOKE else 3
OUT = Path(__file__).resolve().parent
FEATURE_COLS = C.feature_columns(include_location=False, cocontam="all", include_air=True)


def make_rf():
    return RandomForestClassifier(
        n_estimators=120 if SMOKE else 300, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=C.SEED)


def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=120 if SMOKE else 400, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", verbosity=0, random_state=C.SEED,
        **C.xgb_device_params())                       # device=cuda if GPU else CPU hist


def _proba(m, X):
    return m.predict_proba(X)[:, 1]


def oof_threshold(df_tr, y_tr, make_model):
    """F1-optimal threshold from a single grouped+spatial inner CV on TRAIN (no test)."""
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


def run_scheme(df, y, fold, make_model, scheme, name):
    rows = []
    folds = list(S.iter_folds(fold))
    dev = "GPU" if (name == "XGB" and C.gpu_available()) else "CPU"
    for f, tr, te in P.track(folds, total=len(folds), desc=f"{name}/{scheme} [{dev}]"):
        if len(np.unique(y[te])) < 2 or te.sum() < 10:
            continue
        df_tr, df_te = df[tr].reset_index(drop=True), df[te].reset_index(drop=True)
        ytr, yte = y[tr], y[te]
        tau = oof_threshold(df_tr, ytr, make_model)
        pipe = F.FeaturePipeline(FEATURE_COLS, encode="target")
        Xtr, _ = pipe.fit_transform(df_tr, ytr)
        Xte, _ = pipe.transform(df_te)
        m = make_model(); m.fit(Xtr, ytr)
        rows.append(MM.binary_metrics(yte, _proba(m, Xte), tau))
    keys = ["roc_auc", "f1", "accuracy", "recall", "precision", "pr_auc", "balanced_accuracy"]
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    agg.update({f"{k}_std": float(np.nanstd([r[k] for r in rows]))
                for k in ["roc_auc", "f1", "accuracy", "recall", "precision"]})
    agg["n_folds"] = len(rows)
    return agg, rows


def main():
    t0 = time.time()
    df = D.load(smoke=SMOKE, smoke_n=800)
    y = T.build_T1a(df).to_numpy()
    print(f"[data] rows={len(df)} wells={df[C.WELL_ID].nunique()} prev={y.mean():.3f} "
          f"feats={len(FEATURE_COLS)} GPU={C.gpu_available()} SMOKE={SMOKE}")
    spatial = S.spatial_block_folds(df, k=K)
    random = S.group_random_folds(df, k=K)
    S.assert_no_group_leak(df, spatial); S.assert_no_group_leak(df, random)

    models = [("RF", make_rf)]
    if HAS_XGB:
        models.append(("XGB", make_xgb))

    out = {"smoke": SMOKE, "seed": C.SEED, "k": int(K), "n_features": len(FEATURE_COLS),
           "target": "T1a", "prevalence": float(y.mean()), "models": {}}
    for name, mk in models:
        ts = time.time()
        sp, _ = run_scheme(df, y, spatial, mk, "spatial", name)
        rd, _ = run_scheme(df, y, random, mk, "random", name)
        delta = {k: round(rd[k] - sp[k], 4) for k in
                 ["roc_auc", "f1", "accuracy", "recall", "precision"]}
        out["models"][name] = {"spatial": sp, "random": rd, "delta": delta,
                               "elapsed_s": round(time.time() - ts, 1)}
        print(f"[{name}] sp AUC={sp['roc_auc']:.3f} F1={sp['f1']:.3f} acc={sp['accuracy']:.3f} "
              f"rec={sp['recall']:.3f} prec={sp['precision']:.3f} | rd AUC={rd['roc_auc']:.3f} "
              f"| dAUC={delta['roc_auc']:+.3f}  ({time.time()-ts:.0f}s)")

    out["wall_s"] = round(time.time() - t0, 1)
    (OUT / "metrics_spatial.json").write_text(json.dumps(out, indent=2))
    print(f"\nDONE in {out['wall_s']:.0f}s -> {OUT/'metrics_spatial.json'}")


if __name__ == "__main__":
    main()
