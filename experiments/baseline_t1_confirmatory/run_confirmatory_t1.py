"""XGBoost T1 baseline — CONFIRMATORY mode (diagnostic ceiling, NOT deployable).

Adds the raw PFAS *_ngL concentration columns back into the feature set.
The *_detected booleans STAY EXCLUDED: they are near-direct label components
(PFOA_detected AND PFOA_ngL>4 => T1a=1, so detected is a ternary of the target).

IMPORTANT: T1a is DERIVED from PFAS concentrations:
    T1a = (PFOA_ngL > 4) | (PFOS_ngL > 4) | (HazardIndex >= 1)
where HI uses PFHxS_ngL, PFNA_ngL, HFPO_DA_ngL, PFBS_ngL.
Including these as features makes the model partially tautological — it can
"learn" the label definition exactly.  This is a CEILING / SANITY CHECK, never
a deployable model.  It answers: "what is the maximum discriminative information
available in the chemistry?".

Feature set:
    CONFIRMATORY = predictive_features (96 cols) + NGL_COLS (31 *_ngL) = 127 cols
    DETECTED_COLS stay in the blocklist (near-direct label, near-tautological).
    LABEL_COLS stay in the blocklist (direct derivations of the target).
    DERIVED_TARGET_COLS (sum_pfas_ngL, target_sum_gt70, pfas_class_assignment) stay
    excluded for the same reason.

Protocol identical to run_spatial_t1.py (the frozen socle):
    - Group by gm_well_id (no pseudo-replicates)
    - k=8 KMeans spatial block CV (reference) + group random CV (for Δ)
    - OOF F1-optimal threshold (never test)
    - Seed 42

Run:
    python experiments/baseline_t1_confirmatory/run_confirmatory_t1.py          # full
    SMOKE_TEST=1 python experiments/baseline_t1_confirmatory/run_confirmatory_t1.py
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

# ------------------------------------------------------------------
# CONFIRMATORY feature set: predictive context features + raw *_ngL
# The FeaturePipeline hard-blocks the LEAKAGE_BLOCKLIST which includes NGL_COLS.
# To unlock NGL_COLS in confirmatory mode we build a lighter pipeline that only
# blocks DETECTED_COLS + LABEL_COLS + DERIVED_TARGET_COLS (not NGL_COLS).
# We extend feature_columns() with NGL_COLS (31 raw concentration columns).
# ------------------------------------------------------------------

PREDICTIVE_COLS = C.feature_columns(include_location=False, cocontam="all", include_air=True)
CONFIRMATORY_COLS = PREDICTIVE_COLS + C.NGL_COLS   # 96 + 31 = 127 features

# The partial blocklist for the confirmatory pipeline (NGL_COLS are allowed).
CONFIRMATORY_BLOCKLIST = C.DETECTED_COLS + C.LABEL_COLS + C.DERIVED_TARGET_COLS
assert set(C.NGL_COLS).isdisjoint(set(CONFIRMATORY_BLOCKLIST)), \
    "NGL_COLS must not overlap the confirmatory blocklist"
assert len(CONFIRMATORY_BLOCKLIST) == 31 + 31 + 3  # 65


class ConfirmatoryFeaturePipeline(F.FeaturePipeline):
    """Extends FeaturePipeline with a relaxed blocklist (NGL_COLS allowed).

    In predictive mode the hard guard in FeaturePipeline.__init__ would reject
    any *_ngL column.  Here we override the check to use CONFIRMATORY_BLOCKLIST
    (excludes *_detected and *_label but allows raw concentrations).

    All numeric/log/encoding logic is inherited unchanged.
    """

    def __init__(self, feature_cols, encode: str = "target",
                 missing_indicator_thresh: float = 0.20, seed: int = C.SEED):
        # Override the blocklist check with the confirmatory partial blocklist.
        leak = set(feature_cols) & set(CONFIRMATORY_BLOCKLIST)
        if leak:
            raise ValueError(
                f"blocklisted columns in confirmatory features: {sorted(leak)}")
        # Skip the parent __init__ guard; re-implement the rest identically.
        self.feature_cols = list(feature_cols)
        self.encode, self.thresh, self.seed = encode, missing_indicator_thresh, seed
        self.cat_low = [c for c in self.feature_cols if c in C.CATEGORICAL_LOW_CARD]
        self.cat_high = [c for c in self.feature_cols if c in C.CATEGORICAL_HIGH_CARD]
        self.num_cols = [c for c in self.feature_cols
                         if c not in self.cat_low + self.cat_high]
        # NGL_COLS are non-negative concentrations -> apply log1p
        self.log_cols = [c for c in self.num_cols
                         if c in C.LOG1P_FEATS or c in C.NGL_COLS]


def make_rf_smoke():
    return RandomForestClassifier(
        n_estimators=50, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=C.SEED)


def make_rf():
    return (make_rf_smoke() if SMOKE else
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5,
                max_features="sqrt", n_jobs=-1, random_state=C.SEED))


def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=120 if SMOKE else 400, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", verbosity=0, random_state=C.SEED,
        **C.xgb_device_params())


def _proba(m, X):
    return m.predict_proba(X)[:, 1]


def oof_threshold(df_tr, y_tr, make_model):
    """F1-optimal threshold from inner spatial CV on TRAIN only (no test)."""
    fold = S.spatial_block_folds(df_tr, k=INNER_K)
    oof = np.full(len(y_tr), np.nan)
    for _, tr, va in S.iter_folds(fold):
        if tr.sum() < 20 or va.sum() < 5 or len(np.unique(y_tr[tr])) < 2:
            continue
        pipe = ConfirmatoryFeaturePipeline(CONFIRMATORY_COLS, encode="target")
        Xtr, _ = pipe.fit_transform(df_tr[tr], y_tr[tr])
        Xva, _ = pipe.transform(df_tr[va])
        m = make_model()
        m.fit(Xtr, y_tr[tr])
        oof[va] = _proba(m, Xva)
    oof = np.where(np.isnan(oof),
                   np.nanmean(oof) if np.isfinite(np.nanmean(oof)) else 0.5,
                   oof)
    grid = np.linspace(0.1, 0.9, 33)
    f1s = [f1_score(y_tr, (oof >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def run_scheme(df, y, fold, make_model, scheme, name):
    rows = []
    folds_list = list(S.iter_folds(fold))
    dev = "GPU" if (name == "XGB" and C.gpu_available()) else "CPU"
    for f, tr, te in P.track(folds_list, total=len(folds_list),
                              desc=f"{name}/{scheme} [{dev}]"):
        if len(np.unique(y[te])) < 2 or te.sum() < 10:
            continue
        df_tr = df[tr].reset_index(drop=True)
        df_te = df[te].reset_index(drop=True)
        ytr, yte = y[tr], y[te]
        tau = oof_threshold(df_tr, ytr, make_model)
        pipe = ConfirmatoryFeaturePipeline(CONFIRMATORY_COLS, encode="target")
        Xtr, _ = pipe.fit_transform(df_tr, ytr)
        Xte, _ = pipe.transform(df_te)
        m = make_model()
        m.fit(Xtr, ytr)
        rows.append(MM.binary_metrics(yte, _proba(m, Xte), tau))

    keys = ["roc_auc", "f1", "accuracy", "recall", "precision",
            "pr_auc", "balanced_accuracy", "brier"]
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    agg.update({f"{k}_std": float(np.nanstd([r[k] for r in rows]))
                for k in ["roc_auc", "f1", "accuracy", "recall", "precision"]})
    agg["n_folds"] = len(rows)
    return agg, rows


def main():
    t0 = time.time()

    print("=" * 70)
    print("CONFIRMATORY MODE — T1a XGBoost baseline")
    print("Feature set: predictive_context (96) + NGL_COLS (31) = 127 cols")
    print("WARNING: T1a is derived from PFAS concentrations -> PARTIALLY TAUTOLOGICAL")
    print("This is a CEILING number, NOT a deployable model.")
    print("=" * 70)

    # Verify no detected / label cols slipped into feature set
    assert set(CONFIRMATORY_COLS).isdisjoint(set(C.DETECTED_COLS)), "detected cols leaked"
    assert set(CONFIRMATORY_COLS).isdisjoint(set(C.LABEL_COLS)), "label cols leaked"
    assert set(CONFIRMATORY_COLS).isdisjoint(set(C.DERIVED_TARGET_COLS)), "derived cols leaked"
    # Verify NGL_COLS are present
    assert all(c in CONFIRMATORY_COLS for c in C.NGL_COLS), "some NGL_COLS missing"

    smoke_n = 400 if SMOKE else None
    df = D.load(smoke=SMOKE, smoke_n=smoke_n or 600)
    y = T.build_T1a(df).to_numpy()

    print(f"[data] rows={len(df)} wells={df[C.WELL_ID].nunique()} "
          f"prev={y.mean():.3f} feats={len(CONFIRMATORY_COLS)} "
          f"GPU={C.gpu_available()} SMOKE={SMOKE}")

    # Verify all feature columns exist in the loaded dataframe.
    # TEMPORAL_DERIVED (year, month_sin, month_cos) are created inside the
    # FeaturePipeline (features.add_temporal), not present in the raw df.
    missing_cols = [c for c in CONFIRMATORY_COLS
                    if c not in df.columns and c not in C.TEMPORAL_DERIVED]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols[:10]} ...")
    print(f"[columns] all {len(CONFIRMATORY_COLS)} confirmatory columns present in data")

    spatial = S.spatial_block_folds(df, k=K)
    random = S.group_random_folds(df, k=K)
    S.assert_no_group_leak(df, spatial)
    S.assert_no_group_leak(df, random)

    models = []
    if HAS_XGB:
        models.append(("XGB", make_xgb))
    else:
        print("WARNING: xgboost not available, running RF only")
    # Also run RF for comparison (optional but informative)
    models.append(("RF", make_rf))

    out = {
        "smoke": SMOKE,
        "seed": C.SEED,
        "k": int(K),
        "n_predictive_features": len(PREDICTIVE_COLS),
        "n_confirmatory_features": len(CONFIRMATORY_COLS),
        "n_ngl_added": len(C.NGL_COLS),
        "target": "T1a",
        "prevalence": float(y.mean()),
        "mode": "confirmatory",
        "caveat": (
            "TAUTOLOGICAL CEILING: T1a = (PFOA_ngL>4) | (PFOS_ngL>4) | (HI>=1). "
            "The NGL_COLS for PFOA, PFOS, PFHxS, PFNA, HFPO_DA, PFBS are direct "
            "components of the label definition. Including them makes the model "
            "partially tautological. Do NOT use these scores to claim a deployable model. "
            "*_detected columns remain excluded (near-direct binary label components)."
        ),
        "models": {},
    }

    for name, mk in models:
        ts = time.time()
        sp, sp_rows = run_scheme(df, y, spatial, mk, "spatial", name)
        rd, rd_rows = run_scheme(df, y, random, mk, "random", name)
        delta = {k: round(rd[k] - sp[k], 4)
                 for k in ["roc_auc", "f1", "accuracy", "recall", "precision"]}
        out["models"][name] = {
            "spatial": sp,
            "random": rd,
            "delta_random_minus_spatial": delta,
            "elapsed_s": round(time.time() - ts, 1),
        }
        print(
            f"[{name}] sp AUC={sp['roc_auc']:.3f} F1={sp['f1']:.3f} "
            f"acc={sp['accuracy']:.3f} rec={sp['recall']:.3f} prec={sp['precision']:.3f} "
            f"pr_auc={sp.get('pr_auc', float('nan')):.3f} brier={sp.get('brier', float('nan')):.3f} "
            f"| rd AUC={rd['roc_auc']:.3f} | dAUC={delta['roc_auc']:+.3f}  "
            f"({time.time() - ts:.0f}s)"
        )

    out["wall_s"] = round(time.time() - t0, 1)
    out_path = OUT / "metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nDONE in {out['wall_s']:.0f}s  ->  {out_path}")


if __name__ == "__main__":
    main()
