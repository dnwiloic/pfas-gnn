"""Smoke-test for the two Colab notebooks (CLAUDE.md §5).

Exercises the SAME code path that the notebooks use, but locally on CPU with
SMOKE_TEST=True. Does NOT invoke nbconvert (which would trip on Colab-specific
cells). Instead, reproduces the notebook orchestration inline, so every
Colab-specific block is under `if IN_COLAB:` and is inert locally.

Verifications:
  - src/ imports correctly (including FrequencyClassChain, REQUIRED metrics)
  - Dataset loads with correct shape (46338 x 201)
  - Integrity checks pass
  - T1 run_baselines(smoke=True) produces the 5 required metrics, artefacts written
  - T2 evaluate_model x 5 models produces the 5 metrics (micro+macro), per-label table,
    incremental checkpoint written
  - SMOTE ablation, pseudo-label probe, paired comparison all run without error
  - Both runs complete in < 200s each

Run:  python tests/test_colab_notebooks.py
       pytest -q tests/test_colab_notebooks.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ---- setup sys.path (same as notebook Cell 3) ----
REPO_LOCAL = Path(__file__).resolve().parents[1]
if str(REPO_LOCAL) not in sys.path:
    sys.path.insert(0, str(REPO_LOCAL))

DATA_PATH = REPO_LOCAL / "data" / "CA-PFAS-ASGWS.parquet"
SAVE_T1   = REPO_LOCAL / "experiments" / "baseline_t1_smoke"
SAVE_T2   = REPO_LOCAL / "experiments" / "baseline_t2_smoke_nb"


def _ok(name: str, cond: bool, detail: str = ""):
    if not cond:
        raise AssertionError(f"FAILED [{name}]" + (f": {detail}" if detail else ""))
    msg = f"  ok {name}" + (f"  [{detail}]" if detail else "")
    print(msg)


# =============================================================================
# 1. Guard-rail: same check as notebook Cell 3 (guard-rail)
# =============================================================================
def test_guardrail():
    print("\n[guard-rail] Testing import guard...")
    import src.baselines_t2
    import src.baselines_t1
    import src.metrics

    assert hasattr(src.baselines_t2, "FrequencyClassChain"), (
        "FrequencyClassChain not in src.baselines_t2 — code is obsolete."
    )
    from src.metrics import REQUIRED
    assert set(REQUIRED) == {"roc_auc", "f1", "accuracy", "recall", "precision"}, \
        f"REQUIRED mismatch: {REQUIRED}"
    _ok("FrequencyClassChain present", True)
    _ok("metrics.REQUIRED correct", True)


# =============================================================================
# 2. Dataset integrity (notebook Cell 4)
# =============================================================================
def test_dataset_integrity():
    print("\n[integrity] Checking dataset...")
    import pandas as pd

    _ok("parquet exists", DATA_PATH.exists(), str(DATA_PATH))
    df = pd.read_parquet(DATA_PATH)
    n_rows, n_cols = df.shape
    _ok("n_cols == 201", n_cols == 201, f"got {n_cols}")
    _ok("n_rows == 46338", n_rows == 46338, f"got {n_rows}")
    for col in ["gm_well_id", "latitude", "longitude", "PFOA_ngL"]:
        _ok(f"col {col} present", col in df.columns)
    return df


# =============================================================================
# 3. T1 notebook smoke (notebook Cells 5-12)
# =============================================================================
def test_t1_notebook(t1_timeout: float = 200.0):
    print("\n" + "="*60)
    print("NOTEBOOK SMOKE — baseline_t1_colab (T1)")
    print("="*60)

    # mirror notebook Cell 5
    SAVE_T1.mkdir(parents=True, exist_ok=True)

    # mirror notebook Cell 6
    import src.config as C
    C.DATA_PARQUET = DATA_PATH

    from src.baselines_t1 import run_baselines
    t0 = time.time()
    results = run_baselines(
        smoke=True,
        target="T1a",
        run_ablations_flag=True,
        run_shap_flag=True,
        save_dir=SAVE_T1,
    )
    elapsed = time.time() - t0
    _ok("T1 run elapsed < 200s", elapsed < t1_timeout, f"{elapsed:.1f}s")

    # mirror notebook Cell 7
    for mn, res in results["model_results"].items():
        sp = res["spatial"]
        for metric in ("roc_auc", "f1", "accuracy", "recall", "precision"):
            val = sp.get(f"{metric}_mean", float("nan"))
            _ok(f"T1/{mn}/{metric} finite", np.isfinite(val), f"{val:.3f}")
        _ok(f"T1/{mn}/AUC > 0.45", sp.get("roc_auc_mean", 0) > 0.45,
            f"{sp.get('roc_auc_mean', 0):.3f}")

    # mirror notebook Cell 8 — comparisons
    _ok("T1 comparisons dict non-empty", len(results["comparison"]) >= 1)

    # mirror notebook Cell 9 — importance
    imp = results["importance"]
    _ok("T1 importance non-empty", not imp.empty)
    _ok("T1 importance finite", np.isfinite(imp["importance"].to_numpy()).all())

    # mirror notebook Cell 10 — ablations
    abl = results["ablations"]
    _ok("T1 ablations 4 configs", len(abl) == 4)

    # mirror notebook Cell 12 — artefacts
    _ok("T1 config.yaml written", (SAVE_T1 / "config.yaml").exists())
    _ok("T1 metrics.json written", (SAVE_T1 / "metrics.json").exists())
    with open(SAVE_T1 / "metrics.json") as fh:
        m = json.load(fh)
    _ok("T1 metrics.json has models", "models" in m)

    # timing estimate
    from src.baselines_t1 import (SMOKE_N_WELLS, SMOKE_OUTER_K, OPTUNA_TRIALS_SMOKE,
                                   OUTER_SPATIAL_K, OPTUNA_TRIALS_FULL)
    scale = (11333 / SMOKE_N_WELLS) * (OUTER_SPATIAL_K / SMOKE_OUTER_K) * \
            (OPTUNA_TRIALS_FULL / OPTUNA_TRIALS_SMOKE)
    est_min = elapsed * scale / 60
    print(f"\n  T1 smoke elapsed:      {elapsed:.1f}s")
    print(f"  Estimated full run:    ~{est_min:.0f} min CPU "
          f"(x{11333/SMOKE_N_WELLS:.0f} wells, "
          f"x{OUTER_SPATIAL_K/SMOKE_OUTER_K:.0f} folds, "
          f"x{OPTUNA_TRIALS_FULL/OPTUNA_TRIALS_SMOKE:.0f} Optuna trials)")
    print(f"  On Colab GPU (n_jobs=-1): ~20-45 min expected.")

    return elapsed


# =============================================================================
# 4. T2 notebook smoke (notebook Cells 5-15)
# =============================================================================
def test_t2_notebook(t2_timeout: float = 200.0):
    print("\n" + "="*60)
    print("NOTEBOOK SMOKE — baseline_t2_colab (T2)")
    print("="*60)

    SAVE_T2.mkdir(parents=True, exist_ok=True)

    import src.config as C
    import src.data as D
    import src.splits as S
    import src.baselines_t2 as B

    C.DATA_PARQUET = DATA_PATH

    # mirror Cell 5
    SMOKE_TEST = True
    LABELS     = list(C.T2_LABELS)
    FEATURE_COLS = C.feature_columns(include_location=False, cocontam="core", include_air=False)
    K      = 2
    SMALL  = True
    SMOKE_N = 800
    MAX_ITER = 60

    # mirror Cell 6
    t0 = time.time()
    df = D.load(smoke=True, smoke_n=SMOKE_N)
    _ok("T2 data loaded", len(df) > 50, f"n={len(df)}")

    # leak check
    leak = set(FEATURE_COLS) & set(C.LEAKAGE_BLOCKLIST)
    _ok("T2 no feature leakage", not leak, str(leak))

    spatial = S.spatial_block_folds(df, k=K)
    random  = S.group_random_folds(df, k=K)
    S.assert_no_group_leak(df, spatial)
    S.assert_no_group_leak(df, random)

    Y, M = B.masked_targets(df, labels=LABELS)
    _ok("T2 Y shape", Y.shape == (len(df), len(LABELS)), str(Y.shape))
    _ok("T2 M shape", M.shape == (len(df), len(LABELS)), str(M.shape))

    # Patch max_iter
    _orig = B.make_estimator
    def _patched(kind="hgb", *, class_weight=None, small=False):
        est = _orig(kind, class_weight=class_weight, small=small)
        if hasattr(est, "max_iter") and not small:
            est.set_params(max_iter=MAX_ITER)
        return est
    B.make_estimator = _patched

    ORDER = tuple(LABELS)
    MODEL_SPECS = [
        ("Prevalence",       lambda: B.PrevalenceBaseline(labels=LABELS), False),
        ("BinaryRelevance",  lambda: B.BinaryRelevance(kind="hgb", labels=LABELS,
                                       class_weight="balanced", small=SMALL), False),
        ("Chain",            lambda: B.MaskedClassifierChain(kind="hgb", order=ORDER,
                                       out_labels=ORDER, class_weight="balanced",
                                       small=SMALL, inner_k=2), True),
        ("Ensemble",         lambda: B.EnsembleClassifierChains(kind="hgb", n_chains=2,
                                       labels=list(LABELS), class_weight="balanced",
                                       small=SMALL), True),
        ("FreqClassChain",   lambda: B.FrequencyClassChain(kind="hgb", labels=list(LABELS),
                                       n_classes=4, class_weight="balanced",
                                       small=SMALL, inner_k=2), True),
    ]

    REQ5_MICRO = ["micro_AUROC", "micro_F1", "micro_accuracy", "micro_recall", "micro_precision"]
    REQ5_MACRO = ["macro_AUROC", "macro_F1", "macro_accuracy", "macro_recall", "macro_precision"]

    results = {}
    ckpt_path = SAVE_T2 / "metrics_incremental.json"

    for nm, fac, use_groups in MODEL_SPECS:
        t1 = time.time()
        sp = B.evaluate_model(df, fac, spatial, FEATURE_COLS, labels=LABELS,
                              use_groups=use_groups)
        rd = B.evaluate_model(df, fac, random, FEATURE_COLS, labels=LABELS,
                              use_groups=use_groups)
        results[nm] = {"spatial": sp, "random": rd}

        a_sp = sp["aggregate"]
        # all 5 required metrics must be present and in [0,1]
        for k in REQ5_MICRO + REQ5_MACRO:
            val = a_sp.get(k, float("nan"))
            _ok(f"T2/{nm}/{k} in [0,1]", 0.0 <= val <= 1.0, f"{val:.3f}")
        # per-label table must have the 5 column metrics
        for c in ("f1", "precision", "recall", "accuracy"):
            _ok(f"T2/{nm}/per_label/{c}", c in sp["per_label"].columns)
        # OOF probabilities must be finite (on measured cells)
        oof = sp["oof_P"]
        measured_mask = ~np.isnan(oof)
        _ok(f"T2/{nm}/oof_P finite", np.isfinite(oof[measured_mask]).all())

        print(f"  [{nm}] AUROC sp={a_sp['macro_AUROC']:.3f}  "
              f"microF1={a_sp['micro_F1']:.3f}  ({time.time()-t1:.1f}s)")

        # checkpoint
        def _pack_lite(res):
            return {"aggregate": res["aggregate"],
                    "thresholds": res["thresholds"],
                    "per_label": res["per_label"].to_dict(orient="records"),
                    "per_fold": res["per_fold"].to_dict(orient="records"),
                    "spread": res["spread"]}
        ckpt = {nm_: {"spatial": _pack_lite(results[nm_]["spatial"]),
                      "random": _pack_lite(results[nm_]["random"])}
                for nm_ in results}
        ckpt_path.write_text(json.dumps(ckpt, indent=2, default=float))

    _ok("T2 checkpoint written", ckpt_path.exists())

    # learned models must beat prevalence floor
    floor = results["Prevalence"]["spatial"]["aggregate"]["macro_AUROC"]
    for nm in ["BinaryRelevance", "Chain"]:
        v = results[nm]["spatial"]["aggregate"]["macro_AUROC"]
        _ok(f"T2/{nm} > prevalence floor", v > floor + 0.01, f"{v:.3f} vs floor={floor:.3f}")

    # SMOTE ablation
    smote_res = B.evaluate_model(
        df, lambda: B.BinaryRelevance(kind="hgb", labels=LABELS, class_weight="balanced",
                                      smote_labels=("PFNA",), small=SMALL),
        spatial, FEATURE_COLS, labels=LABELS
    )
    pfna_cw_pl = results["BinaryRelevance"]["spatial"]["per_label"]
    pfna_cw = float(pfna_cw_pl.loc[pfna_cw_pl.label == "PFNA", "AUROC"].iloc[0])
    pfna_sm_pl = smote_res["per_label"]
    pfna_sm = float(pfna_sm_pl.loc[pfna_sm_pl.label == "PFNA", "AUROC"].iloc[0])
    _ok("T2 SMOTE ablation ran", np.isfinite(pfna_cw) and np.isfinite(pfna_sm))
    print(f"  PFNA: class_weight={pfna_cw:.3f}  +SMOTE={pfna_sm:.3f}  "
          f"delta={pfna_sm-pfna_cw:+.3f}")

    # paired comparison
    paired = {}
    for metric in ["macro_AUROC", "micro_F1"]:
        md, p = B.paired_compare(
            results["BinaryRelevance"]["spatial"]["per_fold"],
            results["Chain"]["spatial"]["per_fold"],
            metric=metric
        )
        paired[metric] = {"chain_minus_br": md, "wilcoxon_p": p}
        _ok(f"T2 paired {metric} ran", True)

    # pseudo-label probe
    probe = B.pseudo_label_probe(df, spatial, FEATURE_COLS,
                                 target_labels=("PFPeA", "PFPeS"), small=True)
    _ok("T2 pseudo probe ran", isinstance(probe, __import__("pandas").DataFrame))

    # Save final metrics.json + config.yaml (mirror Cell 14)
    import yaml
    cfg = {
        "task": "T2_multilabel_baseline",
        "seed": int(C.SEED),
        "smoke": True,
        "labels": list(LABELS),
        "n_feature_cols": len(FEATURE_COLS),
        "cv": {"spatial_k": int(K), "random_k": int(K)},
        "models": [nm for nm, _, _ in MODEL_SPECS],
        "hgb_max_iter": MAX_ITER,
    }
    (SAVE_T2 / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    metrics_out = {
        "smoke": True, "seed": int(C.SEED),
        "models": {nm: {"spatial": {"aggregate": results[nm]["spatial"]["aggregate"]},
                        "random":  {"aggregate": results[nm]["random"]["aggregate"]}}
                   for nm in results},
        "smote_ablation_PFNA": {"class_weight": pfna_cw, "smote": pfna_sm},
        "paired_br_vs_chain": paired,
    }
    (SAVE_T2 / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=float))
    _ok("T2 config.yaml written", (SAVE_T2 / "config.yaml").exists())
    _ok("T2 metrics.json written", (SAVE_T2 / "metrics.json").exists())

    # 4 frequency classes visible
    Yf, Mf = B.masked_targets(df, labels=LABELS)
    freq = {a: float(Yf[f"label_{a}"].to_numpy()[Mf[f"label_{a}"].to_numpy()].mean())
            if Mf[f"label_{a}"].any() else 0.0 for a in LABELS}
    order = sorted(LABELS, key=lambda a: -freq[a])
    classes = [list(g) for g in np.array_split(np.array(order, dtype=object), 4) if len(g)]
    _ok("T2 4 frequency classes", len(classes) == 4)
    for i, cl in enumerate(classes, 1):
        print(f"  FreqClassChain Class {i}: {'/'.join(cl)}")

    elapsed = time.time() - t0
    _ok("T2 run elapsed < 200s", elapsed < t2_timeout, f"{elapsed:.1f}s")

    # timing estimate
    n_full = 46338
    n_smoke = len(df)
    scale = (n_full / n_smoke) * (8 / K) * 3.5   # non-small models ~3.5x slower
    est_lo = elapsed * scale / 60
    est_hi = est_lo * 1.5
    print(f"\n  T2 smoke elapsed:      {elapsed:.1f}s")
    print(f"  Estimated full run:    ~{est_lo:.0f}-{est_hi:.0f} min CPU "
          f"(x{n_full/n_smoke:.1f} rows, x{8/K:.0f} folds, ~x3.5 model size)")
    print(f"  On Colab (High-RAM CPU): ~30-90 min expected.")

    return elapsed


# =============================================================================
# main
# =============================================================================
def main():
    t_global = time.time()
    print("=" * 60)
    print("SMOKE TEST — Colab notebooks (T1 + T2)")
    print("=" * 60)

    test_guardrail()
    test_dataset_integrity()
    t1_elapsed = test_t1_notebook()
    t2_elapsed = test_t2_notebook()

    total = time.time() - t_global
    print("\n" + "=" * 60)
    print("ALL GREEN")
    print("=" * 60)
    print(f"  T1 notebook smoke: {t1_elapsed:.1f}s")
    print(f"  T2 notebook smoke: {t2_elapsed:.1f}s")
    print(f"  Total:             {total:.1f}s ({total/60:.2f} min)")
    print()
    print("Artifacts written:")
    print(f"  T1: {SAVE_T1}")
    print(f"  T2: {SAVE_T2}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    main()
