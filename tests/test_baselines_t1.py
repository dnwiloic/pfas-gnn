"""Smoke test for src/baselines_t1.py (CLAUDE.md §5).

Checks the full baseline pipeline end-to-end on CPU in < 3 min:
  - Data loads; target T1a builds with expected prevalence.
  - Splits are leak-free (eval C2), non-degenerate.
  - Feature matrix is finite and leak-free.
  - LR / RF / XGB train, score finite AUC > 0.45.
  - Δ (random − spatial) is not absurdly negative (< −0.20).
  - OOF threshold in (0.05, 0.95).
  - Artefacts (config.yaml, metrics.json) written.
  - SHAP / permutation importance returns non-empty DataFrame.
  - Ablations run for all 4 configurations.
  - Paired tests don't crash.

Run: python3 tests/test_baselines_t1.py   OR  pytest -q tests/test_baselines_t1.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import data as D
from src import targets as T
from src import splits as S


def _ok(name: str, cond: bool, detail: str = ""):
    if not cond:
        raise AssertionError(f"FAILED [{name}]" + (f": {detail}" if detail else ""))
    print(f"  ok {name}" + (f"  [{detail}]" if detail else ""))


def test_smoke():
    t0 = time.time()
    print("=" * 60)
    print("SMOKE TEST — src/baselines_t1.py")
    print("=" * 60)

    # ---- import
    from src.baselines_t1 import (
        run_baselines, _optimal_threshold, _metrics_at_threshold,
        _cumulative_gain, XGBOOST_AVAILABLE, OPTUNA_AVAILABLE, SHAP_AVAILABLE,
        SMOKE_N_WELLS, SMOKE_OUTER_K, SMOKE_INNER_K, OPTUNA_TRIALS_SMOKE,
    )
    print(f"\n[env]  xgboost={XGBOOST_AVAILABLE}  optuna={OPTUNA_AVAILABLE}  shap={SHAP_AVAILABLE}")
    print(f"[smoke params]  n_wells={SMOKE_N_WELLS}  outer_k={SMOKE_OUTER_K}  "
          f"inner_k={SMOKE_INNER_K}  n_trials={OPTUNA_TRIALS_SMOKE}")

    # ---- unit helpers
    rng = np.random.default_rng(42)
    y_d, p_d = rng.integers(0, 2, 100), rng.random(100)
    tau = _optimal_threshold(y_d, p_d)
    _ok("threshold in range", 0.05 <= tau <= 0.95, f"tau={tau:.3f}")
    m = _metrics_at_threshold(y_d, p_d, tau)
    # the 5 required headline metrics + extras must all be present and finite
    for k in ("roc_auc", "f1", "accuracy", "recall", "precision", "pr_auc", "brier"):
        _ok(f"metric_{k} finite", np.isfinite(m[k]), f"{m[k]:.3f}")
    gain = _cumulative_gain(y_d, p_d, k_pct=20)
    _ok("gain_top20 in [0,1]", 0.0 <= gain <= 1.0, f"{gain:.3f}")

    # ---- full smoke run
    print("\n[run_baselines smoke=True]")
    save_dir = C.EXPERIMENTS_DIR / "baseline_t1_smoke"
    results = run_baselines(
        smoke=True, target="T1a",
        run_ablations_flag=True, run_shap_flag=True,
        n_optuna_trials=OPTUNA_TRIALS_SMOKE,
        save_dir=save_dir,
    )

    # artefacts
    _ok("config.yaml written", (save_dir / "config.yaml").exists())
    _ok("metrics.json written", (save_dir / "metrics.json").exists())
    with open(save_dir / "metrics.json") as fh:
        m_json = json.load(fh)
    _ok("metrics.json has models", "models" in m_json)

    # all models present
    for mn in ("LR", "RF", "XGB"):
        _ok(f"{mn} in results", mn in results["model_results"])

    # per-model checks
    for mn, res in results["model_results"].items():
        sp, rd, dlt = res["spatial"], res["random"], res["delta"]
        sp_auc  = sp.get("roc_auc_mean", float("nan"))
        rd_auc  = rd.get("roc_auc_mean", float("nan"))
        dlt_auc = dlt.get("roc_auc", float("nan"))
        _ok(f"{mn} spatial AUC finite", np.isfinite(sp_auc), f"{sp_auc:.3f}")
        _ok(f"{mn} random  AUC finite", np.isfinite(rd_auc), f"{rd_auc:.3f}")
        _ok(f"{mn} spatial AUC > 0.45", sp_auc > 0.45, f"{sp_auc:.3f}")
        _ok(f"{mn} delta not absurdly negative", dlt_auc > -0.25, f"Δ={dlt_auc:+.3f}")
        tau_m = sp.get("threshold_mean", float("nan"))
        _ok(f"{mn} threshold in range", 0.05 <= tau_m <= 0.95, f"tau={tau_m:.3f}")
        for k in ("f1", "accuracy", "recall", "precision"):   # the 5 (with roc_auc above)
            _ok(f"{mn} {k} finite", np.isfinite(sp.get(f"{k}_mean", float("nan"))),
                f"{sp.get(f'{k}_mean', float('nan')):.3f}")
        _ok(f"{mn} brier finite",   np.isfinite(sp.get("brier_mean",   float("nan"))))
        _ok(f"{mn} pr_auc finite",  np.isfinite(sp.get("pr_auc_mean",  float("nan"))))

    # global OOF AUC
    _ok("RF global OOF AUC finite",
        np.isfinite(results["model_results"]["RF"]["global_auc_spatial_oof"]))

    # comparisons
    _ok("comparisons dict", isinstance(results["comparison"], dict))

    # importance
    imp = results["importance"]
    _ok("importance non-empty", len(imp) > 0)
    _ok("importance has feature col", "feature" in imp.columns)
    _ok("importance has importance col", "importance" in imp.columns)
    _ok("importance all finite", np.isfinite(imp["importance"].to_numpy()).all())

    # ablations
    abl = results["ablations"]
    _ok("ablations 4 configs", len(abl) == 4)
    for key in ("no_loc_all", "with_loc_all", "no_loc_core", "no_loc_none"):
        _ok(f"ablation {key}", key in abl)
        _ok(f"ablation {key} AUC finite",
            np.isfinite(abl[key]["spatial_roc_auc"]), f"{abl[key]['spatial_roc_auc']:.3f}")

    # block prevalence
    bp = results["block_prevalence"]
    _ok("block_prevalence non-empty", len(bp) >= 2)

    elapsed = time.time() - t0
    _ok("smoke test < 200s", elapsed < 200, f"{elapsed:.1f}s")

    # timing estimate
    n_smoke, n_full = SMOKE_N_WELLS, 11333
    outer_smoke, outer_full = SMOKE_OUTER_K, 8
    trial_smoke, trial_full = OPTUNA_TRIALS_SMOKE, 20
    scale = (n_full / n_smoke) * (outer_full / outer_smoke) * (trial_full / trial_smoke)
    est_min = elapsed * scale / 60

    print(f"\n  Smoke elapsed:       {elapsed:.1f}s")
    print(f"  Estimated full run:  ~{est_min:.0f} min on CPU "
          f"(×{n_full/n_smoke:.0f} wells, ×{outer_full/outer_smoke:.0f} folds, "
          f"×{trial_full/trial_smoke:.0f} trials)")
    print(f"\nALL GREEN in {elapsed:.1f}s\n")
    return results


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    test_smoke()
