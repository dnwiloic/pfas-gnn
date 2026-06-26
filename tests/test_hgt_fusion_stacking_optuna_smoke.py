"""Smoke test for src/hgt_fusion_stacking_t1_optuna.py.

Runs all 4 Optuna studies in smoke mode (500 wells, 3 blocks, 3 trials each),
verifies end-to-end: studies run, best params present, AUC finite, JSON written.
Executes on CPU in < ~3 min.

Run:
    SMOKE_TEST=1 PFAS_FORCE_CPU=1 python -m pytest tests/test_hgt_fusion_stacking_optuna_smoke.py -v
or:
    SMOKE_TEST=1 PFAS_FORCE_CPU=1 python tests/test_hgt_fusion_stacking_optuna_smoke.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# Ensure src/ is on path regardless of invocation style
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("PFAS_FORCE_CPU", "1")


def test_run_all_studies_smoke():
    """End-to-end smoke test: all 4 studies, 500 wells, 3 blocks, 3 trials each."""
    from src import hgt_fusion_stacking_t1_optuna as OPT
    from src import config as C

    # Build feature_cols as in run_v2.py (pure-mechanism, no admin geo)
    feature_cols = [c for c in C.feature_columns(include_location=False,
                                                   cocontam="all", include_air=True)
                    if c not in C.ADMIN_GEO_CAT]

    exp_dir = REPO / "experiments" / "hgt_fusion_stacking_t1_v2" / "_smoke_optuna_test"
    exp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = OPT.run_all_studies(
        smoke=True,
        feature_cols=feature_cols,
        n_blocks=3,
        n_trials_hgt=3,
        n_trials_xgb=3,
        n_trials_rf=3,
        n_trials_stack=3,
        seed=42,
        exp_dir=str(exp_dir),
        write=True,
        verbose=True,
    )
    elapsed = time.time() - t0

    # ---- assertions ----
    assert "studies" in results, "results must have 'studies' key"
    for name in ["hgt_standalone", "xgb_tabular", "rf_tabular", "stacking_meta"]:
        assert name in results["studies"], f"study '{name}' missing from results"
        s = results["studies"][name]
        assert "best_params" in s, f"'{name}' missing best_params"
        assert "best_value" in s, f"'{name}' missing best_value"
        assert isinstance(s["best_params"], dict), f"'{name}' best_params must be dict"
        assert math.isfinite(s["best_value"]), \
            f"'{name}' best_value not finite: {s['best_value']}"
        assert s["best_value"] > 0.4, \
            f"'{name}' best_value suspiciously low: {s['best_value']:.4f}"
        assert s["n_trials"] >= 1, f"'{name}' n_trials must be >= 1"
        print(f"  {name}: best_value={s['best_value']:.4f} "
              f"params={s['best_params']} n_trials={s['n_trials']}")

    # JSON artefact written
    out_path = exp_dir / "optuna_best_params.json"
    assert out_path.exists(), f"optuna_best_params.json not written to {exp_dir}"
    loaded = json.loads(out_path.read_text())
    assert "studies" in loaded, "JSON missing 'studies' key"
    assert loaded["meta"]["smoke"] is True

    print(f"\nSmoke test elapsed: {elapsed:.1f}s ({elapsed/60:.2f} min)")
    assert elapsed < 300, (
        f"Smoke test took {elapsed:.0f}s > 300s (5 min limit). "
        "Check that PFAS_FORCE_CPU=1 is set and smoke mode is active."
    )
    print("SMOKE TEST PASSED.")


if __name__ == "__main__":
    test_run_all_studies_smoke()
