"""Driver: GNN-hybrid T1a run — nested-OOF fusion (GraphSAGE embedding + XGBoost).

Two relations: "subbasin_knn" (mechanistic, cap 2 km) and "spatial" (bare k-NN, cap 1.5 km).
Both are run sequentially and their Δ(spatial, random) is reported.

SMOKE_TEST=True: CPU-only, ~500 wells, 3 outer folds, 2 inner folds, 15 GNN epochs.
  Expected wall time: < 3 min on modern CPU.

SMOKE_TEST=False: COLAB GPU ONLY. Do NOT run locally (CLAUDE.md §4/§5, memory note).
  Full run: K=8 outer × (J=4 inner + 1 test) × 2 relations = 80 GNN trainings.
  Estimated: ~80 × 10-20 min = 13-27 h on Colab T4 GPU (or ~8-13 h on A100).
  Use SMOKE_TEST=True to verify the pipeline, then launch on Colab.

Usage:
    # Smoke (CPU):
    cd /path/to/pfas-gnn && PFAS_FORCE_CPU=1 python experiments/gnn_hybrid_t1/run_hybrid_t1.py

    # Full (Colab GPU — from the Colab notebook):
    SMOKE_TEST = False
    python experiments/gnn_hybrid_t1/run_hybrid_t1.py

Outputs (in experiments/gnn_hybrid_t1/):
    config.yaml                      — full run configuration
    metrics.json                     — final metrics (spatial + random + comparison)
    spatial/metrics_incremental.json — checkpointed per-fold (spatial arm)
    random/metrics_incremental.json  — checkpointed per-fold (random arm, subbasin_knn)
    spatial_spatial/...              — spatial arm, bare-spatial graph
    random_spatial/...               — random arm, bare-spatial graph
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# --- path setup (works both as a script and when the package is installed)
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("PFAS_FORCE_CPU", "1")   # safety: override only if not already set

import numpy as np
import pandas as pd

from src import config as C
from src import data as D
from src import splits as S
from src import targets as T
from src import hybrid as H
from src import gnn
from src import baselines_t1 as BL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("run_hybrid_t1")

# ------------------------------------------------------------------- toggle
SMOKE_TEST = True          # flip to False for Colab GPU full run

# ------------------------------------------------------------------- params
SEED      = C.SEED
OUT_DIR   = _HERE

RELATIONS = ["subbasin_knn", "spatial"]   # both eval-approved (§C.3)

# smoke params
SMOKE_N_WELLS     = 500
SMOKE_OUTER_K     = 3
SMOKE_INNER_K     = 2
SMOKE_GNN_EPOCHS  = 15
SMOKE_GNN_PATIENCE= 6
SMOKE_HIDDEN      = 32

# full params (Colab GPU only)
FULL_OUTER_K      = 8
FULL_INNER_K      = 4
FULL_GNN_EPOCHS   = 400
FULL_GNN_PATIENCE = 50
FULL_HIDDEN       = 64

# XGB-alone wall (from baselines_t1 spatial CV, stored here for reference).
# These numbers come from experiments/baseline_t1_smoke/metrics.json (smoke).
# The full-run wall values (to update after Colab run): RF~0.601 / XGB~0.588.
XGB_WALL_SPATIAL_AUC_MEAN = 0.588   # from prior full-run (experiments/profilage context)
XGB_WALL_SPATIAL_AUCS     = []      # will be filled if baselines are rerun

# ------------------------------------------------------------------- GNN-alone reference
# GNN-alone spatial AUC from gnn_phase2 runs (best subbasin_knn, spatial CV).
# Values from experiments/gnn_phase2/ (to be updated post Colab run).
GNN_ALONE_SPATIAL_AUC = 0.605   # approximate from phase 2 logs
GNN_ALONE_SPATIAL_AUCS = []     # per-fold list (to fill)

# ===================================================================== main

def _estimate_full_cost(outer_k: int, inner_k: int, n_relations: int,
                        gpu_min_per_gnn: float = 15.0) -> str:
    """Estimate the number of GNN trainings and wall-clock cost."""
    # Per relation, spatial arm: K*(J+1) GNN trainings
    # Plus same for random arm: K*(J+1) GNN trainings
    n_per_relation = 2 * outer_k * (inner_k + 1)
    n_total = n_per_relation * n_relations
    hours = n_total * gpu_min_per_gnn / 60
    return (f"{n_total} GNN trainings ({n_relations} relations × 2 arms × "
            f"{outer_k} outer × ({inner_k} inner + 1 test)) = "
            f"~{hours:.0f} h on Colab GPU @ {gpu_min_per_gnn} min/GNN")


def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info(f"run_hybrid_t1  SMOKE_TEST={SMOKE_TEST}  seed={SEED}")
    logger.info("=" * 70)

    if not SMOKE_TEST:
        logger.warning("FULL RUN — intended for Colab GPU only. Running locally may take >13 h.")

    # Parameters
    outer_k  = SMOKE_OUTER_K   if SMOKE_TEST else FULL_OUTER_K
    inner_k  = SMOKE_INNER_K   if SMOKE_TEST else FULL_INNER_K
    epochs   = SMOKE_GNN_EPOCHS if SMOKE_TEST else FULL_GNN_EPOCHS
    patience = SMOKE_GNN_PATIENCE if SMOKE_TEST else FULL_GNN_PATIENCE
    hidden   = SMOKE_HIDDEN    if SMOKE_TEST else FULL_HIDDEN

    cost_str = _estimate_full_cost(FULL_OUTER_K, FULL_INNER_K, len(RELATIONS))
    logger.info(f"Full-run cost estimate: {cost_str}")
    logger.info(f"  Smoke params: outer_k={outer_k} inner_k={inner_k} "
                f"epochs={epochs} patience={patience} hidden={hidden}")

    # Load data
    df = D.load(smoke=SMOKE_TEST, smoke_n=SMOKE_N_WELLS if SMOKE_TEST else 99999)
    y_row = T.build_T1a(df).to_numpy()
    prevalence = float(y_row.mean())
    feature_cols = C.feature_columns(include_location=False, cocontam="core")
    logger.info(f"Data: {df.shape}  wells={df[C.WELL_ID].nunique()}  "
                f"prevalence={prevalence:.3f}  features={len(feature_cols)}")

    # Run hybrid for each relation
    all_results = {}
    for relation in RELATIONS:
        logger.info(f"\n{'='*60}")
        logger.info(f"RELATION: {relation}")
        logger.info(f"{'='*60}")
        rel_dir = OUT_DIR / f"run_{relation}"

        res = H.run_hybrid_t1(
            df, smoke=SMOKE_TEST, relation=relation, hidden=hidden,
            inner_k=inner_k, gnn_max_epochs=epochs, gnn_patience=patience,
            outer_k=outer_k, seed=SEED, verbose=SMOKE_TEST,
            out_dir=rel_dir,
        )
        all_results[relation] = res

        sp_auc = res["hybrid_spatial"]["aggregated"].get("roc_auc_mean", float("nan"))
        rd_auc = res["hybrid_random"]["aggregated"].get("roc_auc_mean", float("nan"))
        delta  = rd_auc - sp_auc
        logger.info(f"\n[{relation}] hybrid spatial AUC={sp_auc:.4f}  "
                    f"random AUC={rd_auc:.4f}  Δ(random−spatial)={delta:+.4f}")

    # Three-way comparison (for the primary relation: subbasin_knn)
    primary = "subbasin_knn"
    spatial_cv = all_results[primary]["hybrid_spatial"]
    random_cv  = all_results[primary]["hybrid_random"]

    gnn_spatial_stub = {
        "auc_mean": GNN_ALONE_SPATIAL_AUC,
        "per_fold_auc": GNN_ALONE_SPATIAL_AUCS,
    }
    gnn_random_stub = {"auc_mean": float("nan")}

    comparison = H.run_three_way_comparison(
        spatial_cv, random_cv,
        gnn_spatial_stub, gnn_random_stub,
        xgb_spatial_auc_mean=XGB_WALL_SPATIAL_AUC_MEAN,
        xgb_spatial_aucs=XGB_WALL_SPATIAL_AUCS,
        noise_threshold=0.03,
    )

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info("THREE-WAY COMPARISON SUMMARY")
    logger.info("=" * 70)
    for arm, tri in comparison["triplets"].items():
        rd_str  = f"{tri['random']:.4f}"  if np.isfinite(tri['random'])  else "n/a"
        dlt_str = f"{tri['delta']:+.4f}" if np.isfinite(tri['delta'])  else "n/a"
        logger.info(f"  {arm:<12}  spatial={tri['spatial']:.4f}  "
                    f"random={rd_str:>8}  delta={dlt_str:>8}")
    rv = comparison["reality_rule"]
    logger.info(f"\n  Hybrid gain over XGB wall: {rv['hybrid_gain_over_xgb_wall']:+.4f}")
    logger.info(f"  Significant: {rv['significant']}  Above noise ({rv['noise_threshold']}): "
                f"{rv['above_noise_threshold']}")
    logger.info(f"  Verdict: {rv['verdict']}")

    # Write artefacts
    elapsed = time.time() - t_start
    cfg_out = {
        "smoke": SMOKE_TEST,
        "outer_k": outer_k, "inner_k": inner_k,
        "gnn_epochs": epochs, "gnn_patience": patience, "hidden": hidden,
        "relations": RELATIONS,
        "feature_cols_count": len(feature_cols),
        "prevalence": float(prevalence),
        "seed": SEED,
        "xgboost_available": H.XGBOOST_AVAILABLE,
        "elapsed_s": elapsed,
        "cost_estimate": cost_str,
    }

    with open(OUT_DIR / "config.yaml", "w") as fh:
        for k, v in cfg_out.items():
            fh.write(f"{k}: {v}\n")

    metrics_out = {
        "config": cfg_out,
        "per_relation": {
            rel: {
                "spatial": {
                    "aggregated": res["hybrid_spatial"]["aggregated"],
                    "global_oof_auc": res["hybrid_spatial"]["global_oof_auc"],
                    "bootstrap_ci": res["hybrid_spatial"]["bootstrap_ci_by_well"],
                    "per_fold": res["hybrid_spatial"]["per_fold"],
                },
                "random": {
                    "aggregated": res["hybrid_random"]["aggregated"],
                    "global_oof_auc": res["hybrid_random"]["global_oof_auc"],
                },
                "delta_auc": (
                    res["hybrid_random"]["aggregated"].get("roc_auc_mean", float("nan"))
                    - res["hybrid_spatial"]["aggregated"].get("roc_auc_mean", float("nan"))
                ),
                "elapsed_s": res["elapsed_s"],
            }
            for rel, res in all_results.items()
        },
        "three_way_comparison": comparison,
    }

    with open(OUT_DIR / "metrics.json", "w") as fh:
        json.dump(metrics_out, fh, indent=2, default=str)

    logger.info(f"\nArtefacts -> {OUT_DIR}")
    logger.info(f"Total elapsed: {elapsed:.1f}s")
    logger.info("=" * 70)
    if SMOKE_TEST:
        n_full_wells = 11333
        est_factor = (n_full_wells / SMOKE_N_WELLS) * (FULL_GNN_EPOCHS / epochs)
        logger.info(f"Estimated full-run time (Colab GPU): {cost_str}")
        logger.info("SMOKE_TEST DONE — flip SMOKE_TEST=False for Colab GPU full run.")
    return metrics_out


if __name__ == "__main__":
    main()
