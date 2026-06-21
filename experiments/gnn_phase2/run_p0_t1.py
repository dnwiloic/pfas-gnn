"""P0 driver — T1a GraphSAGE/GCN with ROBUST early-stopping (multi-micro-block validation
+ ReduceLROnPlateau + more patience). Same protocol/metrics as the phase-1 WALL comparison;
writes experiments/gnn_phase2/metrics_p0.json (triplet + per-fold + best-epoch diagnostics).

Reuses the frozen socle and the extended src/gnn.py (run_t1_cv with n_val_micro / lr_schedule).
Toggle SMOKE_TEST for a fast CPU sanity run.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config as C
from src import data, gnn

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
OUT = Path(__file__).resolve().parent
MODELS = ["graphsage", "gcn"]
CAP_KM, K_NN = 1.5, 8


def main():
    t0 = time.time()
    df = data.load(smoke=SMOKE_TEST, smoke_n=1500)
    common = dict(k=K_NN, cap_km=CAP_KM, cut_blocks=True, encode="frequency",
                  hidden=(32 if SMOKE_TEST else 64), layers=2, dropout=0.5,
                  lr=5e-3, n_val_micro=(4 if SMOKE_TEST else 8), val_frac=0.18,
                  lr_schedule=True,
                  max_epochs=(50 if SMOKE_TEST else 400),
                  patience=(20 if SMOKE_TEST else 50))
    nb = 3 if SMOKE_TEST else 8

    results = {}
    for model in MODELS:
        sp, _ = gnn.run_t1_cv(df, model_name=model, regime="spatial", n_blocks=nb, **common)
        rd, _ = gnn.run_t1_cv(df, model_name=model, regime="random", n_blocks=nb, **common)
        results[model] = {
            "auc_spatial": sp["auc_mean"], "auc_spatial_std": sp["auc_std"],
            "auc_random": rd["auc_mean"], "auc_random_std": rd["auc_std"],
            "delta_random_minus_spatial": rd["auc_mean"] - sp["auc_mean"],
            "per_fold_spatial": sp["per_fold_auc"],
            "per_fold_random": rd["per_fold_auc"],
            "per_fold_best_epoch_spatial": sp["per_fold_best_epoch"],
            "per_fold_n_val_micro_spatial": sp["per_fold_n_val_micro"],
            "total_removed_cross_block": sp["total_removed_cross_block"],
        }
        # incremental checkpoint after each model
        (OUT / "metrics_p0_incremental.json").write_text(json.dumps(results, indent=2))
        print(f"{model}: spatial={sp['auc_mean']:.4f}±{sp['auc_std']:.3f} "
              f"random={rd['auc_mean']:.4f} Δ={rd['auc_mean']-sp['auc_mean']:.4f} "
              f"best_ep={sp['per_fold_best_epoch']}")

    wall = {"RF_spatial": 0.601, "XGB_spatial": 0.588, "RF_random": 0.898,
            "phase1_graphsage_spatial": 0.618, "phase1_gcn_spatial": 0.624}
    out = {"task": "T1a", "phase": "P0", "smoke": SMOKE_TEST, "seed": C.SEED,
           "cap_km": CAP_KM, "k": K_NN, "n_blocks": nb,
           "change": "robust multi-micro-block early-stop + ReduceLROnPlateau + patience 50",
           "models": results, "wall": wall,
           "elapsed_min": round((time.time() - t0) / 60, 2)}
    (OUT / "metrics_p0.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT/'metrics_p0.json'} in {out['elapsed_min']} min")


if __name__ == "__main__":
    main()
