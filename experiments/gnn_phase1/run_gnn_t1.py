"""Driver — GraphSAGE/GCN T1a on the well-level spatial graph, full triplet vs the WALL.

Writes experiments/gnn_phase1/{config.yaml,metrics.json}. Toggle SMOKE_TEST for a fast
CPU sanity run. The full run is light enough for CPU (~5 min per regime, graph has only
11 333 nodes / ~30k edges); on Colab GPU it is a few seconds per fold.

Reuses the frozen socle end to end: targets.build_T1a, splits.spatial_block_folds /
group_random_folds, graph.build_well_graph (C4 cut), metrics.binary_metrics.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src import config as C
from src import data, gnn

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
OUT = Path(__file__).resolve().parent
MODELS = ["graphsage", "gcn"]
CAP_KM = 1.5
K_NN = 8


def main():
    t0 = time.time()
    df = data.load(smoke=SMOKE_TEST, smoke_n=1500)
    common = dict(k=K_NN, cap_km=CAP_KM, cut_blocks=True, encode="frequency",
                  hidden=(32 if SMOKE_TEST else 64), layers=2, dropout=0.5,
                  max_epochs=(40 if SMOKE_TEST else 300),
                  patience=(15 if SMOKE_TEST else 30))
    nb = 3 if SMOKE_TEST else 8

    results = {}
    for model in MODELS:
        sp, _ = gnn.run_t1_cv(df, model_name=model, regime="spatial", n_blocks=nb, **common)
        rd, _ = gnn.run_t1_cv(df, model_name=model, regime="random", n_blocks=nb, **common)
        delta = sp["auc_mean"] - rd["auc_mean"]
        results[model] = {
            "auc_spatial": sp["auc_mean"], "auc_spatial_std": sp["auc_std"],
            "auc_random": rd["auc_mean"], "auc_random_std": rd["auc_std"],
            "delta_random_minus_spatial": rd["auc_mean"] - sp["auc_mean"],
            "per_fold_spatial": sp["per_fold_auc"],
            "per_fold_random": rd["per_fold_auc"],
            "total_removed_cross_block": sp["total_removed_cross_block"],
        }
        print(f"{model}: spatial={sp['auc_mean']:.4f}±{sp['auc_std']:.3f} "
              f"random={rd['auc_mean']:.4f} Δ={rd['auc_mean']-sp['auc_mean']:.4f}")

    wall = {"RF_spatial": 0.601, "XGB_spatial": 0.588,
            "RF_random_strict": 0.898, "RF_delta": 0.297}
    out = {"task": "T1a", "smoke": SMOKE_TEST, "seed": C.SEED,
           "cap_km": CAP_KM, "k": K_NN, "n_blocks": nb,
           "models": results, "wall": wall,
           "elapsed_min": round((time.time() - t0) / 60, 2)}
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT/'metrics.json'} in {out['elapsed_min']} min")


if __name__ == "__main__":
    main()
