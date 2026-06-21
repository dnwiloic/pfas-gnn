"""P1 driver — T2 by bipartite matrix completion (wells x analytes), vs the T2 wall
(BinaryRelevance macro-AUROC spatial 0.680). Same protocol/metrics as the wall: spatial-
block CV (reference) + group-random (Δ), per-label measurement mask, the 5 multilabel
metrics via metrics.multilabel_metrics, OOF thresholds. Writes
experiments/gnn_phase2/metrics_p1.json (triplet, per-label, C4 audit).

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
from src import data, gnn_bipartite as GB

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
OUT = Path(__file__).resolve().parent


def _strip(res):
    """Drop heavy arrays before serialising."""
    return {k: v for k, v in res.items() if k not in ("P_row", "Y_row", "M_row")}


def main():
    t0 = time.time()
    df = data.load(smoke=SMOKE_TEST, smoke_n=1500)
    common = dict(emb_dim=(16 if SMOKE_TEST else 32), hidden=(32 if SMOKE_TEST else 64),
                  layers=2, dropout=0.3, lr=5e-3,
                  max_epochs=(40 if SMOKE_TEST else 300),
                  patience=(15 if SMOKE_TEST else 40))
    nb = 3 if SMOKE_TEST else 8

    sp = GB.run_t2_bipartite_cv(df, regime="spatial", n_blocks=nb, **common)
    (OUT / "metrics_p1_incremental.json").write_text(json.dumps(_strip(sp), indent=2))
    print(f"spatial: macro_AUROC={sp['macro_AUROC']:.4f} micro_F1={sp['micro_F1']:.4f} "
          f"cross_block_edges={sp['n_cross_block_edges']}")

    rd = GB.run_t2_bipartite_cv(df, regime="random", n_blocks=nb, **common)
    print(f"random:  macro_AUROC={rd['macro_AUROC']:.4f} micro_F1={rd['micro_F1']:.4f}")

    delta = rd["macro_AUROC"] - sp["macro_AUROC"]
    wall = {
        "BinaryRelevance_macro_AUROC_spatial": 0.680,
        "BinaryRelevance_macro_AUROC_random": 0.902,
        "BinaryRelevance_delta": 0.222,
        "BinaryRelevance_micro_F1_spatial": 0.542,
        "Prevalence_macro_AUROC_spatial": 0.348,
    }
    out = {"task": "T2", "phase": "P1", "model": "BipartiteCompletion",
           "smoke": SMOKE_TEST, "seed": C.SEED, "n_blocks": nb,
           "spatial": _strip(sp), "random": _strip(rd),
           "delta_random_minus_spatial_macro_AUROC": delta,
           "wall": wall, "elapsed_min": round((time.time() - t0) / 60, 2)}
    (OUT / "metrics_p1.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT/'metrics_p1.json'} in {out['elapsed_min']} min "
          f"(macro_AUROC spatial {sp['macro_AUROC']:.3f} vs wall 0.680, Δ={delta:.3f})")


if __name__ == "__main__":
    main()
