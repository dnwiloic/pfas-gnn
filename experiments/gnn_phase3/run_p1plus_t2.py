"""P1+ driver — T2 by HETEROGENEOUS matrix completion (bipartite wells<->analytes
AUGMENTED with capped spatial well<->well edges), vs the T2 wall (BinaryRelevance
macro-AUROC spatial 0.680) AND vs P1 bipartite (0.681).

Same protocol/metrics as the wall and P1: spatial-block CV (reference) + group-random (Δ),
per-label measurement mask, the 5 multilabel metrics via metrics.multilabel_metrics, OOF
thresholds, per-label AUROC+AP. Writes experiments/gnn_phase3/metrics_p1plus.json with the
triplet, per-label table, and the FULL C4 audit (cross-block bipartite AND well edges, both
must be 0). Incremental checkpoint after the spatial regime so a kill never loses it.

Env toggles:
  SMOKE_TEST=1   fast CPU sanity run (tiny subsample, 3 blocks, few epochs).
  P1PLUS_ENCODER hetero_sage (default) | hgt | rgcn
  P1PLUS_DECODER mlp (default) | vgae
  P1PLUS_EPOCHS  max epochs (default 200; 40 in smoke)
  P1PLUS_REGIMES "spatial,random" (default) | "spatial"  (skip random to fit a CPU budget)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config as C
from src import data, gnn_hetero as GH

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
ENCODER = os.environ.get("P1PLUS_ENCODER", "hetero_sage")
DECODER = os.environ.get("P1PLUS_DECODER", "mlp")
EPOCHS = int(os.environ.get("P1PLUS_EPOCHS", "40" if SMOKE_TEST else "200"))
REGIMES = os.environ.get("P1PLUS_REGIMES", "spatial,random").split(",")
OUT = Path(__file__).resolve().parent

# Wall (baseline_t2) and P1 (gnn_phase2) reference numbers for the triplet comparison.
WALL = {
    "BinaryRelevance_macro_AUROC_spatial": 0.680,
    "BinaryRelevance_macro_AUROC_random": 0.902,
    "BinaryRelevance_delta": 0.222,
    "BinaryRelevance_micro_F1_spatial": 0.542,
    "Prevalence_macro_AUROC_spatial": 0.348,
}
P1 = {
    "Bipartite_macro_AUROC_spatial": 0.681,
    "Bipartite_macro_AUROC_random": 0.843,
    "Bipartite_delta": 0.162,
    "Bipartite_micro_F1_spatial": 0.547,
}
# wall BR per-label spatial AUROC / AP (baseline_t2 REPORT §2); P1 bipartite per-label AUROC.
WALL_PER_LABEL = {
    "PFOS": {"AUROC": 0.588, "AP": 0.450}, "PFBS": {"AUROC": 0.632, "AP": 0.487},
    "PFHxA": {"AUROC": 0.656, "AP": 0.519}, "PFOA": {"AUROC": 0.665, "AP": 0.443},
    "PFHpA": {"AUROC": 0.634, "AP": 0.379}, "PFBA": {"AUROC": 0.728, "AP": 0.592},
    "PFPeA": {"AUROC": 0.689, "AP": 0.559}, "PFHxS": {"AUROC": 0.660, "AP": 0.288},
    "PFPeS": {"AUROC": 0.721, "AP": 0.367}, "PFNA": {"AUROC": 0.831, "AP": 0.169},
}
P1_PER_LABEL_AUROC = {"PFOS": 0.638, "PFBS": 0.641, "PFHxA": 0.669, "PFOA": 0.663,
                      "PFHpA": 0.672, "PFBA": 0.706, "PFPeA": 0.698, "PFHxS": 0.657,
                      "PFPeS": 0.702, "PFNA": 0.766}


def _strip(res):
    return {k: v for k, v in res.items() if k not in ("P_row", "Y_row", "M_row")}


def main():
    t0 = time.time()
    df = data.load(smoke=SMOKE_TEST, smoke_n=1500)
    common = dict(
        encoder=ENCODER, decoder=DECODER,
        emb_dim=(16 if SMOKE_TEST else 32), hidden=(32 if SMOKE_TEST else 64),
        layers=2, dropout=0.3, heads=2, k=8, cap_km=1.5, lr=5e-3,
        gamma=1.0,                       # focal: concentrate on rare positives (PFNA)
        beta_kl=(1e-3 if DECODER == "vgae" else 0.0),
        max_epochs=EPOCHS, patience=(15 if SMOKE_TEST else 40))
    nb = 3 if SMOKE_TEST else 8

    results = {}
    if "spatial" in REGIMES:
        sp = GH.run_t2_hetero_cv(df, regime="spatial", n_blocks=nb, **common)
        results["spatial"] = sp
        (OUT / "metrics_p1plus_incremental.json").write_text(
            json.dumps({"spatial": _strip(sp), "encoder": ENCODER, "decoder": DECODER,
                        "elapsed_min_so_far": round((time.time() - t0) / 60, 2)}, indent=2))
        print(f"spatial[{ENCODER}/{DECODER}]: macro_AUROC={sp['macro_AUROC']:.4f} "
              f"micro_F1={sp['micro_F1']:.4f} cross_bip={sp['n_cross_block_bipartite']} "
              f"cross_well={sp['n_cross_block_well']} removed_well={sp['n_removed_well_cross_total']}")

    if "random" in REGIMES:
        rd = GH.run_t2_hetero_cv(df, regime="random", n_blocks=nb, **common)
        results["random"] = rd
        print(f"random[{ENCODER}/{DECODER}]:  macro_AUROC={rd['macro_AUROC']:.4f} "
              f"micro_F1={rd['micro_F1']:.4f}")

    out = {"task": "T2", "phase": "P1+", "model": f"HeteroCompletion[{ENCODER}/{DECODER}]",
           "smoke": SMOKE_TEST, "seed": C.SEED, "n_blocks": nb,
           "encoder": ENCODER, "decoder": DECODER, "focal_gamma": 1.0,
           "wall": WALL, "p1_bipartite": P1,
           "wall_per_label": WALL_PER_LABEL, "p1_per_label_AUROC": P1_PER_LABEL_AUROC,
           "elapsed_min": round((time.time() - t0) / 60, 2)}
    if "spatial" in results:
        out["spatial"] = _strip(results["spatial"])
    if "random" in results:
        out["random"] = _strip(results["random"])
    if "spatial" in results and "random" in results:
        out["delta_random_minus_spatial_macro_AUROC"] = (
            results["random"]["macro_AUROC"] - results["spatial"]["macro_AUROC"])

    (OUT / "metrics_p1plus.json").write_text(json.dumps(out, indent=2))
    msg = f"\nwrote {OUT/'metrics_p1plus.json'} in {out['elapsed_min']} min"
    if "spatial" in results:
        sp = results["spatial"]
        msg += (f" | spatial macro_AUROC {sp['macro_AUROC']:.3f} vs wall 0.680 / P1 0.681")
        if "delta_random_minus_spatial_macro_AUROC" in out:
            msg += f", Δ={out['delta_random_minus_spatial_macro_AUROC']:.3f}"
    print(msg)


if __name__ == "__main__":
    main()
