"""HGT -> embedding fusion -> stacking on the v2 dataset, PURE-MECHANISM feature set.

Figure phare du mémoire : compare HGT seul / fusion / stacking au mur tabulaire v2
(XGB pure_mech 0.653), apples-to-apples (MÊMES 98 features pure_mech comme features de
noeud du graphe ET pour le mur XGB in-run). Triplet (spatial/random/Delta) conservé.

    SMOKE_TEST=1 PFAS_FORCE_CPU=1 python3 experiments/hgt_fusion_stacking_t1_v2/run_v2.py
    PFAS_FORCE_CPU=1 python3 experiments/hgt_fusion_stacking_t1_v2/run_v2.py   # full, ~2-3h CPU
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config as C
from src import hgt_fusion_stacking_t1 as HFS

SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
EXP_DIR = Path(__file__).resolve().parent

# Same 98-feature PURE-MECHANISM set as the committed v2 wall (no admin encodings):
FEATURE_COLS = [c for c in C.feature_columns(include_location=False, cocontam="all",
                                             include_air=True)
                if c not in C.ADMIN_GEO_CAT]


def main():
    print(f"[v2-stacking] SMOKE={SMOKE} n_features={len(FEATURE_COLS)} "
          f"GPU={C.gpu_available()}")
    out = HFS.run(
        smoke=SMOKE,
        feature_cols=FEATURE_COLS,
        compute_delta=True,          # keep the (random, spatial, Delta) triplet
        exp_dir=EXP_DIR,
        write=True,
        verbose=False,
    )
    comp = out.get("comparison", {})
    wall = comp.get("in_run_xgb_wall_auc_mean", float("nan"))
    print(f"\n=== spatial (in-run XGB wall per-fold-mean = {wall:.4f}) ===")
    ba = comp.get("by_architecture", {})
    for arch in ["hgt_standalone", "embedding_fusion", "stacking"]:
        a = ba.get(arch, {})
        print(f"  {arch:18s} OOF={a.get('auc_oof_global', float('nan')):.4f} "
              f"pfm={a.get('auc_mean', float('nan')):.4f} "
              f"gain_vs_wall={a.get('gain_vs_in_run_wall', float('nan')):+.4f} "
              f"NBp={a.get('nadeau_bengio', {}).get('p', float('nan')):.3f} "
              f"verdict={a.get('verdict', '?')}")
    print(f"\nwall_s={out['meta'].get('elapsed_s', 0):.0f} -> {EXP_DIR/'metrics.json'}")


if __name__ == "__main__":
    main()
