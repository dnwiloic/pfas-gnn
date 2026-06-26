"""T2 multilabel wall on the v2 dataset — isolate the enrichment lift vs v1 (macro-AUROC).

Same flags as the v1 baseline_t2 (cocontam="core", include_air=False) so the ONLY change is
the 16 v2 hydrogeo derived features (include_derived defaults True) -> clean v1->v2 lift.
Models: BinaryRelevance (the wall) + FrequencyClassChain (Dong's per-class chain). Spatial
+ random CV (triplet), per-label OOF thresholds. Source of truth: metrics_v2_t2.json.

    SMOKE_TEST=1 PFAS_FORCE_CPU=1 python3 experiments/baseline_t2_v2/run_v2_t2.py
    PFAS_FORCE_CPU=1 python3 experiments/baseline_t2_v2/run_v2_t2.py        # full ~40 min CPU
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config as C
from src import data as D
from src import splits as S
from src import baselines_t2 as B

SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
LABELS = C.T2_LABELS
FEATURE_COLS = C.feature_columns(include_location=False, cocontam="core", include_air=False)
K = 2 if SMOKE else 8
SMALL = SMOKE


BASE = B.default_base_kind()        # "xgb" if GPU else "hgb" (CPU)


def br_factory():
    return B.BinaryRelevance(kind=BASE, labels=list(LABELS), small=SMALL)


def fcc_factory():
    return B.FrequencyClassChain(kind=BASE, labels=list(LABELS), small=SMALL)


def main():
    t0 = time.time()
    df = D.load(smoke=SMOKE, smoke_n=800)
    spatial = S.spatial_block_folds(df, k=K)
    random = S.group_random_folds(df, k=K)
    S.assert_no_group_leak(df, spatial); S.assert_no_group_leak(df, random)
    print(f"[data] v2 rows={len(df)} wells={df[C.WELL_ID].nunique()} "
          f"labels={len(LABELS)} feats={len(FEATURE_COLS)} base={B.default_base_kind()} SMOKE={SMOKE}")

    out = {"smoke": SMOKE, "seed": C.SEED, "k": int(K), "dataset": "v2",
           "n_features": len(FEATURE_COLS), "labels": list(LABELS), "models": {}}
    for name, fac in [("BinaryRelevance", br_factory), ("FreqClassChain", fcc_factory)]:
        ts = time.time()
        sp = B.evaluate_model(df, fac, spatial, FEATURE_COLS, labels=LABELS, desc=f"{name}/sp")
        rd = B.evaluate_model(df, fac, random, FEATURE_COLS, labels=LABELS, desc=f"{name}/rd")
        spA, rdA = sp["aggregate"], rd["aggregate"]
        out["models"][name] = {
            "spatial": spA, "random": rdA,
            "delta_macroAUROC": float(rdA["macro_AUROC"] - spA["macro_AUROC"]),
            "per_label_spatial": sp["per_label"].to_dict("records"),
            "spread_spatial": sp["spread"],
            "elapsed_s": round(time.time() - ts, 1),
        }
        print(f"[{name}] sp macroAUROC={spA['macro_AUROC']:.4f} microF1={spA['micro_F1']:.3f} "
              f"EMR={spA.get('EMR', 0):.3f} | rd macroAUROC={rdA['macro_AUROC']:.4f} "
              f"| Δ={rdA['macro_AUROC']-spA['macro_AUROC']:+.4f}  ({time.time()-ts:.0f}s)")

    # ---- per-label lift figure (BR spatial)
    pl = out["models"]["BinaryRelevance"]["per_label_spatial"]
    # per_label is a list of dicts with 'label' and 'AUROC'
    rows = sorted([(d["label"], d.get("AUROC", np.nan)) for d in pl],
                  key=lambda t: -(t[1] if t[1] == t[1] else -1))
    names = [r[0] for r in rows]; aucs = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, aucs, color="#3d6b9c")
    ax.axhline(0.5, ls=":", color="grey")
    macro = out["models"]["BinaryRelevance"]["spatial"]["macro_AUROC"]
    ax.axhline(macro, ls="--", color="#b03a2e", label=f"macro-AUROC = {macro:.3f}")
    for i, a in enumerate(aucs):
        ax.text(i, a + .005, f"{a:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("AUROC spatiale (OOF)"); ax.set_ylim(0.4, 1.0)
    ax.set_title("T2 v2 — AUROC par PFAS (BinaryRelevance, CV spatiale)")
    ax.legend(); plt.xticks(rotation=45, ha="right"); fig.tight_layout()
    fig.savefig(FIG / "t2_per_label_auroc.png", dpi=140); plt.close(fig)
    print(f"[fig] {FIG/'t2_per_label_auroc.png'}")

    out["wall_s"] = round(time.time() - t0, 1)
    (OUT / "metrics_v2_t2.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nDONE in {out['wall_s']:.0f}s -> {OUT/'metrics_v2_t2.json'}")


if __name__ == "__main__":
    main()
