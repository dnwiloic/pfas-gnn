"""Driver for the T2 multilabel baseline wall. Heavy logic lives in src/baselines_t2.py;
this script only orchestrates: builds folds, runs each model on spatial + random CV,
optimises per-label thresholds OOF, runs the SMOTE and pseudo-label ablations, writes
config.yaml / metrics.json / REPORT.md under experiments/baseline_t2/.

SMOKE_TEST=1 -> tiny subsample, 2 folds, small models, CPU < ~3 min (CLAUDE.md §5).
Full run is CPU-heavy (chains do inner-CV per label): use a multi-core box or Colab.

Usage:
    SMOKE_TEST=1 python experiments/baseline_t2/run_baseline_t2.py     # smoke
    python experiments/baseline_t2/run_baseline_t2.py                  # full
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config as C
from src import data as D
from src import splits as S
from src import baselines_t2 as B

SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
# Smoke writes to a subdir so it never clobbers the canonical full-data artifacts.
OUT = Path(__file__).resolve().parent / ("smoke" if SMOKE else "")
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- configuration
LABELS = C.T2_LABELS                       # 9 core + PFNA
FEATURE_COLS = C.feature_columns(include_location=False, cocontam="core",
                                 include_air=False)
K = 2 if SMOKE else 8
SMOKE_N = 800
# tractable HGB (class_weight handles imbalance; SMOTE measured as an ablation)
SMALL = SMOKE
MAX_ITER = 60 if SMOKE else 200


def _patch_max_iter():
    """Point the full-run HGB max_iter at MAX_ITER without editing src defaults."""
    import src.baselines_t2 as bt
    orig = bt.make_estimator

    def patched(kind="hgb", *, class_weight=None, small=False):
        est = orig(kind, class_weight=class_weight, small=small)
        if hasattr(est, "max_iter") and not small:
            est.set_params(max_iter=MAX_ITER)
        return est
    bt.make_estimator = patched


def main():
    _patch_max_iter()
    t0 = time.time()
    df = D.load(smoke=SMOKE, smoke_n=SMOKE_N)
    print(f"[data] rows={len(df)} wells={df[C.WELL_ID].nunique()} "
          f"labels={len(LABELS)} feats={len(FEATURE_COLS)} SMOKE={SMOKE}")

    spatial = S.spatial_block_folds(df, k=K)
    random = S.group_random_folds(df, k=K)
    S.assert_no_group_leak(df, spatial)
    S.assert_no_group_leak(df, random)

    order = tuple(LABELS)                  # high->low prevalence order for the chain

    def f_prev():  return B.PrevalenceBaseline(labels=LABELS)
    def f_br():    return B.BinaryRelevance(kind="hgb", labels=LABELS,
                                            class_weight="balanced", small=SMALL)
    def f_chain(): return B.MaskedClassifierChain(kind="hgb", order=order,
                                                  out_labels=order,
                                                  class_weight="balanced",
                                                  small=SMALL,
                                                  inner_k=2 if SMOKE else 3)
    def f_ecc():   return B.EnsembleClassifierChains(kind="hgb",
                                                     n_chains=2 if SMOKE else 3,
                                                     labels=list(LABELS),
                                                     class_weight="balanced",
                                                     small=SMALL)
    def f_fcc():   return B.FrequencyClassChain(kind="hgb", labels=list(LABELS),
                                                n_classes=4, class_weight="balanced",
                                                small=SMALL, inner_k=2 if SMOKE else 3)

    models = {"Prevalence": (f_prev, False), "BinaryRelevance": (f_br, False),
              "Chain": (f_chain, True), "Ensemble": (f_ecc, True),
              "FreqClassChain": (f_fcc, True)}

    results = {}
    for nm, (fac, ug) in models.items():
        t1 = time.time()
        sp = B.evaluate_model(df, fac, spatial, FEATURE_COLS, labels=LABELS, use_groups=ug)
        rd = B.evaluate_model(df, fac, random, FEATURE_COLS, labels=LABELS, use_groups=ug)
        results[nm] = {"spatial": sp, "random": rd}
        a, b = sp["aggregate"], rd["aggregate"]
        print(f"[{nm:15s}] sp macroAUROC={a['macro_AUROC']:.3f} microF1={a['micro_F1']:.3f} "
              f"Ham={a['Hamming']:.3f} EMR={a['EMR']:.3f} | rd macroAUROC={b['macro_AUROC']:.3f}"
              f"  ({time.time()-t1:.0f}s)")

    # ---- SMOTE ablation on the rare label PFNA --------------------------------
    def f_br_smote(): return B.BinaryRelevance(kind="hgb", labels=LABELS,
                                               class_weight="balanced",
                                               smote_labels=("PFNA",), small=SMALL)
    smote = B.evaluate_model(df, f_br_smote, spatial, FEATURE_COLS, labels=LABELS)
    pfna_cw = results["BinaryRelevance"]["spatial"]["per_label"]
    pfna_cw = float(pfna_cw.loc[pfna_cw.label == "PFNA", "AUROC"].iloc[0])
    pfna_sm = smote["per_label"]
    pfna_sm = float(pfna_sm.loc[pfna_sm.label == "PFNA", "AUROC"].iloc[0])
    print(f"[SMOTE ablation] PFNA AUROC class_weight={pfna_cw:.3f} vs +SMOTE={pfna_sm:.3f}")

    # ---- paired BR vs Chain (same spatial folds) -------------------------------
    paired = {}
    for metric in ["macro_AUROC", "micro_F1"]:
        md, p = B.paired_compare(results["BinaryRelevance"]["spatial"]["per_fold"],
                                 results["Chain"]["spatial"]["per_fold"], metric=metric)
        paired[metric] = {"chain_minus_br": md, "wilcoxon_p": p}
        print(f"[paired] {metric}: chain-BR={md:+.4f} p={p}")

    # ---- pseudo-label probe (semi-supervision) --------------------------------
    probe = B.pseudo_label_probe(df, spatial, FEATURE_COLS,
                                 target_labels=("PFBA", "PFPeA", "PFPeS"), small=SMALL)
    if len(probe):
        print("[pseudo] " + " | ".join(
            f"{r.label} d={r.delta:+.3f}" for _, r in probe.iterrows()))

    # ---- persist artifacts -----------------------------------------------------
    cfg = {"task": "T2_multilabel_baseline", "seed": C.SEED, "smoke": SMOKE,
           "labels": list(LABELS), "n_feature_cols": len(FEATURE_COLS),
           "feature_cols": list(FEATURE_COLS),
           "cv": {"spatial_k": int(K), "random_k": int(K), "group_key": C.WELL_ID},
           "models": list(models), "hgb_max_iter": MAX_ITER,
           "encode": "frequency", "threshold": "per-label F1 on OOF probabilities",
           "imbalance": "class_weight=balanced; SMOTE ablation on PFNA"}
    (OUT / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    def pack(res):
        return {"aggregate": res["aggregate"], "spread": res["spread"],
                "thresholds": res["thresholds"],
                "per_label": res["per_label"].to_dict(orient="records"),
                "per_fold": res["per_fold"].to_dict(orient="records")}

    metrics = {"smoke": SMOKE, "seed": C.SEED, "wall_s": round(time.time() - t0, 1),
               "models": {nm: {"spatial": pack(r["spatial"]), "random": pack(r["random"])}
                          for nm, r in results.items()},
               "smote_ablation_PFNA": {"class_weight": pfna_cw, "smote": pfna_sm},
               "paired_br_vs_chain": paired,
               "pseudo_label_probe": probe.to_dict(orient="records") if len(probe) else []}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    write_report(df, results, paired, smote, probe, metrics, pfna_cw, pfna_sm)
    print(f"\nDONE in {time.time()-t0:.0f}s -> {OUT}/{{config.yaml,metrics.json,REPORT.md}}")


# --------------------------------------------------------------------- reporting
def _row(nm, r):
    """One headline row: the 5 metrics (micro, spatial) + random AUROC + Delta."""
    sp, rd = r["spatial"]["aggregate"], r["random"]["aggregate"]
    return (f"| {nm} | {sp['macro_AUROC']:.3f} | {sp['micro_F1']:.3f} | "
            f"{sp['micro_accuracy']:.3f} | {sp['micro_recall']:.3f} | "
            f"{sp['micro_precision']:.3f} | {rd['macro_AUROC']:.3f} | "
            f"{rd['macro_AUROC']-sp['macro_AUROC']:+.3f} |")


def write_report(df, results, paired, smote, probe, metrics, pfna_cw, pfna_sm):
    L = []
    L.append("# REPORT — T2 multilabel baseline wall (PFAS / CA, strict predictive mode)\n")
    L.append(f"Seed {C.SEED}. Generated by `experiments/baseline_t2/run_baseline_t2.py` "
             f"(SMOKE={metrics['smoke']}). Targets = `src.targets.build_T2` (hybrid "
             "EPA-MCL/analytical + detection guard C1); features = socle frequency "
             "encoding, no PFAS measurement used; CV = socle spatial-block (reference) "
             "and group-random (Delta). Per-label measurement masking applied.\n")
    L.append("## Models x the 5 headline metrics (micro, spatial CV) + random + Delta\n")
    L.append("Metrics required on both tasks: **AUC-ROC, F1, accuracy, recall, "
             "precision** (here micro-averaged over measured cells; macro and per-label "
             "are in metrics.json). AUROC is macro (threshold-free).\n")
    L.append("| model | AUROC | F1 | accuracy | recall | precision | AUROC(rd) | dAUROC |")
    L.append("|---|---|---|---|---|---|---|---|")
    model_order = [nm for nm in ["Prevalence", "BinaryRelevance", "Chain",
                                 "Ensemble", "FreqClassChain"] if nm in results]
    for nm in model_order:
        L.append(_row(nm, results[nm]))
    L.append("")
    # 4 frequency classes used by FreqClassChain (Dong-style chain per class)
    Yf, Mf = B.masked_targets(df, labels=list(C.T2_LABELS))
    freq = {a: (float(Yf[f"label_{a}"].to_numpy()[Mf[f"label_{a}"].to_numpy()].mean())
                if Mf[f"label_{a}"].any() else 0.0) for a in C.T2_LABELS}
    order = sorted(C.T2_LABELS, key=lambda a: -freq[a])
    classes = [list(g) for g in np.array_split(np.array(order, dtype=object), 4) if len(g)]
    L.append("## FreqClassChain — 4 frequency classes (least rare -> rarest)\n")
    L.append("Cascade classifier chain in frequency order; per-class blocks below "
             "(prevalence on measured rows).\n")
    for i, cl in enumerate(classes, 1):
        L.append(f"- **Class {i}**: " + ", ".join(f"{a} ({freq[a]:.2f})" for a in cl))
    L.append("")
    L.append("## Per-label — Binary Relevance, spatial CV (5 metrics @ OOF threshold)\n")
    L.append("| label | n_meas | prevalence | AUROC | F1 | accuracy | recall | precision | AP |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    plbr = results["BinaryRelevance"]["spatial"]["per_label"]
    plch = results["Chain"]["spatial"]["per_label"].set_index("label")
    for _, r in plbr.iterrows():
        L.append(f"| {r.label} | {int(r.n_measured)} | {r.prevalence:.3f} | "
                 f"{r.AUROC:.3f} | {r.f1:.3f} | {r.accuracy:.3f} | {r.recall:.3f} | "
                 f"{r.precision:.3f} | {r.AP:.3f} |")
    L.append("")
    L.append("## Where do chains help? (per-label AUROC, chain - BR, spatial)\n")
    L.append("| label | BR AUROC | Chain AUROC | delta |")
    L.append("|---|---|---|---|")
    for _, r in plbr.iterrows():
        ch = float(plch.loc[r.label, "AUROC"])
        L.append(f"| {r.label} | {r.AUROC:.3f} | {ch:.3f} | {ch-r.AUROC:+.3f} |")
    L.append("")
    L.append("## Imbalance / rare labels\n")
    L.append(f"- PFNA (rare regulated, prevalence ~2.6%): class_weight AUROC "
             f"{pfna_cw:.3f} vs +SMOTE {pfna_sm:.3f} (delta {pfna_sm-pfna_cw:+.3f}).")
    L.append("- All learned labels use class_weight='balanced'; SMOTE measured, not assumed.\n")
    L.append("## Semi-supervision (pseudo-label probe on reduced-panel labels)\n")
    if len(probe):
        L.append("| label | AUROC base | AUROC pseudo | delta |")
        L.append("|---|---|---|---|")
        for _, r in probe.iterrows():
            L.append(f"| {r.label} | {r.AUROC_base:.3f} | {r.AUROC_pseudo:.3f} | "
                     f"{r.delta:+.3f} |")
    L.append("")
    L.append("## Paired BR vs Chain (Wilcoxon on spatial folds)\n")
    for m, v in paired.items():
        L.append(f"- {m}: chain-BR = {v['chain_minus_br']:+.4f}, Wilcoxon p = "
                 f"{v['wilcoxon_p']}.")
    L.append("")
    (OUT / "REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
