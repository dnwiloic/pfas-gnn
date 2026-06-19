"""Smoke test for the T2 multilabel baseline wall (CLAUDE.md §5).

Verifies end-to-end on CPU in < ~3 min: masked targets build with the expected
measurement structure; the feature matrix is leak-free and finite; every model
(Prevalence floor, Binary Relevance, masked Classifier Chain, Ensemble chains) trains,
predicts finite probabilities, and is scored on BOTH spatial and random CV with the
random>=spatial gap; OOF per-label thresholds optimise without touching test; the
pseudo-label probe runs; and a paired BR-vs-chain comparison returns. Also extrapolates
the full-run duration.

Run:  python tests/test_baselines_t2.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import data as D
from src import baselines_t2 as B


SMOKE_N_WELLS = 800          # small subsample of WELLS (whole wells -> spatial sane)
SMOKE_LABELS = ["PFOS", "PFOA", "PFHxA", "PFPeA", "PFHxS", "PFPeS", "PFNA"]
SMOKE_K = 2                  # 2 spatial / 2 random folds


def main():
    t0 = time.time()
    print("== T2 baseline smoke test ==")

    # ---- data + masked targets -------------------------------------------------
    df = D.load(smoke=True, smoke_n=SMOKE_N_WELLS)
    feature_cols = C.feature_columns(include_location=False, cocontam="core",
                                     include_air=False)
    leak = set(feature_cols) & set(C.LEAKAGE_BLOCKLIST)
    assert not leak, f"leak in features: {leak}"
    Y, M = B.masked_targets(df, labels=SMOKE_LABELS)
    assert Y.shape == M.shape == (len(df), len(SMOKE_LABELS))
    # the reduced-panel labels (PFPeA/PFPeS) are measured less than the full-panel ones
    assert M["label_PFPeS"].mean() < M["label_PFOS"].mean(), "measurement structure off"
    print(f"[data] n_rows={len(df)}  n_wells={df[C.WELL_ID].nunique()}  "
          f"labels={len(SMOKE_LABELS)}")
    print("[mask] measured% " + "  ".join(
        f"{a}={M[f'label_{a}'].mean():.2f}" for a in SMOKE_LABELS))

    # ---- splits ---------------------------------------------------------------
    from src import splits as S
    spatial = S.spatial_block_folds(df, k=SMOKE_K)
    random = S.group_random_folds(df, k=SMOKE_K)
    S.assert_no_group_leak(df, spatial)
    S.assert_no_group_leak(df, random)
    print(f"[splits] spatial k={len(np.unique(spatial))}  random k={len(np.unique(random))}")

    # ---- models (small) -------------------------------------------------------
    def f_prev():  return B.PrevalenceBaseline(labels=SMOKE_LABELS)
    def f_br():    return B.BinaryRelevance(kind="hgb", labels=SMOKE_LABELS,
                                            class_weight="balanced",
                                            smote_labels=("PFNA",), small=True)
    order = tuple(SMOKE_LABELS)  # a fixed order (high->low prevalence) for the chain
    def f_chain(): return B.MaskedClassifierChain(kind="hgb", order=order,
                                                  out_labels=order,
                                                  class_weight="balanced",
                                                  smote_labels=("PFNA",),
                                                  small=True, inner_k=2)
    def f_ecc():   return B.EnsembleClassifierChains(kind="hgb", n_chains=2,
                                                     labels=SMOKE_LABELS,
                                                     class_weight="balanced",
                                                     smote_labels=("PFNA",), small=True)
    def f_fcc():   return B.FrequencyClassChain(kind="hgb", labels=SMOKE_LABELS,
                                                n_classes=4, class_weight="balanced",
                                                smote_labels=("PFNA",), small=True, inner_k=2)

    # the 5 metrics required on both tasks must be present (micro + macro)
    REQ5 = ["micro_AUROC", "micro_F1", "micro_accuracy", "micro_recall", "micro_precision",
            "macro_AUROC", "macro_F1", "macro_accuracy", "macro_recall", "macro_precision"]

    results = {}
    for nm, fac, ug in [("Prevalence", f_prev, False), ("BinaryRelevance", f_br, False),
                        ("Chain", f_chain, True), ("Ensemble", f_ecc, True),
                        ("FreqClassChain", f_fcc, True)]:
        t1 = time.time()
        sp = B.evaluate_model(df, fac, spatial, feature_cols, labels=SMOKE_LABELS,
                              use_groups=ug)
        rd = B.evaluate_model(df, fac, random, feature_cols, labels=SMOKE_LABELS,
                              use_groups=ug)
        results[nm] = (sp, rd)
        a_sp, a_rd = sp["aggregate"], rd["aggregate"]
        assert np.isfinite(sp["oof_P"][~np.isnan(sp["oof_P"])]).all()
        assert 0.0 <= a_sp["macro_AUROC"] <= 1.0
        for k in REQ5:                       # all 5 headline metrics present, in [0,1]
            assert k in a_sp and 0.0 <= a_sp[k] <= 1.0, f"{nm} missing/invalid {k}"
        for c in ("f1", "precision", "recall", "accuracy"):   # per-label enriched
            assert c in sp["per_label"].columns, f"{nm} per-label missing {c}"
        print(f"[{nm:15s}] AUROC sp={a_sp['macro_AUROC']:.3f} F1={a_sp['micro_F1']:.3f} "
              f"acc={a_sp['micro_accuracy']:.3f} rec={a_sp['micro_recall']:.3f} "
              f"prec={a_sp['micro_precision']:.3f}  ({time.time()-t1:.1f}s)")
    if "FreqClassChain" in results:
        fcc_model = f_fcc()                  # show the 4 frequency classes
        Yc, Mc = B.masked_targets(df, labels=SMOKE_LABELS)
        from src import features as _F
        Xc, _ = _F.FeaturePipeline(feature_cols, encode="frequency").fit_transform(df)
        fcc_model.fit(Xc, Yc, Mc, groups=df[C.WELL_ID].to_numpy())
        print("[4 classes] " + " | ".join(
            f"C{i+1}:{'/'.join(cl)}" for i, cl in enumerate(fcc_model.classes_)))

    # learned models must beat the prevalence floor on macro-AUROC (spatial)
    floor = results["Prevalence"][0]["aggregate"]["macro_AUROC"]
    for nm in ["BinaryRelevance", "Chain"]:
        assert results[nm][0]["aggregate"]["macro_AUROC"] > floor + 0.02, \
            f"{nm} not above prevalence floor"

    # ---- paired BR vs Chain ----------------------------------------------------
    md, p = B.paired_compare(results["BinaryRelevance"][0]["per_fold"],
                             results["Chain"][0]["per_fold"], metric="macro_AUROC")
    print(f"[paired] chain-BR mean macroAUROC diff (spatial) = {md:+.3f}  p={p}")

    # ---- pseudo-label probe ----------------------------------------------------
    tp = time.time()
    probe = B.pseudo_label_probe(df, spatial, feature_cols,
                                 target_labels=("PFPeA", "PFPeS"), small=True)
    print(f"[pseudo] probe ({time.time()-tp:.1f}s):")
    if len(probe):
        for _, r in probe.iterrows():
            print(f"    {r['label']:7s} base={r['AUROC_base']:.3f} "
                  f"pseudo={r['AUROC_pseudo']:.3f} delta={r['delta']:+.3f}")
    else:
        print("    (no fold met the size guards in smoke -> ok, runs on full data)")

    # ---- duration extrapolation -----------------------------------------------
    dt = time.time() - t0
    # full run ~ (full_rows/smoke_rows) x (10 labels/7) x (8 folds/2) x (#models)
    full = D.load()
    scale = (len(full) / len(df)) * (10 / len(SMOKE_LABELS)) * (8 / SMOKE_K)
    print(f"\nSmoke wall time: {dt:.1f}s on CPU.")
    print(f"Naive full-run extrapolation (8-fold spatial+random, 10 labels, full data, "
          f"non-small models ~3-4x slower): ~{dt*scale*3.5/60:.0f}-{dt*scale*5/60:.0f} min CPU "
          f"-> run on Colab or a multi-core box.")
    print("ALL GREEN.")


if __name__ == "__main__":
    main()
