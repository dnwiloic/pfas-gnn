"""Smoke test for the src/ socle (CLAUDE.md §5): verifies the whole non-graph
pipeline end-to-end on CPU in a few seconds, BEFORE any long Colab run.

Checks: data loads & cleans; targets build with the detection guard and expected
prevalences; the *_label_* censoring inflation is corrected; splits are leak-free
(no well straddles folds) and non-degenerate; the feature matrix contains no
target-leaking column and is finite; an end-to-end logistic baseline runs and the
spatial-CV AUC is BELOW the random-CV AUC (the expected spatial-leakage gap).

Run:  python tests/test_socle.py        (or: pytest -q tests/test_socle.py)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import data as D
from src import targets as T
from src import splits as S
from src import features as F

TOL = 0.02


def _approx(name, got, exp, tol=TOL):
    assert abs(got - exp) <= tol, f"{name}: {got:.3f} not within {tol} of {exp:.3f}"
    return f"  ok {name:24s} {got:.3f} (~{exp:.3f})"


def test_targets_and_guard():
    df = D.load()                       # full data (cheap, 3.6 MB)
    out = T.build_all(df)
    p = out["prevalence"]
    msgs = [
        _approx("T1a prevalence", p["T1a"], 0.445),
        _approx("T1b prevalence", p["T1b"], 0.248),
        _approx("T2 PFOS", p["label_PFOS"], 0.393),
        _approx("T2 PFOA", p["label_PFOA"], 0.340),
        _approx("T2 PFHxS (MCL 10)", p["label_PFHxS"], 0.146),
        _approx("T2 PFBS (fallback 2.0)", p["label_PFBS"], 0.373),
        _approx("T2 PFNA (rare reg.)", p["label_PFNA"], 0.025),
    ]
    # detection guard corrects the inflated raw label_* (censoring)
    guarded = float(T.build_T2(df, labels=["FTS_6_2"])["label_FTS_6_2"].mean())
    raw = float(df["label_FTS_6_2"].mean())
    assert guarded < 0.10 < raw, (guarded, raw)
    msgs.append(f"  ok detection guard FTS_6_2  guarded={guarded:.3f} < raw={raw:.3f}")
    return df, out, msgs


def test_splits(df, y):
    spatial = S.spatial_block_folds(df, k=C.N_SPATIAL_BLOCKS)
    random = S.group_random_folds(df, k=C.N_RANDOM_FOLDS)
    S.assert_no_group_leak(df, spatial)
    S.assert_no_group_leak(df, random)
    bp = S.block_prevalence(y, spatial)
    assert len(bp) >= 5, "need >=5 spatial folds for paired test (eval C3)"
    assert bp["n"].min() > 50 and 0.02 < bp["prevalence"].min(), "degenerate spatial fold"
    return spatial, random, [
        f"  ok spatial folds          k={len(bp)}  n[min,max]=[{bp.n.min()},{bp.n.max()}]"
        f"  prev[min,max]=[{bp.prevalence.min():.2f},{bp.prevalence.max():.2f}]",
        "  ok no well straddles folds (group leak C2)",
    ]


def _cv_auc(df, y, fold, feature_cols):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    aucs = []
    for _, tr, te in S.iter_folds(fold):
        pipe = F.FeaturePipeline(feature_cols, encode="target")
        Xtr, _ = pipe.fit_transform(df[tr], y[tr])
        Xte, _ = pipe.transform(df[te])
        assert np.isfinite(Xtr).all() and np.isfinite(Xte).all(), "non-finite features"
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs))


def test_features_no_leak_and_delta(df, y):
    cols = C.feature_columns(include_location=False, cocontam="all", include_air=True)
    leak = set(cols) & set(C.LEAKAGE_BLOCKLIST)
    assert not leak, f"leak in feature list: {leak}"
    # end-to-end on a moderate well subsample to stay fast
    sub = D.load(smoke=True, smoke_n=2500)
    ys = T.build_T1a(sub).to_numpy()
    sp = S.spatial_block_folds(sub, k=5)
    rd = S.group_random_folds(sub, k=5)
    auc_sp = _cv_auc(sub, ys, sp, cols)
    auc_rd = _cv_auc(sub, ys, rd, cols)
    assert 0.5 < auc_sp < 1.0 and 0.5 < auc_rd < 1.0, (auc_sp, auc_rd)
    assert auc_rd >= auc_sp - 0.01, "random CV should not be below spatial CV"
    return [
        f"  ok feature list leak-free  ({len(cols)} candidate cols)",
        f"  ok end-to-end LR  AUC random={auc_rd:.3f}  spatial={auc_sp:.3f}"
        f"  Delta={auc_rd - auc_sp:+.3f}",
    ]


def main():
    t0 = time.time()
    print("== PFAS socle smoke test ==")
    df, out, m1 = test_targets_and_guard()
    print("[targets]"); [print(x) for x in m1]
    sp, rd, m2 = test_splits(df, out["T1a"].to_numpy())
    print("[splits]"); [print(x) for x in m2]
    m3 = test_features_no_leak_and_delta(df, out["T1a"].to_numpy())
    print("[features/end-to-end]"); [print(x) for x in m3]
    dt = time.time() - t0
    print(f"\nALL GREEN in {dt:.1f}s on CPU.")


if __name__ == "__main__":
    main()
