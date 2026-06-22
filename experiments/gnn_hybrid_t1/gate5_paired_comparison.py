"""Gate-5 matched paired comparison (CPU, cheap, tabular only — NO GNN training).

Builds the MISSING matched arms and runs the paired tests the driver could not:
  - XGB-alone on the SAME core features as the hybrid's tabular block (isolates the
    embedding contribution; the published wall 0.588 used a LARGER feature set so is
    not a clean ablation).
  - Hybrid per-fold (from metrics.json) vs GNN-alone P0 (phase2 metrics_p0.json),
    both seed 42, KMeans k=8, sorted-block order -> fold-aligned.
Reuses the project's own Nadeau-Bengio + Wilcoxon helpers for consistency.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/wiloic/M2/Recherche/tmp/pfas/pfas-gnn")
from src import config as C
from src import data as D
from src import features as F
from src import splits as S
from src import targets as T
from src.baselines_t1 import _corrected_resampled_ttest, _wilcoxon_paired
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

t0 = time.time()
df = D.load(smoke=False)
y = T.build_T1a(df).to_numpy().astype(int)
feature_cols = C.feature_columns(include_location=False, cocontam="core")
fold = S.spatial_block_folds(df, k=8)            # seed default = 42 in config
blocks = sorted(set(fold.tolist()))
print(f"data={df.shape} prevalence={y.mean():.4f} n_features={len(feature_cols)} blocks={blocks}")

# ---- XGB-alone (core features), row-level, per fold, FeaturePipeline fit on TRAIN only
xgb_per_fold, n_tr_list, n_te_list = [], [], []
for b in blocks:
    te = fold == b
    tr = ~te
    pipe = F.FeaturePipeline(feature_cols, encode="frequency")
    pipe.fit_transform(df.iloc[np.where(tr)[0]], None)
    Xtr, _ = pipe.transform(df.iloc[np.where(tr)[0]])
    Xte, _ = pipe.transform(df.iloc[np.where(te)[0]])
    ytr, yte = y[tr], y[te]
    pos = ytr.sum(); neg = len(ytr) - pos
    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=neg / max(pos, 1), eval_metric="logloss",
        tree_method="hist", n_jobs=4, random_state=42)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else float("nan")
    xgb_per_fold.append(float(auc))
    n_tr_list.append(int(tr.sum())); n_te_list.append(int(te.sum()))
    print(f"  fold {b}: XGB-core row-AUC={auc:.4f}  n_tr={tr.sum()} n_te={te.sum()}")

xgb_core = np.array(xgb_per_fold)
print(f"\nXGB-alone (core feats): mean={xgb_core.mean():.4f} ± {xgb_core.std():.4f}")

# ---- load the matched per-fold arrays
hyb = json.load(open("experiments/gnn_hybrid_t1/metrics.json"))
hyb_pf = np.array([f["roc_auc"] for f in
                   hyb["per_relation"]["subbasin_knn"]["spatial"]["per_fold"]])
gnn_p0 = json.load(open("experiments/gnn_phase2/metrics_p0.json"))
gnn_pf = np.array(gnn_p0["models"]["graphsage"]["per_fold_spatial"])

print(f"\nHYBRID   per-fold: {np.round(hyb_pf,4).tolist()}  mean={hyb_pf.mean():.4f}")
print(f"GNN-alone(P0)    : {np.round(gnn_pf,4).tolist()}  mean={gnn_pf.mean():.4f}")
print(f"XGB-alone(core)  : {np.round(xgb_core,4).tolist()}  mean={xgb_core.mean():.4f}")
print(f"global OOF hybrid: {hyb['per_relation']['subbasin_knn']['spatial']['global_oof_auc']:.4f}")

ntr = int(np.mean(n_tr_list)); nte = int(np.mean(n_te_list))

def report(name, a, bvec):
    nb = _corrected_resampled_ttest(a, bvec, ntr, nte)
    wc = _wilcoxon_paired(a, bvec)
    md = float(np.mean(a - bvec))
    nwin = int((a > bvec).sum())
    print(f"\n[{name}]  mean_diff={md:+.4f}  wins={nwin}/{len(a)}")
    print(f"   Nadeau-Bengio: t={nb['t']:.3f} p={nb['p']:.4f}")
    print(f"   Wilcoxon     : W={wc['w']} p={wc['p']}")
    sig = (nb["p"] < 0.05) or (wc["p"] < 0.05 if wc["p"] == wc["p"] else False)
    print(f"   >0.03 noise? {md>0.03}   significant(p<.05)? {sig}   "
          f"REAL GAIN? {bool(md>0.03 and sig)}")

report("HYBRID vs XGB-alone(core)", hyb_pf, xgb_core)
report("HYBRID vs GNN-alone(P0)",  hyb_pf, gnn_pf)
print(f"\nelapsed={time.time()-t0:.1f}s")
