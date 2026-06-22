"""Gate-5 follow-up: SHAP of the XGB-alone wall on the SAME core features as the hybrid
tabular block. Certifies the base signal is physical (sources / soil / depth) and not a
design confounder. `gm_dataset_name` is already excluded by C6, so the remaining confounder
risk is whether co-contaminants dominate as "sample-was-analysed" proxies. CPU, no GPU.
"""
import json
import time

import numpy as np

from src import config as C
from src import data as D
from src import features as F
from src import targets as T
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import shap

t0 = time.time()
df = D.load(smoke=False)
y = T.build_T1a(df).to_numpy().astype(int)
feature_cols = C.feature_columns(include_location=False, cocontam="core")

pipe = F.FeaturePipeline(feature_cols, encode="frequency")
pipe.fit_transform(df, None)
X, names = pipe.transform(df)
names = list(names)
pos = y.sum(); neg = len(y) - pos
clf = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=neg / max(pos, 1),
    eval_metric="logloss", tree_method="hist", n_jobs=4, random_state=42)
clf.fit(X, y)
print(f"trained XGB-core on {X.shape}  in-sample AUC={roc_auc_score(y, clf.predict_proba(X)[:,1]):.3f}")

rng = np.random.RandomState(42)
idx = rng.choice(len(X), size=min(6000, len(X)), replace=False)
expl = shap.TreeExplainer(clf)
sv = expl.shap_values(X[idx])
mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(mean_abs)[::-1]

# signed direction: corr(feature value, shap) sign on the sample
Xs = X[idx]
ranked = []
for j in order:
    fv = Xs[:, j]; s = sv[:, j]
    if np.std(fv) > 0 and np.std(s) > 0:
        sign = float(np.corrcoef(fv, s)[0, 1])
    else:
        sign = float("nan")
    ranked.append({"feature": names[j], "mean_abs_shap": float(mean_abs[j]),
                   "dir_corr_value_shap": sign})

print("\n=== top 20 features by mean|SHAP| (XGB-alone, core) ===")
print(f"{'rank':>4} {'feature':32s} {'mean|SHAP|':>11} {'dir':>6}")
for i, r in enumerate(ranked[:20], 1):
    print(f"{i:>4} {r['feature']:32s} {r['mean_abs_shap']:>11.4f} {r['dir_corr_value_shap']:>+6.2f}")

# group families for the confounder check
def fam(n):
    if n.startswith("cocontam"): return "cocontaminant"
    if "geotracker" in n: return "source_proximity"
    if n.startswith("soil"): return "soil"
    if n.startswith(("rainfall","et_","runoff","temp","snow","gldas","soil_moi","root_zone")): return "climate_hydro"
    if n.startswith("aqs"): return "air_quality"
    if n in ("county","regional_board","dwr_region","dwr_basin","sgma_basin_name","sgma_subbasin_name","sgma_region_office"): return "admin_geo"
    if n in ("well_depth_ft","gm_well_category") or "well_depth" in n: return "well"
    if n in ("year","month_sin","month_cos"): return "temporal"
    return "other"

fam_tot = {}
for j in range(len(names)):
    fam_tot[fam(names[j])] = fam_tot.get(fam(names[j]), 0.0) + float(mean_abs[j])
tot = sum(fam_tot.values())
print("\n=== importance by family (share of total mean|SHAP|) ===")
for k, v in sorted(fam_tot.items(), key=lambda kv: -kv[1]):
    print(f"  {k:18s} {v/tot:6.1%}")

out = {"in_sample_auc": float(roc_auc_score(y, clf.predict_proba(X)[:, 1])),
       "n_features": len(names), "ranked": ranked,
       "family_share": {k: v / tot for k, v in fam_tot.items()},
       "gm_dataset_name_in_features": "gm_dataset_name" in feature_cols}
json.dump(out, open("experiments/gnn_hybrid_t1/shap_xgb_core.json", "w"), indent=2)

shap.summary_plot(sv, Xs, feature_names=names, plot_type="bar", max_display=20, show=False)
import matplotlib.pyplot as plt
plt.tight_layout(); plt.savefig("experiments/gnn_hybrid_t1/shap_xgb_core_top.png", dpi=110)
print(f"\nsaved shap_xgb_core.json + shap_xgb_core_top.png  elapsed={time.time()-t0:.1f}s")
