#!/usr/bin/env python3
"""
Profilage exhaustif et reproductible du jeu CA-PFAS-ASGWS.
Graine fixee. Aucun modele entraine. Sorties chiffrees ecrites en JSON + stdout.

Usage: python3 experiments/profilage/profile.py
Repertoire de travail attendu: racine du depot pfas-gnn.
"""
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

DATA = "data/CA-PFAS-ASGWS.parquet"
OUT = "experiments/profilage/profile_metrics.json"

df = pd.read_parquet(DATA)
R = {}  # results dict

# ----------------------------------------------------------------------
# 1. STRUCTURE & TYPES
# ----------------------------------------------------------------------
R["shape"] = list(df.shape)
R["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1e6, 1)
R["dtypes"] = df.dtypes.astype(str).value_counts().to_dict()

# Column families by name semantics
cols = list(df.columns)
ngL = [c for c in cols if c.endswith("_ngL") and c != "sum_pfas_ngL"]
det = [c for c in cols if c.endswith("_detected")]
lab = [c for c in cols if c.startswith("label_")]
coc = [c for c in cols if c.startswith("cocontam_")]
soil = [c for c in cols if c.startswith("soil_")]
aqs = [c for c in cols if c.startswith("aqs_")]
geot = [c for c in cols if "geotracker" in c]
R["family_counts"] = {
    "pfas_concentration_ngL": len(ngL),
    "detection_flags": len(det),
    "precomputed_labels": len(lab),
    "cocontaminants": len(coc),
    "soil": len(soil),
    "aqs_air": len(aqs),
    "geotracker": len(geot),
}
R["pfas_analytes_ngL"] = ngL
R["detection_cols"] = det
R["label_cols"] = lab

# Constant / quasi-constant
nunq = df.nunique(dropna=False)
const_cols = nunq[nunq <= 1].index.tolist()
R["constant_cols"] = const_cols
quasi = []
for c in cols:
    vc = df[c].value_counts(dropna=False, normalize=True)
    if len(vc) > 1 and vc.iloc[0] >= 0.99:
        quasi.append({"col": c, "top_frac": round(float(vc.iloc[0]), 4),
                      "top_val": str(vc.index[0])})
R["quasi_constant_cols"] = quasi

# Duplicate columns (identical content)
dup_pairs = []
num_cols = df.select_dtypes(include=[np.number, bool]).columns
checked = set()
for i, a in enumerate(num_cols):
    if a in checked:
        continue
    for b in num_cols[i + 1:]:
        if b in checked:
            continue
        if df[a].equals(df[b]):
            dup_pairs.append([a, b])
            checked.add(b)
R["duplicate_columns"] = dup_pairs

# ----------------------------------------------------------------------
# 2. GRANULARITY
# ----------------------------------------------------------------------
R["granularity"] = {
    "n_rows": len(df),
    "n_unique_wells": int(df["gm_well_id"].nunique()),
    "n_unique_well_date": int(df.drop_duplicates(["gm_well_id", "collection_date"]).shape[0]),
    "full_duplicate_rows": int(df.duplicated().sum()),
    "date_min": str(df["collection_date"].min()),
    "date_max": str(df["collection_date"].max()),
}
vc = df.groupby("gm_well_id").size()
R["granularity"]["rows_per_well"] = {
    "min": int(vc.min()), "median": float(vc.median()),
    "mean": round(float(vc.mean()), 2), "max": int(vc.max()),
    "wells_with_multiple_rows": int((vc > 1).sum()),
}

# ----------------------------------------------------------------------
# 3. MISSINGNESS & CARDINALITY per column
# ----------------------------------------------------------------------
prof = []
for c in cols:
    s = df[c]
    miss = float(s.isna().mean())
    entry = {"col": c, "dtype": str(s.dtype), "missing_frac": round(miss, 4),
             "n_unique": int(s.nunique(dropna=True))}
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        d = s.dropna()
        if len(d):
            entry.update({
                "min": float(d.min()), "p50": float(d.median()),
                "max": float(d.max()), "mean": round(float(d.mean()), 4),
                "skew": round(float(d.skew()), 3) if len(d) > 2 else None,
                "frac_zero": round(float((d == 0).mean()), 4),
                "frac_negative": round(float((d < 0).mean()), 4),
            })
    elif s.dtype == object or str(s.dtype).startswith("large_string") or s.dtype == "string":
        top = s.value_counts(dropna=True).head(5)
        entry["top_modalities"] = {str(k): int(v) for k, v in top.items()}
    prof.append(entry)
R["column_profile"] = prof

# Sentinel scan: look for -999, -9999, 9999 style sentinels in numeric cols
sentinels = {}
for c in num_cols:
    d = df[c].dropna()
    for sv in (-999, -9999, -99, 9999, -1):
        if (d == sv).sum() > 0 and c not in ngL:
            sentinels.setdefault(c, []).append({"val": sv, "count": int((d == sv).sum())})
R["sentinel_candidates"] = sentinels

# ----------------------------------------------------------------------
# 4. PFAS MEASUREMENT / DETECTION availability
# ----------------------------------------------------------------------
analyte_stats = []
for c in ngL:
    base = c[:-4]  # strip _ngL
    dcol = base + "_detected"
    s = df[c]
    meas_frac = float(s.notna().mean())  # fraction of rows with a numeric value
    det_frac = float(df[dcol].mean()) if dcol in df.columns else None
    # among measured & detected, distribution
    pos = s[(df.get(dcol) == True)] if dcol in df.columns else s.dropna()
    analyte_stats.append({
        "analyte": base, "ngL_col": c, "det_col": dcol if dcol in df.columns else None,
        "frac_rows_measured(non_null)": round(meas_frac, 4),
        "frac_detected": round(det_frac, 4) if det_frac is not None else None,
        "median_when_detected": round(float(pos.median()), 4) if len(pos) else None,
        "p95_when_detected": round(float(pos.quantile(0.95)), 4) if len(pos) else None,
        "max": round(float(s.max()), 4) if s.notna().any() else None,
    })
analyte_stats.sort(key=lambda x: -(x["frac_detected"] or 0))
R["analyte_stats"] = analyte_stats

# How are non-detects encoded in *_ngL? compare nulls vs zeros vs detected flag
nd_encoding = {}
for c in ngL[:5] + ["PFOA_ngL", "PFOS_ngL", "PFHxS_ngL"]:
    if c not in df.columns:
        continue
    base = c[:-4]
    dcol = base + "_detected"
    s = df[c]
    nd_encoding[c] = {
        "n_null": int(s.isna().sum()),
        "n_zero": int((s == 0).sum()),
        "n_pos": int((s > 0).sum()),
        "n_detected_true": int((df[dcol] == True).sum()) if dcol in df.columns else None,
        "null_iff_not_detected": None,
    }
    if dcol in df.columns:
        # when not detected, is ngL null or zero?
        nd = s[df[dcol] == False]
        nd_encoding[c]["when_notdet_null_frac"] = round(float(nd.isna().mean()), 4)
        nd_encoding[c]["when_notdet_zero_frac"] = round(float((nd == 0).mean()), 4)
        dd = s[df[dcol] == True]
        nd_encoding[c]["when_det_null_frac"] = round(float(dd.isna().mean()), 4)
        nd_encoding[c]["when_det_pos_frac"] = round(float((dd > 0).mean()), 4)
R["nondetect_encoding"] = nd_encoding

# ----------------------------------------------------------------------
# 5. LEAKAGE EVIDENCE - correlation of precomputed target/labels with concentrations
# ----------------------------------------------------------------------
# sum_pfas_ngL vs individual concentrations
sumcheck = {}
sumcheck["sum_pfas_null_frac"] = round(float(df["sum_pfas_ngL"].isna().mean()), 4)
# reconstruct sum from analytes (treating null as 0)
recon = df[ngL].fillna(0).sum(axis=1)
both = pd.DataFrame({"stored": df["sum_pfas_ngL"], "recon": recon}).dropna()
sumcheck["corr_storedsum_vs_reconstructed"] = round(float(both["stored"].corr(both["recon"])), 6)
sumcheck["max_abs_diff"] = round(float((both["stored"] - both["recon"]).abs().max()), 4)
sumcheck["frac_exact_match_1ngL"] = round(float(((both["stored"] - both["recon"]).abs() < 1).mean()), 4)
R["sum_reconstruction"] = sumcheck

# target_sum_gt70 vs sum_pfas_ngL
tg = df["target_sum_gt70"]
R["target_sum_gt70"] = {
    "values": df["target_sum_gt70"].value_counts(dropna=False).to_dict(),
    "prevalence": round(float(tg.mean()), 4),
    "matches_sum_gt_70": round(float(((df["sum_pfas_ngL"] > 70).astype(int) == tg).mean()), 4),
    "corr_with_sum_pfas": round(float(pd.Series(tg).corr(df["sum_pfas_ngL"])), 4),
}

# label_X vs X_detected and X_ngL
label_leak = []
for lc in lab:
    base = lc[len("label_"):]
    dcol = base + "_detected"
    ncol = base + "_ngL"
    e = {"label": lc, "prevalence": round(float(df[lc].mean()), 5)}
    if dcol in df.columns:
        e["equals_detected_frac"] = round(float((df[lc] == df[dcol].astype(int)).mean()), 5)
        e["corr_with_detected"] = round(float(df[lc].corr(df[dcol].astype(int))), 4)
    if ncol in df.columns:
        e["corr_with_ngL"] = round(float(df[lc].corr(df[ncol])), 4)
    label_leak.append(e)
R["label_leakage"] = label_leak

# Correlation of every numeric context col with target_sum_gt70 (to catch hidden leakage)
target = df["target_sum_gt70"].astype(float)
ctx_num = [c for c in num_cols if c not in ngL and c != "sum_pfas_ngL"
           and c != "target_sum_gt70" and not c.startswith("label_")]
corrs = []
for c in ctx_num:
    s = df[c]
    if s.nunique(dropna=True) <= 1:
        continue
    cc = s.corr(target)
    if pd.notna(cc):
        corrs.append({"col": c, "corr_with_T1": round(float(cc), 4)})
corrs.sort(key=lambda x: -abs(x["corr_with_T1"]))
R["context_corr_with_T1_top20"] = corrs[:20]

# detection flag correlations (these are leakage by construction)
detcorr = []
for c in det:
    cc = df[c].astype(int).corr(target)
    if pd.notna(cc):
        detcorr.append({"col": c, "corr_with_T1": round(float(cc), 4)})
detcorr.sort(key=lambda x: -abs(x["corr_with_T1"]))
R["detection_corr_with_T1"] = detcorr

# ----------------------------------------------------------------------
# 6. TARGET DEFINITIONS (proposed, EPA 2024)
# ----------------------------------------------------------------------
# EPA 2024 MCL (ng/L): PFOA 4, PFOS 4. Hazard Index for PFHxS, PFNA, HFPO-DA (GenX), PFBS.
# HI health-based water concentrations (ng/L): PFHxS 10, PFNA 10, HFPO_DA 10, PFBS 2000.
def col(name):
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

pfoa = col("PFOA_ngL").fillna(0)
pfos = col("PFOS_ngL").fillna(0)
pfhxs = col("PFHxS_ngL").fillna(0)
pfna = col("PFNA_ngL").fillna(0)
genx = col("HFPO_DA_ngL").fillna(0)
pfbs = col("PFBS_ngL").fillna(0)

HI = pfhxs / 10.0 + pfna / 10.0 + genx / 10.0 + pfbs / 2000.0
exceed_epa = ((pfoa > 4) | (pfos > 4) | (HI >= 1.0)).astype(int)
exceed_sum70 = (df["sum_pfas_ngL"].fillna(0) > 70).astype(int)
any_detect = (df[det].any(axis=1)).astype(int)

R["target_definitions"] = {
    "T1a_EPA2024_MCL_HI": {
        "rule": "PFOA>4 OR PFOS>4 OR HazardIndex>=1 (HI=PFHxS/10+PFNA/10+HFPODA/10+PFBS/2000)",
        "prevalence": round(float(exceed_epa.mean()), 4),
        "n_positive": int(exceed_epa.sum()),
        "imbalance_ratio_neg_per_pos": round(float((1 - exceed_epa.mean()) / max(exceed_epa.mean(), 1e-9)), 1),
    },
    "T1b_sum_gt70_precomputed": {
        "rule": "sum_pfas_ngL > 70 (matches stored target_sum_gt70)",
        "prevalence": round(float(exceed_sum70.mean()), 4),
        "n_positive": int(exceed_sum70.sum()),
    },
    "T1c_any_detection": {
        "rule": "any analyte detected",
        "prevalence": round(float(any_detect.mean()), 4),
        "n_positive": int(any_detect.sum()),
    },
}
# agreement between T1a and stored target
R["target_definitions"]["T1a_vs_T1b_agreement"] = round(float((exceed_epa == exceed_sum70).mean()), 4)

# ----------------------------------------------------------------------
# 7. T2 multilabel structure
# ----------------------------------------------------------------------
# Use detection-based labels (label_* == detected) ; report positive counts
lab_pos = {lc: int(df[lc].sum()) for lc in lab}
lab_prev = {lc: round(float(df[lc].mean()), 5) for lc in lab}
R["t2_label_positive_counts"] = dict(sorted(lab_pos.items(), key=lambda x: -x[1]))
R["t2_label_prevalence"] = lab_prev
# candidate labels with >=1% prevalence and >=200 positives
cand = [lc for lc in lab if df[lc].mean() >= 0.01 and df[lc].sum() >= 200]
R["t2_candidate_labels"] = cand
# co-occurrence correlation among candidate labels
if len(cand) >= 2:
    cm = df[cand].astype(int).corr()
    # store top correlated pairs
    pairs = []
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            pairs.append({"a": a, "b": b, "corr": round(float(cm.loc[a, b]), 3)})
    pairs.sort(key=lambda x: -x["corr"])
    R["t2_top_label_correlations"] = pairs[:15]
# cardinality of label set per row (how many analytes detected)
lab_card = df[cand].sum(axis=1) if cand else pd.Series(0, index=df.index)
R["t2_labels_per_row"] = {
    "mean": round(float(lab_card.mean()), 3),
    "median": float(lab_card.median()),
    "max": int(lab_card.max()),
    "frac_zero_labels": round(float((lab_card == 0).mean()), 4),
}

# ----------------------------------------------------------------------
# 8. SPATIAL STRUCTURE & AUTOCORRELATION
# ----------------------------------------------------------------------
spatial = {}
spatial["lat_range"] = [round(float(df["latitude"].min()), 4), round(float(df["latitude"].max()), 4)]
spatial["lon_range"] = [round(float(df["longitude"].min()), 4), round(float(df["longitude"].max()), 4)]
spatial["lat_lon_missing"] = [int(df["latitude"].isna().sum()), int(df["longitude"].isna().sum())]
spatial["n_counties"] = int(df["county"].nunique())
spatial["n_dwr_basin"] = int(df["dwr_basin"].nunique())
spatial["n_sgma_subbasin"] = int(df["sgma_subbasin_name"].nunique())
spatial["n_regional_board"] = int(df["regional_board"].nunique())

# Moran's I on T1 (EPA def) using well-level aggregation to avoid temporal pseudo-rep.
# Aggregate to unique wells: T1 = any sampling event of that well exceeds.
well = df.groupby("gm_well_id").agg(
    lat=("latitude", "first"), lon=("longitude", "first"),
    county=("county", "first")
)
well["t1"] = exceed_epa.groupby(df["gm_well_id"]).max().reindex(well.index).values
well = well.dropna(subset=["lat", "lon", "t1"])

# Sample for Moran (k-NN binary weights) deterministically
from sklearn.neighbors import NearestNeighbors
rng = np.random.default_rng(SEED)
n_samp = min(4000, len(well))
idx = rng.choice(len(well), size=n_samp, replace=False)
W = well.iloc[idx].reset_index(drop=True)
coords = W[["lat", "lon"]].values
y = W["t1"].astype(float).values
ybar = y.mean()
k = 8
nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
_, nbr = nn.kneighbors(coords)
nbr = nbr[:, 1:]  # drop self
num = 0.0
wsum = 0.0
for i in range(len(y)):
    for j in nbr[i]:
        num += (y[i] - ybar) * (y[j] - ybar)
        wsum += 1.0
den = np.sum((y - ybar) ** 2)
morans_I = (len(y) / wsum) * (num / den) if den > 0 else None
spatial["morans_I_T1"] = {
    "value": round(float(morans_I), 4),
    "k_neighbors": k,
    "n_wells_sampled": int(n_samp),
    "expected_under_null": round(-1 / (len(y) - 1), 5),
    "note": "well-level, T1=EPA exceedance (max over events); kNN binary weights",
}

# Distance-decay: positive-rate similarity vs distance bins using pairwise on a sample
# Compute fraction of concordant pairs (same T1) by distance band
from scipy.spatial import cKDTree
tree = cKDTree(coords)
bands_km = [(0, 1), (1, 5), (5, 20), (20, 100), (100, 1000)]
# approximate deg->km: 1 deg ~ 111 km
deg_per_km = 1 / 111.0
concord = {}
sample_pairs = 200000
ii = rng.integers(0, len(y), sample_pairs)
jj = rng.integers(0, len(y), sample_pairs)
mask = ii != jj
ii, jj = ii[mask], jj[mask]
dlat = (coords[ii, 0] - coords[jj, 0])
dlon = (coords[ii, 1] - coords[jj, 1]) * np.cos(np.radians(coords[ii, 0]))
dist_km = np.sqrt(dlat ** 2 + dlon ** 2) * 111.0
same = (y[ii] == y[jj]).astype(float)
for lo, hi in bands_km:
    m = (dist_km >= lo) & (dist_km < hi)
    if m.sum() > 50:
        concord[f"{lo}-{hi}km"] = {
            "n_pairs": int(m.sum()),
            "frac_same_T1": round(float(same[m].mean()), 4),
        }
spatial["concordance_by_distance"] = concord
spatial["baseline_concordance_random"] = round(float(ybar ** 2 + (1 - ybar) ** 2), 4)

# ----------------------------------------------------------------------
# 9. SPATIAL BLOCKS proposal (KMeans on lat/lon at WELL level)
# ----------------------------------------------------------------------
from sklearn.cluster import KMeans
blockinfo = {}
for nb in (5, 8, 10):
    km = KMeans(n_clusters=nb, random_state=SEED, n_init=10)
    lab_block = km.fit_predict(well[["lat", "lon"]].values)
    sizes = pd.Series(lab_block).value_counts().sort_index()
    # propagate block to row level via well id, compute T1 prevalence per block
    well_block = pd.Series(lab_block, index=well.index)
    prev = {}
    for b in range(nb):
        wb = well_block[well_block == b].index
        mb = df["gm_well_id"].isin(wb)
        prev[int(b)] = {
            "n_wells": int((well_block == b).sum()),
            "n_rows": int(mb.sum()),
            "T1_prevalence": round(float(exceed_epa[mb.values].mean()), 4) if mb.sum() else None,
        }
    blockinfo[f"kmeans_{nb}"] = {
        "well_sizes_min_max": [int(sizes.min()), int(sizes.max())],
        "per_block": prev,
    }
R["spatial_blocks"] = blockinfo
R["spatial"] = spatial

# ----------------------------------------------------------------------
# write
# ----------------------------------------------------------------------
with open(OUT, "w") as f:
    json.dump(R, f, indent=2, default=str)
print("WROTE", OUT)
print(json.dumps({k: R[k] for k in ["shape", "memory_mb", "family_counts",
      "granularity", "target_definitions", "t2_candidate_labels"]}, indent=2, default=str))
