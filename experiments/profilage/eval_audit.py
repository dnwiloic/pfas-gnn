#!/usr/bin/env python3
"""
Audit methodologique du contrat d'evaluation PFAS (etape 3, eval-methodologist).
Verifie EMPIRIQUEMENT, graine fixee, sans refaire tout le profilage :
  A. Fuite cible : corr. residuelles des features candidates avec T1a/T1b ;
     dependance logique de toute colonne au vecteur cible.
  B. Garde-fou detection : combien de positifs T1a viennent d'un >seuil sur
     analyte NON detecte (limite de rapport) ? Impact sur la prevalence.
  C. Fuite pseudo-replicats : un meme puits dans train ET test gonfle-t-il le score ?
     (baseline LogisticRegression, split aleatoire ligne vs split GroupKFold puits).
  D. CV spatiale : contamination inter-blocs aux frontieres (puits proches, blocs
     differents) ; prevalence/positifs par bloc (degenerescence) ; buffer.
  E. Inflation spatiale : Delta(AUC split aleatoire - AUC split spatial) sur baseline.

Usage: python3 experiments/profilage/eval_audit.py
CWD attendu: racine pfas-gnn.
"""
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

DATA = "data/CA-PFAS-ASGWS.parquet"
OUT = "experiments/profilage/eval_audit_metrics.json"
df = pd.read_parquet(DATA)
R = {}

cols = list(df.columns)
ngL = [c for c in cols if c.endswith("_ngL") and c != "sum_pfas_ngL"]
det = [c for c in cols if c.endswith("_detected")]
lab = [c for c in cols if c.startswith("label_")]

# ----------------------------------------------------------------------
# Construire les cibles (recalcul independant)
# ----------------------------------------------------------------------
def col(name):
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

pfoa = col("PFOA_ngL").fillna(0); pfoa_d = col("PFOA_detected").fillna(False)
pfos = col("PFOS_ngL").fillna(0); pfos_d = col("PFOS_detected").fillna(False)
pfhxs = col("PFHxS_ngL").fillna(0); pfhxs_d = col("PFHxS_detected").fillna(False)
pfna = col("PFNA_ngL").fillna(0); pfna_d = col("PFNA_detected").fillna(False)
genx = col("HFPO_DA_ngL").fillna(0); genx_d = col("HFPO_DA_detected").fillna(False)
pfbs = col("PFBS_ngL").fillna(0); pfbs_d = col("PFBS_detected").fillna(False)

HI = pfhxs / 10.0 + pfna / 10.0 + genx / 10.0 + pfbs / 2000.0
# T1a SANS garde-fou (definition naive proposee)
T1a_raw = ((pfoa > 4) | (pfos > 4) | (HI >= 1.0)).astype(int)
# T1a AVEC garde-fou detection : ne compter un analyte que s'il est detecte
HI_g = (pfhxs.where(pfhxs_d, 0) / 10.0 + pfna.where(pfna_d, 0) / 10.0
        + genx.where(genx_d, 0) / 10.0 + pfbs.where(pfbs_d, 0) / 2000.0)
T1a_guard = (((pfoa > 4) & pfoa_d) | ((pfos > 4) & pfos_d) | (HI_g >= 1.0)).astype(int)
T1b = (df["sum_pfas_ngL"].fillna(0) > 70).astype(int)

# ----------------------------------------------------------------------
# B. GARDE-FOU DETECTION : impact chiffre
# ----------------------------------------------------------------------
guard = {}
guard["T1a_raw_prevalence"] = round(float(T1a_raw.mean()), 4)
guard["T1a_guarded_prevalence"] = round(float(T1a_guard.mean()), 4)
guard["n_raw_pos"] = int(T1a_raw.sum())
guard["n_guard_pos"] = int(T1a_guard.sum())
# Positifs perdus par le garde-fou = declenches uniquement sur un non-detect a LD elevee
lost = (T1a_raw == 1) & (T1a_guard == 0)
guard["n_positives_from_nondetect_only"] = int(lost.sum())
guard["frac_raw_positives_that_are_nondetect_artifacts"] = round(float(lost.sum() / max(T1a_raw.sum(), 1)), 4)
# Decompose : combien par voie PFOA / PFOS / HI
pfoa_only_nd = ((pfoa > 4) & ~pfoa_d).sum()
pfos_only_nd = ((pfos > 4) & ~pfos_d).sum()
guard["rows_PFOA_gt4_but_not_detected"] = int(pfoa_only_nd)
guard["rows_PFOS_gt4_but_not_detected"] = int(pfos_only_nd)
# Combien de fois un ngL>seuil coincide avec non-detect (par analyte HI)
for nm, s, d, thr in [("PFOA", pfoa, pfoa_d, 4), ("PFOS", pfos, pfos_d, 4),
                      ("PFHxS", pfhxs, pfhxs_d, 10), ("PFNA", pfna, pfna_d, 10),
                      ("HFPO_DA", genx, genx_d, 10), ("PFBS", pfbs, pfbs_d, 2000)]:
    guard[f"{nm}_gt_HBWC_and_NOT_detected"] = int(((s > thr) & ~d.astype(bool)).sum())
    guard[f"{nm}_gt_HBWC_total"] = int((s > thr).sum())
R["detection_guardrail"] = guard

# ----------------------------------------------------------------------
# A. FUITE CIBLE : balayage independant de TOUTE colonne vs cible
#    Inclut numerique ET categoriel (via prevalence cible par modalite -> eta).
# ----------------------------------------------------------------------
# blocklist proposee (96) : on verifie que tout le RESTE est propre
blocklist = set(ngL) | set(det) | set(lab) | {"sum_pfas_ngL", "target_sum_gt70", "pfas_class_assignment"}
candidate_features = [c for c in cols if c not in blocklist
                      and c not in ("gm_well_id", "collection_date")]
R["n_candidate_features_after_blocklist"] = len(candidate_features)

def corr_or_eta(series, target):
    """Corr de Pearson si numerique ; sinon eta (force d'association categorielle)."""
    s = series
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if s.nunique(dropna=True) <= 1:
            return None, "const"
        c = s.corr(target)
        return (None if pd.isna(c) else round(float(c), 4)), "pearson"
    if pd.api.types.is_bool_dtype(s):
        c = s.astype(float).corr(target)
        return (None if pd.isna(c) else round(float(c), 4)), "pearson_bool"
    # categoriel : eta^2 = var inter-groupes / var totale de la cible
    g = pd.DataFrame({"x": s.astype(str), "y": target})
    grand = g["y"].mean()
    ssb = g.groupby("x")["y"].apply(lambda v: len(v) * (v.mean() - grand) ** 2).sum()
    sst = ((g["y"] - grand) ** 2).sum()
    eta2 = ssb / sst if sst > 0 else 0.0
    return round(float(np.sqrt(eta2)), 4), "eta(cat)"

for tname, tvec in [("T1a", T1a_guard.astype(float)), ("T1b", T1b.astype(float))]:
    scan = []
    for c in candidate_features:
        val, kind = corr_or_eta(df[c], tvec)
        if val is not None:
            scan.append({"col": c, "assoc": val, "kind": kind, "absval": abs(val)})
    scan.sort(key=lambda x: -x["absval"])
    R[f"leakage_scan_{tname}_top25"] = [{k: v for k, v in d.items() if k != "absval"} for d in scan[:25]]

# ----------------------------------------------------------------------
# Construire un tableau de features simple pour les baselines C/E
#   (objectif : mesurer la STRUCTURE de validation, pas la performance ultime)
# ----------------------------------------------------------------------
num_feats = [c for c in candidate_features
             if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
# On retire lat/lon des FEATURES pour la mesure d'inflation spatiale honnete
for drop in ("latitude", "longitude"):
    if drop in num_feats:
        num_feats.remove(drop)
X = df[num_feats].copy()
# imputation mediane GLOBALE (suffisant pour un diagnostic ; le pipeline final impute intra-fold)
X = X.fillna(X.median(numeric_only=True))
X = X.loc[:, X.nunique() > 1]
y = T1a_guard.values
groups = df["gm_well_id"].values
lat = df["latitude"].values
lon = df["longitude"].values
R["baseline_n_features"] = X.shape[1]

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.cluster import KMeans

def fit_auc(tr, te):
    sc = StandardScaler().fit(X.iloc[tr])
    clf = LogisticRegression(max_iter=200, C=1.0)
    clf.fit(sc.transform(X.iloc[tr]), y[tr])
    p = clf.predict_proba(sc.transform(X.iloc[te]))[:, 1]
    return roc_auc_score(y[te], p)

# ----------------------------------------------------------------------
# C. FUITE PSEUDO-REPLICATS : split aleatoire LIGNE (puits peut etre des 2 cotes)
#    vs GroupKFold par puits. Sur les memes 5 plis stratifies vs groupes.
# ----------------------------------------------------------------------
# (C1) random row-level KFold (PSEUDO-REPLICAT POSSIBLE)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
auc_row = [fit_auc(tr, te) for tr, te in skf.split(X, y)]
# (C2) GroupKFold par puits (pas de fuite temporelle, mais split spatial ALEATOIRE)
gkf = GroupKFold(n_splits=5)
auc_group = [fit_auc(tr, te) for tr, te in gkf.split(X, y, groups)]
R["pseudo_replicate_leak"] = {
    "auc_random_row_kfold_mean": round(float(np.mean(auc_row)), 4),
    "auc_random_row_kfold_std": round(float(np.std(auc_row)), 4),
    "auc_groupkfold_well_mean": round(float(np.mean(auc_group)), 4),
    "auc_groupkfold_well_std": round(float(np.std(auc_group)), 4),
    "delta_row_minus_group": round(float(np.mean(auc_row) - np.mean(auc_group)), 4),
    "note": "delta>0 = un meme puits des deux cotes gonfle le score (fuite pseudo-replicats)",
}
# fraction des puits multi-prelevements pour contextualiser
vc = df.groupby("gm_well_id").size()
R["pseudo_replicate_leak"]["frac_rows_in_multi_event_wells"] = round(float((df["gm_well_id"].map(vc) > 1).mean()), 4)

# ----------------------------------------------------------------------
# E. INFLATION SPATIALE : split aleatoire-par-puits vs split SPATIAL par blocs.
#    Blocs KMeans sur (lat,lon) au niveau PUITS, k=8 (principal) et k=5.
#    On compare AUC GroupKFold(puits) [non-spatial] vs LeaveOneBlockOut [spatial].
# ----------------------------------------------------------------------
well = df.groupby("gm_well_id").agg(lat=("latitude", "first"), lon=("longitude", "first"))
spatial_res = {}
for nb in (8, 5):
    km = KMeans(n_clusters=nb, random_state=SEED, n_init=10)
    well_block = pd.Series(km.fit_predict(well[["lat", "lon"]].values), index=well.index)
    row_block = df["gm_well_id"].map(well_block).values
    # Leave-One-Block-Out
    aucs = []
    block_diag = {}
    for b in range(nb):
        te = np.where(row_block == b)[0]
        tr = np.where(row_block != b)[0]
        npos_te = int(y[te].sum()); npos_tr = int(y[tr].sum())
        block_diag[int(b)] = {
            "n_rows_test": int(len(te)), "test_prevalence": round(float(y[te].mean()), 4),
            "n_pos_test": npos_te,
        }
        # garde-fou degenerescence : besoin des 2 classes dans test
        if npos_te > 0 and npos_te < len(te):
            aucs.append(fit_auc(tr, te))
            block_diag[int(b)]["auc"] = round(float(aucs[-1]), 4)
        else:
            block_diag[int(b)]["auc"] = None
    spatial_res[f"kmeans_{nb}"] = {
        "auc_spatial_LOBO_mean": round(float(np.mean(aucs)), 4),
        "auc_spatial_LOBO_std": round(float(np.std(aucs)), 4),
        "auc_spatial_LOBO_min": round(float(np.min(aucs)), 4),
        "auc_spatial_LOBO_max": round(float(np.max(aucs)), 4),
        "delta_random_minus_spatial": round(float(np.mean(auc_group) - np.mean(aucs)), 4),
        "per_block": block_diag,
    }
R["spatial_inflation"] = spatial_res
R["spatial_inflation"]["auc_random_groupkfold_well_mean"] = round(float(np.mean(auc_group)), 4)

# ----------------------------------------------------------------------
# D. CONTAMINATION INTER-BLOCS : puits proches (< buffer) dans des blocs differents.
#    Quantifie combien de puits-test ont un voisin train a < d km (fuite de frontiere).
# ----------------------------------------------------------------------
from sklearn.neighbors import NearestNeighbors
coords = np.radians(well[["lat", "lon"]].values)  # haversine attend radians
km = KMeans(n_clusters=8, random_state=SEED, n_init=10)
wb = pd.Series(km.fit_predict(well[["lat", "lon"]].values), index=well.index)
nn = NearestNeighbors(n_neighbors=2, metric="haversine").fit(coords)
dist, nbr = nn.kneighbors(coords)  # rad ; *6371 -> km
nn_dist_km = dist[:, 1] * 6371.0
nn_idx = nbr[:, 1]
same_block = (wb.values == wb.values[nn_idx])
buffers = {}
for d_km in (0.5, 1.0, 2.0, 5.0):
    # puits dont le plus proche voisin est < d_km ET dans un AUTRE bloc => arete de frontiere
    border = (nn_dist_km < d_km) & (~same_block)
    within = (nn_dist_km < d_km)
    buffers[f"{d_km}km"] = {
        "n_wells_with_NN_within": int(within.sum()),
        "n_wells_NN_within_but_other_block": int(border.sum()),
        "frac_wells_cross_block_within": round(float(border.sum() / len(well)), 4),
        "frac_of_close_pairs_cross_block": round(float(border.sum() / max(within.sum(), 1)), 4),
    }
R["interblock_contamination"] = {
    "method": "KMeans k=8 well-level; NN haversine; cross-block = NN in different block",
    "buffers": buffers,
    "note": "frac_wells_cross_block_within mesure la part de puits dont le plus proche "
            "voisin (<d km) tombe dans un bloc different => fuite de frontiere a corriger par buffer",
}

with open(OUT, "w") as f:
    json.dump(R, f, indent=2, default=str)
print("WROTE", OUT)
print(json.dumps({
    "detection_guardrail": R["detection_guardrail"],
    "pseudo_replicate_leak": R["pseudo_replicate_leak"],
    "spatial_inflation_k8": {k: v for k, v in R["spatial_inflation"]["kmeans_8"].items() if k != "per_block"},
    "spatial_inflation_k5": {k: v for k, v in R["spatial_inflation"]["kmeans_5"].items() if k != "per_block"},
    "interblock": R["interblock_contamination"]["buffers"],
}, indent=2, default=str))
