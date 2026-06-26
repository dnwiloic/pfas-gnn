"""Frozen shared contract for the PFAS pipeline.

Every constant here is justified by experiments/profilage/ (REPORT.md, T2_TARGETS.md,
EVAL_PROTOCOL.md). Downstream agents (baselines, GNN, multilabel) import from here so
the target definitions, leakage blocklist and CV scheme stay identical everywhere.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- paths
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PARQUET = PROJECT_ROOT / "data" / "CA-PFAS-ASGWS_v2.parquet"
# Fallback sur v1 si v2 absent (compatibility)
if not DATA_PARQUET.exists():
    DATA_PARQUET = PROJECT_ROOT / "data" / "CA-PFAS-ASGWS.parquet"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# ----------------------------------------------------------------- keys / controls
WELL_ID = "gm_well_id"           # group key for all splits (eval C2)
DATE_COL = "collection_date"     # used to derive temporal features, not a feature itself
DATASET_COL = "gm_dataset_name"  # design confounder -> control/audit only, NOT a feature (C6)
LAT, LON = "latitude", "longitude"
SGMA_SUBBASIN = "sgma_subbasin_name"  # aquifer sub-basin: CONSTRAINS mechanistic edges only,
                                      # NEVER a node feature (eval §2.4; full clique REFUSED)

# ----------------------------------------------------------------------- analytes
# The 31 measured PFAS analytes (each has *_ngL, *_detected, label_* in the raw file).
ANALYTES = [
    "ADONA", "F53B_major", "F53B_minor", "FTS_4_2", "FTS_6_2", "FTS_8_2", "HFPO_DA",
    "NEtFOSAA", "NFDHA", "NMeFOSAA", "PFBA", "PFBS", "PFDA", "PFDS", "PFDoDA",
    "PFEESA", "PFHpA", "PFHpS", "PFHxA", "PFHxS", "PFMBA", "PFMPA", "PFNA", "PFOA",
    "PFOS", "PFOSAm", "PFPeA", "PFPeS", "PFTeDA", "PFTrDA", "PFUnDA",
]

def ngl(a: str) -> str: return f"{a}_ngL"
def detected(a: str) -> str: return f"{a}_detected"
def label(a: str) -> str: return f"label_{a}"

NGL_COLS = [ngl(a) for a in ANALYTES]
DETECTED_COLS = [detected(a) for a in ANALYTES]
LABEL_COLS = [label(a) for a in ANALYTES]

# Labels constant at 0 in the raw file (never measurable as positive).
CONSTANT_LABELS = ["label_PFEESA", "label_PFMBA", "label_PFMPA"]

# ----------------------------------------------------- target-leakage blocklist (96)
# Anything derived from a PFAS measurement. MUST be excluded from FEATURES.
# (The *_ngL / *_detected columns are still READ to BUILD the targets, then dropped.)
DERIVED_TARGET_COLS = ["sum_pfas_ngL", "target_sum_gt70", "pfas_class_assignment"]
LEAKAGE_BLOCKLIST = NGL_COLS + DETECTED_COLS + LABEL_COLS + DERIVED_TARGET_COLS
assert len(LEAKAGE_BLOCKLIST) == 96, len(LEAKAGE_BLOCKLIST)

# --------------------------------------------------------- regulatory thresholds
# US EPA 2024 final NPDWR individual MCLs (ng/L). Compounds absent here are not
# federally regulated individually. See T2_TARGETS.md §7 for references.
EPA_MCL = {"PFOA": 4.0, "PFOS": 4.0, "PFHxS": 10.0, "PFNA": 10.0, "HFPO_DA": 10.0}

# EPA Hazard Index health-based water concentrations (ng/L); HI = sum(C_i / HBWC_i).
HI_HBWC = {"PFHxS": 10.0, "PFNA": 10.0, "HFPO_DA": 10.0, "PFBS": 2000.0}

ANALYTICAL_THRESHOLD = 2.0       # quantification-level fallback for T2 (ng/L)
T1B_SUM_THRESHOLD = 70.0         # secondary T1: sum_pfas_ngL > 70

# ------------------------------------------------------------------- T2 label set
# Hybrid scheme (T2_TARGETS.md): EPA MCL where regulated else analytical 2.0 ng/L,
# detection guard applied. Core = prevalence >= 5%; PFNA kept as rare-regulated.
T2_CORE = ["PFOS", "PFBS", "PFHxA", "PFOA", "PFHpA", "PFBA", "PFPeA", "PFHxS", "PFPeS"]
T2_RARE_REGULATED = ["PFNA"]
T2_LABELS = T2_CORE + T2_RARE_REGULATED   # HFPO_DA excluded (~0% in CA groundwater)

def t2_threshold(analyte: str) -> float:
    """Threshold used to build the T2 label for one analyte."""
    return EPA_MCL.get(analyte, ANALYTICAL_THRESHOLD)

# --------------------------------------------------------------- columns to drop
# Constant / duplicated / quasi-empty context columns (profiling §1 + extended blacklist v2).

# >85% missing — unusable
HIGH_NA_COCONTAM = [
    "cocontam_xylenes",   # 99.8% missing
    "cocontam_btbzt",     # 98.7% missing (duplicate of tmb124)
    "cocontam_dce12c",    # 98.7% missing (duplicate of tmb124)
    "cocontam_tmb124",    # 98.7% missing (master of the 3 duplicates, still unusable)
    "cocontam_no3n",      # 94.9% missing
]
HIGH_NA_WELL = ["well_depth_ft"]    # 94.5% missing; superseded by depth_eff_ft in v2
HIGH_NA_SOIL = ["soil_silt_coarse_pct", "soil_silt_fine_pct",
                "soil_gradation_uniformity", "soil_gradation_curvature"]

# Pre-computed log1p transforms — redundant with their raw columns.
# Keep the RAW (depth_eff_ft, depth_to_water_m) and apply log1p in the pipeline.
REDUNDANT_PRETRANSFORM = ["depth_eff_log1p", "depth_to_water_log1p"]

# Methodological artifacts — encode data-collection quality, not a hydro mechanism.
# gldas_dist_km  : distance to nearest GLDAS grid cell (imputation quality indicator).
# dist_nearest_gwl_km: distance to nearest groundwater-level gauge (gradient reliability flag).
# Including either as a predictor would let the model learn "well-instrumented sites differ
# from poorly-instrumented ones" — a study-design artefact, not a PFAS transport mechanism.
METHODOLOGICAL_ARTIFACTS = ["gldas_dist_km", "dist_nearest_gwl_km"]

DROP_COLS = (HIGH_NA_COCONTAM + HIGH_NA_WELL + HIGH_NA_SOIL
             + REDUNDANT_PRETRANSFORM + METHODOLOGICAL_ARTIFACTS
             + ["pfas_class_assignment"])

# ------------------------------------------------------------------- feature groups
# Non-leaking context features, grouped by family (profiling §7, hydro critique).
LOCATION_PURE = [LAT, LON]                                   # toggle: carry via graph k-NN
# ⚠️  SPATIAL PREVALENCE RISK: these encode the contamination map, not a hydro mechanism.
#    regional_board AUC=0.621, sgma_region_office AUC=0.586 — equivalent to raw lat/lon.
#    Prefer using spatial CV blocks over including these as features; audit SHAP.
ADMIN_GEO_CAT = ["county", "regional_board", "dwr_region", "dwr_basin",
                 "sgma_basin_name", "sgma_subbasin_name", "sgma_region_office"]
# ⚠️  gm_well_category: AUC=0.608 — MONITORING 55% vs DOMESTIC 7% prevalence.
#    Encodes surveillance intent (monitoring wells are placed at contaminated sites),
#    not a hydrogeological predictor. Audit SHAP; consider ablation for supply-well prediction.
WELL_FEATS = ["gm_well_category"]   # well_depth_ft dropped (94.5% missing, superseded by depth_eff_ft)
GEOTRACKER = ["dist_geotracker_km", "nearest_geotracker_type",
              "n_geotracker_within_1km", "n_geotracker_within_3km",
              "n_geotracker_within_10km", "n_geotracker_within_50km"]

# Cocontaminants: hydro-trusted hydrogeochemical core vs broader (audit via SHAP).
# cocontam_no3n removed from CORE (94.9% missing → in DROP_COLS).
COCONTAM_CORE = ["cocontam_tds", "cocontam_mn", "cocontam_as",
                 "cocontam_so4", "cocontam_fe"]
COCONTAM_ALL = COCONTAM_CORE + [
    "cocontam_btbzs", "cocontam_pbzn", "cocontam_edb", "cocontam_dbcp",
    "cocontam_tcpr123", "cocontam_tce", "cocontam_pce", "cocontam_mtbe",
    "cocontam_tca111", "cocontam_tca112", "cocontam_dce12t", "cocontam_vc",
    "cocontam_sty", "cocontam_ebz", "cocontam_naph", "cocontam_bz", "cocontam_bzme",
    # cocontam_tmb124 removed (98.7% missing → DROP_COLS)
    "cocontam_dce11", "cocontam_dca12", "cocontam_dca11",
    "cocontam_btbzn", "cocontam_fc113", "cocontam_bdcme", "cocontam_tcb124",
    "cocontam_dbcme", "cocontam_fc12", "cocontam_tbme", "cocontam_ctcl",
    "cocontam_clbz", "cocontam_fc11", "cocontam_pca", "cocontam_dcbz12",
    "cocontam_dcbz13", "cocontam_dcpa12",
]
SOIL = ["soil_sand_pct", "soil_clay_pct", "soil_silt_pct", "soil_om_pct", "soil_ph",
        "soil_ksat_um_s", "soil_awc_cm_cm", "soil_bulk_density", "soil_sand_vfine_pct",
        "soil_sand_fine_pct", "soil_sand_medium_pct", "soil_sand_coarse_pct",
        "soil_sand_vcoarse_pct", "soil_water_1bar_pct", "soil_water_15bar_pct",
        "soil_texture_class", "soil_ratio_water_clay"]
CLIMATE_HYDRO = ["rainfall_mm_month", "et_mm_month", "runoff_mm", "soil_moi_0_10_kg_m2",
                 "soil_moi_10_40_kg_m2", "soil_moi_40_100_kg_m2", "soil_moi_100_200_kg_m2",
                 "root_zone_moist_kg_m2", "temp_c", "snowpack_mm",
                 "soil_moisture_total_mm"]
# gldas_dist_km removed (methodological artefact → DROP_COLS)
AIR_AQS = ["aqs_pm25_ugm3", "aqs_pm10_ugm3", "aqs_no2_ppb", "aqs_so2_ppb",
           "aqs_wind_ms", "aqs_humidity_pct", "aqs_ozone_ppb", "aqs_co_ppm"]
TEMPORAL_DERIVED = ["year", "month_sin", "month_cos"]   # built from collection_date

# ---- features dérivées validées (audit intra-bloc k=8, signal propre ~0.55-0.61)
# Occupation du sol NLCD (dev_intensity AUC intra=0.61, lc_developed=0.56)
LANDUSE = ["dev_intensity", "lc_developed"]
# Topographie 3DEP (elevation_m AUC intra=0.57 supply wells ; topo_missing = flag NA)
TOPOGRAPHY = ["elevation_m", "topo_missing"]
# Profondeur de nappe DWR-IDW (AUC intra=0.59 ; dtw_far = flag station >20 km)
DEPTH_WATER = ["depth_to_water_m", "dtw_far"]
# Profondeur/crépine de puits GAMA (AUC intra=0.58 supply wells ; ~51% couverture)
# ⚠️  gm_well_category encode le type de surveillance → risque de fuite en mode prédictif strict.
#    Ne PAS inclure gm_well_category dans les features si l'on veut la vraie perf supply-wells.
WELL_CONSTRUCTION = ["depth_eff_ft", "screen_mid_ft", "screen_length_ft", "depth_missing"]
# Gradient hydraulique DWR (mécaniste : direction/magnitude d'écoulement de la nappe)
# dist_nearest_gwl_km removed (interpolation quality indicator → DROP_COLS, not a predictor)
HYDRAULIC_GRADIENT = ["hydr_grad_mag_permil", "flow_dir_sin", "flow_dir_cos"]

# Categorical vs numeric handling.
CATEGORICAL_LOW_CARD = ["gm_well_category", "regional_board", "dwr_region",
                        "sgma_region_office", "nearest_geotracker_type"]
CATEGORICAL_HIGH_CARD = ["county", "dwr_basin", "sgma_basin_name", "sgma_subbasin_name",
                         "soil_texture_class"]
# Features that are counts/distances/concentrations -> log1p before scaling.
LOG1P_FEATS = (GEOTRACKER[2:] + ["dist_geotracker_km"] + COCONTAM_ALL
              + ["depth_to_water_m", "depth_eff_ft", "screen_mid_ft", "elevation_m"])

def feature_columns(*, include_location=False, cocontam="all", include_air=True,
                    include_derived=True):
    """Assemble the candidate feature column list (before fold-aware transforms).

    Args mirror the ablations the eval asked for:
      include_location : add raw lat/lon as node features (else carry via graph only).
      cocontam         : "core" (hydro-trusted) | "all" | "none".
      include_air      : keep low-value AQS air block.
      include_derived  : include validated derived features (landuse, topo, depth,
                         well construction, hydraulic gradient) — default True.
                         Set False to reproduce the original v1 feature set for ablation.
    """
    cc = {"all": COCONTAM_ALL, "core": COCONTAM_CORE, "none": []}[cocontam]
    cols = (ADMIN_GEO_CAT + WELL_FEATS + GEOTRACKER + cc + SOIL + CLIMATE_HYDRO
            + TEMPORAL_DERIVED)
    if include_air:
        cols += AIR_AQS
    if include_derived:
        cols += LANDUSE + TOPOGRAPHY + DEPTH_WATER + WELL_CONSTRUCTION + HYDRAULIC_GRADIENT
    if include_location:
        cols += LOCATION_PURE
    return cols

# --------------------------------------------------------------------- spatial CV
N_SPATIAL_BLOCKS = 8     # primary KMeans blocks on well coords (profiling/eval C3)
N_RANDOM_FOLDS = 8       # group-by-well random folds for the random-vs-spatial Δ


# --------------------------------------------------------------------------- GPU
import functools as _functools


@_functools.lru_cache(maxsize=1)
def gpu_available() -> bool:
    """True iff an NVIDIA GPU is usable (for XGBoost device='cuda').

    Cached (detection cost paid once). On a CPU smoke-test box -> False, so the tree
    models transparently fall back to CPU 'hist'. Force CPU with env PFAS_FORCE_CPU=1.
    Note: scikit-learn RandomForest / HistGradientBoosting / LogisticRegression are
    CPU-only — only XGBoost honours the GPU.
    """
    import os
    import shutil
    import subprocess
    if os.environ.get("PFAS_FORCE_CPU") == "1":
        return False
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def xgb_device_params() -> dict:
    """XGBoost params selecting GPU when available (xgboost>=2: device='cuda')."""
    return {"tree_method": "hist", "device": "cuda"} if gpu_available() \
        else {"tree_method": "hist", "device": "cpu"}
