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
DATA_PARQUET = PROJECT_ROOT / "data" / "CA-PFAS-ASGWS.parquet"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# ----------------------------------------------------------------- keys / controls
WELL_ID = "gm_well_id"           # group key for all splits (eval C2)
DATE_COL = "collection_date"     # used to derive temporal features, not a feature itself
DATASET_COL = "gm_dataset_name"  # design confounder -> control/audit only, NOT a feature (C6)
LAT, LON = "latitude", "longitude"

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
# Constant / duplicated / quasi-empty context columns (profiling §1).
DUPLICATE_COCONTAM = ["cocontam_dce12c", "cocontam_btbzt"]  # == cocontam_tmb124 (keep tmb124)
QUASI_EMPTY_COLS = ["cocontam_xylenes"]                     # 99.8% missing
# Soil sub-features >90% missing (hydro critique: prune).
HIGH_NA_SOIL = ["soil_silt_coarse_pct", "soil_silt_fine_pct",
                "soil_gradation_uniformity", "soil_gradation_curvature"]
DROP_COLS = DUPLICATE_COCONTAM + QUASI_EMPTY_COLS + HIGH_NA_SOIL + ["pfas_class_assignment"]

# ------------------------------------------------------------------- feature groups
# Non-leaking context features, grouped by family (profiling §7, hydro critique).
LOCATION_PURE = [LAT, LON]                                   # toggle: carry via graph k-NN
ADMIN_GEO_CAT = ["county", "regional_board", "dwr_region", "dwr_basin",
                 "sgma_basin_name", "sgma_subbasin_name", "sgma_region_office"]
WELL_FEATS = ["gm_well_category", "well_depth_ft"]
GEOTRACKER = ["dist_geotracker_km", "nearest_geotracker_type",
              "n_geotracker_within_1km", "n_geotracker_within_3km",
              "n_geotracker_within_10km", "n_geotracker_within_50km"]

# Cocontaminants: hydro-trusted hydrogeochemical core vs broader (audit via SHAP).
COCONTAM_CORE = ["cocontam_no3n", "cocontam_tds", "cocontam_mn", "cocontam_as",
                 "cocontam_so4", "cocontam_fe"]
COCONTAM_ALL = COCONTAM_CORE + [
    "cocontam_btbzs", "cocontam_pbzn", "cocontam_edb", "cocontam_dbcp",
    "cocontam_tcpr123", "cocontam_tce", "cocontam_pce", "cocontam_mtbe",
    "cocontam_tca111", "cocontam_tca112", "cocontam_dce12t", "cocontam_vc",
    "cocontam_sty", "cocontam_ebz", "cocontam_naph", "cocontam_bz", "cocontam_bzme",
    "cocontam_tmb124", "cocontam_dce11", "cocontam_dca12", "cocontam_dca11",
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
                 "root_zone_moist_kg_m2", "temp_c", "snowpack_mm", "gldas_dist_km",
                 "soil_moisture_total_mm"]
AIR_AQS = ["aqs_pm25_ugm3", "aqs_pm10_ugm3", "aqs_no2_ppb", "aqs_so2_ppb",
           "aqs_wind_ms", "aqs_humidity_pct", "aqs_ozone_ppb", "aqs_co_ppm"]
TEMPORAL_DERIVED = ["year", "month_sin", "month_cos"]   # built from collection_date

# Categorical vs numeric handling.
CATEGORICAL_LOW_CARD = ["gm_well_category", "regional_board", "dwr_region",
                        "sgma_region_office", "nearest_geotracker_type"]
CATEGORICAL_HIGH_CARD = ["county", "dwr_basin", "sgma_basin_name", "sgma_subbasin_name",
                         "soil_texture_class"]
# Features that are counts/distances/concentrations -> log1p before scaling.
LOG1P_FEATS = (GEOTRACKER[2:] + ["dist_geotracker_km"] + COCONTAM_ALL)

def feature_columns(*, include_location=False, cocontam="all", include_air=True):
    """Assemble the candidate feature column list (before fold-aware transforms).

    Args mirror the ablations the eval asked for:
      include_location : add raw lat/lon as node features (else carry via graph only).
      cocontam         : "core" (hydro-trusted) | "all" | "none".
      include_air      : keep low-value AQS air block.
    """
    cc = {"all": COCONTAM_ALL, "core": COCONTAM_CORE, "none": []}[cocontam]
    cols = (ADMIN_GEO_CAT + WELL_FEATS + GEOTRACKER + cc + SOIL + CLIMATE_HYDRO
            + TEMPORAL_DERIVED)
    if include_air:
        cols += AIR_AQS
    if include_location:
        cols += LOCATION_PURE
    return cols

# --------------------------------------------------------------------- spatial CV
N_SPATIAL_BLOCKS = 8     # primary KMeans blocks on well coords (profiling/eval C3)
N_RANDOM_FOLDS = 8       # group-by-well random folds for the random-vs-spatial Δ
