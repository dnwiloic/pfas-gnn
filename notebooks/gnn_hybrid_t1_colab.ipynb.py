"""Generator for notebooks/gnn_hybrid_t1_colab.ipynb — autonomous Colab notebook (CLAUDE.md §4).
Run: python3 notebooks/gnn_hybrid_t1_colab.ipynb.py  -> writes the .ipynb next to it.

GNN-hybrid T1: nested-OOF GraphSAGE embedding + XGBoost fusion on two mechanistic-edge
relations ("subbasin_knn" and "spatial"), per the eval-validated §3 protocol
(EVAL_PROTOCOL_HYBRID.md).  Code lives in src/hybrid.py; this notebook only orchestrates.
"""
import json
from pathlib import Path


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


cells = []

# ---------------------------------------------------------------------------
cells.append(md("""# PFAS Groundwater — GNN-Hybrid T1 (nested-OOF GraphSAGE + XGBoost)

**AUTONOMOUS** (CLAUDE.md §4): bootstraps `src/` + the versioned dataset via `git clone`
(no Google Drive), installs **PyTorch Geometric** for the Colab torch wheel, then runs the
full nested-OOF hybrid loop from `src/hybrid.py` via `experiments/gnn_hybrid_t1/run_hybrid_t1.py`.

### What this notebook does
- **Two relations** (both eval-approved, §C.3): `subbasin_knn` (k-NN *within* SGMA
  sub-basin, cap 2 km — the mechanistic prior) and `spatial` (bare k-NN, cap 1.5 km —
  the baseline graph). Their AUC difference measures mechanistic gain.
- **Three arms per relation**: hybrid (GNN embed ⊕ XGBoost), GNN-alone reference,
  XGBoost-alone wall. The triplet `(random, spatial, Δ)` per arm is the honest metric.
- **Nested anti-leak loop** (§3.2–§3.3): every train-row embedding comes from a GNN that
  never saw that row's label; every test-row embedding comes from a GNN trained only on
  train blocks; threshold and calibration from inner-OOF only. C4 assertedon every GNN call
  (0 cross-block edges).
- **Checkpoints** written after each outer fold to `experiments/gnn_hybrid_t1/` so a Colab
  disconnect does not lose finished work.
- **Persistence cell** at the end: `files.download()` archive and/or `git push` (no Drive).

> SMOKE_TEST=True: CPU sanity check (< 3 min, tiny subsample).
> SMOKE_TEST=False: full GPU run — see duration estimate in Cell 0.
"""))

# ---------------------------------------------------------------------------
cells.append(md("## Cell 0 — User parameters (read this before running)"))
cells.append(code("""# ============================================================
# USER PARAMETERS — adjust before running
# ============================================================

SMOKE_TEST = False        # True = fast CPU sanity (<3 min); False = full GPU run

REPO_URL = "https://github.com/dnwiloic/pfas-gnn.git"
GIT_REF  = "main"        # branch or commit SHA to clone
DATA_PATH = "data/CA-PFAS-ASGWS.parquet"  # relative to repo root

# Which relations to run. Both = full sweep (recommended). One = ~half the time.
# "subbasin_knn" = mechanistic (primary); "spatial" = baseline graph (Δ reference).
RELATIONS = ["subbasin_knn", "spatial"]   # or ["subbasin_knn"] for a trimmed first run

# Full-run parameters (only used when SMOKE_TEST=False).
# These match run_hybrid_t1.py defaults; change only for ablation.
FULL_OUTER_K      = 8    # outer spatial CV blocks (LOBO)
FULL_INNER_K      = 4    # inner micro-blocks for OOF embeddings
FULL_GNN_EPOCHS   = 400
FULL_GNN_PATIENCE = 50
FULL_HIDDEN       = 64   # GNN hidden dim = XGB embedding features

# ============================================================
# DURATION ESTIMATE (Colab T4 GPU, SMOKE_TEST=False)
# ============================================================
# GNN count: 2 relations x 2 arms (spatial+random) x 8 outer x (4 inner + 1 test) = 160
# Measured phase-2 baseline: ~1 min/GNN at 400 epochs on full 11k-node graph (T4 GPU).
#   -> FULL sweep (2 relations): ~160 min ~ 2.5-3 h  [optimistic; expect 3-5 h with I/O]
#   -> TRIMMED (1 relation, e.g. RELATIONS=["subbasin_knn"]): ~80 min ~ 1.5-2.5 h
#
# NOTE: the prior REPORT.md figure (~40 h) used a CPU extrapolation of 15 min/GNN.
# The GPU is ~15x faster: actual measured pace is ~1 min/GNN on T4.
# Both estimates are printed at run time (see below). The trimmed config fits comfortably
# in one Colab session (12 h limit). Checkpoints allow resuming after a disconnect.
# ============================================================
print("Parameters set.")
print(f"  SMOKE_TEST={SMOKE_TEST}  RELATIONS={RELATIONS}")
if not SMOKE_TEST:
    n_gnn = len(RELATIONS) * 2 * FULL_OUTER_K * (FULL_INNER_K + 1)
    print(f"  Full-run GNN count: {n_gnn} trainings")
    print(f"  Estimated GPU wall time: {n_gnn * 1:.0f}–{n_gnn * 2:.0f} min "
          f"({n_gnn/60:.1f}–{n_gnn*2/60:.1f} h) on Colab T4")
    print("  If this exceeds your session budget, set RELATIONS=['subbasin_knn'] only.")
"""))

# ---------------------------------------------------------------------------
cells.append(md("## Cell 1 — GPU detection & versions"))
cells.append(code("""import sys, platform
print("Python  :", sys.version.split()[0])
print("Platform:", platform.platform())

IN_COLAB = False
try:
    import google.colab  # noqa
    IN_COLAB = True
except ImportError:
    pass
print("IN_COLAB:", IN_COLAB)

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print("torch   :", torch.__version__, " CUDA avail:", cuda_ok)
    if cuda_ok:
        print("GPU     :", torch.cuda.get_device_name(0))
    else:
        print("WARNING: no GPU detected. The GNN training will use CPU.")
        print("  -> For SMOKE_TEST=True this is fine (CPU run expected).")
        print("  -> For SMOKE_TEST=False: go to Runtime > Change runtime type > GPU.")
except ImportError:
    print("torch not yet installed (Colab base usually has it pre-installed).")

# Note: XGBoost part of the hybrid runs on CPU (tree_method='hist'); only the GNN
# embedding step benefits from GPU. A GPU runtime is strongly recommended.
"""))

# ---------------------------------------------------------------------------
cells.append(md("## Cell 2 — Clone repo (code + versioned dataset), no Drive"))
cells.append(code("""import os, subprocess

REPO_DIR = "/content/pfas-gnn" if IN_COLAB else os.getcwd()

if IN_COLAB:
    if not os.path.isdir(REPO_DIR):
        print("Cloning repo (brings src/ AND data/ — no Drive needed)...")
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    print(f"Checking out {GIT_REF} ...")
    subprocess.run(["git", "-C", REPO_DIR, "checkout", GIT_REF], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

# Guard: dataset must be present (it is versioned in the repo)
assert os.path.exists(DATA_PATH), (
    f"Dataset missing at {DATA_PATH} — clone may have failed or DATA_PATH is wrong. "
    "Check REPO_URL and GIT_REF above."
)
print("workdir :", os.getcwd())
print("dataset :", DATA_PATH, "— present:", os.path.exists(DATA_PATH))

# Guard: check key src symbols exist (anti-stale-code safeguard)
import importlib.util, pathlib
for mod_name, symbol in [
    ("src.hybrid", "run_hybrid_t1"),
    ("src.gnn",    "train_gnn_and_embed"),
    ("src.graph",  "build_well_graph"),
]:
    spec = importlib.util.spec_from_file_location(
        mod_name, pathlib.Path(REPO_DIR) / mod_name.replace(".", "/") + ".py")
    mod = importlib.util.module_from_spec(spec)
    assert hasattr(spec.loader, "exec_module"), f"Cannot load {mod_name}"
    # Light check: just verify the source file contains the expected symbol
    src_text = (pathlib.Path(REPO_DIR) / (mod_name.replace(".", "/") + ".py")).read_text()
    assert f"def {symbol}" in src_text, (
        f"Symbol '{symbol}' not found in {mod_name} — push the latest src/ to the repo "
        f"at {REPO_URL} and re-run this cell."
    )
    print(f"  {mod_name}.{symbol} — OK")
print("Code guard passed.")
"""))

# ---------------------------------------------------------------------------
cells.append(md("## Cell 3 — Install PyTorch Geometric for the runtime's torch wheel"))
cells.append(code("""def ensure_pyg():
    try:
        import torch_geometric  # noqa
        print("torch_geometric already present:", torch_geometric.__version__)
        return
    except ImportError:
        pass
    import torch
    tv = torch.__version__.split("+")[0]
    cuda = torch.version.cuda
    tag = f"cu{cuda.replace('.', '')}" if (cuda and torch.cuda.is_available()) else "cpu"
    idx = f"https://data.pyg.org/whl/torch-{tv}+{tag}.html"
    print(f"Installing PyG wheels for torch {tv}, tag {tag} ...")
    print(f"  Index URL: {idx}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"],
                   check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch_scatter", "torch_sparse", "-f", idx], check=False)

ensure_pyg()
import torch_geometric
print("PyG:", torch_geometric.__version__)

# Verify the GNN imports that the hybrid pipeline needs
from torch_geometric.nn import SAGEConv, GCNConv, GraphConv  # noqa
print("SAGEConv/GCNConv/GraphConv import — OK")

# Also ensure xgboost is available (Colab usually has it)
try:
    import xgboost as xgb
    print("xgboost:", xgb.__version__)
except ImportError:
    print("xgboost not found — installing ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xgboost"], check=True)
    import xgboost as xgb
    print("xgboost:", xgb.__version__, "(just installed)")
"""))

# ---------------------------------------------------------------------------
cells.append(md("## Cell 4 — Load dataset + integrity check"))
cells.append(code("""import pandas as pd

# Expected dimensions for the full dataset
EXPECTED_ROWS  = 46338
EXPECTED_WELLS = 11333
KEY_COLS = ["gm_well_id", "latitude", "longitude", "PFOA_ngL"]

df_check = pd.read_parquet(DATA_PATH)
n_rows, n_cols = df_check.shape
n_wells = df_check["gm_well_id"].nunique()
missing_keys = [c for c in KEY_COLS if c not in df_check.columns]

print(f"Dataset: {n_rows} rows x {n_cols} cols  |  {n_wells} unique wells")
print(f"Key columns present: {[c for c in KEY_COLS if c in df_check.columns]}")

if missing_keys:
    raise RuntimeError(
        f"Key columns missing from dataset: {missing_keys}. "
        f"Check DATA_PATH={DATA_PATH} and repo ref GIT_REF={GIT_REF}."
    )

if not SMOKE_TEST:
    if n_rows != EXPECTED_ROWS:
        raise RuntimeError(
            f"Row count mismatch: got {n_rows}, expected {EXPECTED_ROWS}. "
            "Clone may be stale or DATA_PATH points to wrong file."
        )
    if n_wells != EXPECTED_WELLS:
        raise RuntimeError(
            f"Well count mismatch: got {n_wells}, expected {EXPECTED_WELLS}."
        )
    print(f"Integrity check PASSED ({n_rows} x {n_cols}, {n_wells} wells).")
else:
    print(f"SMOKE_TEST=True — skipping strict row/well count check.")
    print(f"  (got {n_rows} rows, {n_wells} wells — will be subsampled by the driver)")

del df_check   # free memory; the driver reloads via src.data.load()
"""))

# ---------------------------------------------------------------------------
cells.append(md("""## Cell 5 — Run the hybrid pipeline

The driver `experiments/gnn_hybrid_t1/run_hybrid_t1.py` is imported and called directly
so that its `SMOKE_TEST` / `RELATIONS` / parameter toggles can be set from this cell.
All heavy logic stays in `src/hybrid.py`; the notebook only orchestrates parameters and
wires output paths.

Checkpoints are written after each outer fold to
`experiments/gnn_hybrid_t1/run_<relation>/spatial/metrics_incremental.json` and
`experiments/gnn_hybrid_t1/run_<relation>/random/metrics_incremental.json`.
A Colab disconnect after fold `k` loses only fold `k`'s work; folds 0..k-1 are safe.
"""))
cells.append(code("""import json, time, logging, sys, os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)

# Import pipeline modules (available after Cell 2 bootstrap)
from src import config as C
from src import data as D
from src import targets as T
from src import splits as S
from src import hybrid as H
from src import gnn

OUT_DIR = Path("experiments/gnn_hybrid_t1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters: SMOKE_TEST and RELATIONS come from Cell 0
SEED = C.SEED

smoke_n      = 500 if SMOKE_TEST else None
outer_k      = 3   if SMOKE_TEST else FULL_OUTER_K
inner_k      = 2   if SMOKE_TEST else FULL_INNER_K
gnn_epochs   = 15  if SMOKE_TEST else FULL_GNN_EPOCHS
gnn_patience = 6   if SMOKE_TEST else FULL_GNN_PATIENCE
hidden       = 32  if SMOKE_TEST else FULL_HIDDEN

# GNN count and wall-time estimate
n_gnn_per_rel = 2 * outer_k * (inner_k + 1)
n_gnn_total   = n_gnn_per_rel * len(RELATIONS)
print("=" * 70)
print(f"GNN-hybrid T1  SMOKE_TEST={SMOKE_TEST}  seed={SEED}")
print(f"  Relations : {RELATIONS}")
print(f"  outer_k={outer_k}  inner_k={inner_k}  epochs={gnn_epochs}  hidden={hidden}")
print(f"  GNN trainings: {n_gnn_total} total ({n_gnn_per_rel} per relation)")
if not SMOKE_TEST:
    print(f"  Estimated GPU wall time: {n_gnn_total}–{n_gnn_total*2} min "
          f"({n_gnn_total/60:.1f}–{n_gnn_total*2/60:.1f} h) on Colab T4")
    print("  Checkpoints: experiments/gnn_hybrid_t1/run_<rel>/{spatial,random}/"
          "metrics_incremental.json")
print("=" * 70)

# Load data
df = D.load(smoke=SMOKE_TEST, smoke_n=smoke_n)
y_row = T.build_T1a(df).to_numpy()
prevalence = float(y_row.mean())
feature_cols = C.feature_columns(include_location=False, cocontam="core")
print(f"Data: {df.shape}  wells={df[C.WELL_ID].nunique()}  "
      f"prevalence={prevalence:.3f}  features={len(feature_cols)}")

# Run each relation
t_total = time.time()
all_results = {}

for relation in RELATIONS:
    print(f"\\n{'='*60}")
    print(f"RELATION: {relation}")
    print(f"{'='*60}")
    rel_dir = OUT_DIR / f"run_{relation}"

    res = H.run_hybrid_t1(
        df,
        smoke=SMOKE_TEST,
        relation=relation,
        hidden=hidden,
        inner_k=inner_k,
        gnn_max_epochs=gnn_epochs,
        gnn_patience=gnn_patience,
        outer_k=outer_k,
        seed=SEED,
        verbose=SMOKE_TEST,
        out_dir=rel_dir,
    )
    all_results[relation] = res

    sp_auc = res["hybrid_spatial"]["aggregated"].get("roc_auc_mean", float("nan"))
    rd_auc = res["hybrid_random"]["aggregated"].get("roc_auc_mean", float("nan"))
    delta  = rd_auc - sp_auc
    print(f"[{relation}] hybrid spatial AUC={sp_auc:.4f}  "
          f"random AUC={rd_auc:.4f}  delta(random-spatial)={delta:+.4f}")

elapsed_total = time.time() - t_total
print(f"\\nAll relations done in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""## Cell 6 — Three-way comparison + write final metrics.json

Assembles the `(hybrid, gnn_alone, xgb_alone)` triplet for the primary relation
(`subbasin_knn`), applies the reality rule (§4.5): a gain is "real" only if
significant (p<0.05) AND > 0.03 AUC noise floor. Writes `experiments/gnn_hybrid_t1/metrics.json`.
"""))
cells.append(code("""import numpy as np

# GNN-alone reference from phase 2 (subbasin_knn, spatial CV, 8 folds).
# These are the established numbers; update after running a fresh phase-2 if needed.
GNN_ALONE_SPATIAL_AUC  = 0.605   # GraphSAGE subbasin_knn spatial AUC (phase 2 best)
GNN_ALONE_SPATIAL_AUCS = []      # per-fold list (fill from gnn_phase2 if available)

# XGBoost-alone wall (spatial CV, 8 folds, from experiments/baseline_t1_smoke).
XGB_WALL_SPATIAL_AUC_MEAN = 0.588
XGB_WALL_SPATIAL_AUCS     = []

primary = "subbasin_knn"
if primary not in all_results:
    print(f"WARNING: primary relation '{primary}' was not run. "
          "Add it to RELATIONS and re-run Cell 5.")
else:
    spatial_cv = all_results[primary]["hybrid_spatial"]
    random_cv  = all_results[primary]["hybrid_random"]

    gnn_spatial_stub = {"auc_mean": GNN_ALONE_SPATIAL_AUC,
                        "per_fold_auc": GNN_ALONE_SPATIAL_AUCS}
    gnn_random_stub  = {"auc_mean": float("nan")}

    comparison = H.run_three_way_comparison(
        spatial_cv, random_cv,
        gnn_spatial_stub, gnn_random_stub,
        xgb_spatial_auc_mean=XGB_WALL_SPATIAL_AUC_MEAN,
        xgb_spatial_aucs=XGB_WALL_SPATIAL_AUCS,
        noise_threshold=0.03,
    )

    print("\\n" + "=" * 70)
    print("THREE-WAY COMPARISON (primary relation: subbasin_knn)")
    print("=" * 70)
    for arm, tri in comparison["triplets"].items():
        rd_str  = f"{tri['random']:.4f}"  if np.isfinite(tri['random'])  else "n/a"
        dlt_str = f"{tri['delta']:+.4f}" if np.isfinite(tri['delta'])  else "n/a"
        print(f"  {arm:<12}  spatial={tri['spatial']:.4f}  "
              f"random={rd_str:>8}  delta={dlt_str:>8}")

    rv = comparison["reality_rule"]
    print(f"\\n  Hybrid gain over XGB wall: {rv['hybrid_gain_over_xgb_wall']:+.4f}")
    print(f"  Significant (p<0.05): {rv['significant']}")
    print(f"  Above noise threshold ({rv['noise_threshold']}): {rv['above_noise_threshold']}")
    print(f"  Verdict: {rv['verdict']}")
    if not SMOKE_TEST:
        print("\\n  NOTE: embedding-axis misalignment caveat applies (see REPORT.md §4).")
        print("  The hybrid gain may be underestimated vs a jointly-trained architecture.")

    # Build final metrics output
    metrics_out = {
        "smoke": SMOKE_TEST,
        "config": {
            "outer_k": outer_k, "inner_k": inner_k,
            "gnn_epochs": gnn_epochs, "gnn_patience": gnn_patience,
            "hidden": hidden, "relations": RELATIONS,
            "feature_cols_count": len(feature_cols),
            "prevalence": float(prevalence),
            "seed": SEED,
            "elapsed_total_s": elapsed_total,
        },
        "per_relation": {
            rel: {
                "spatial": {
                    "aggregated": res["hybrid_spatial"]["aggregated"],
                    "global_oof_auc": res["hybrid_spatial"]["global_oof_auc"],
                    "bootstrap_ci": res["hybrid_spatial"]["bootstrap_ci_by_well"],
                    "per_fold": res["hybrid_spatial"]["per_fold"],
                },
                "random": {
                    "aggregated": res["hybrid_random"]["aggregated"],
                    "global_oof_auc": res["hybrid_random"]["global_oof_auc"],
                },
                "delta_auc": (
                    res["hybrid_random"]["aggregated"].get("roc_auc_mean", float("nan"))
                    - res["hybrid_spatial"]["aggregated"].get("roc_auc_mean", float("nan"))
                ),
                "elapsed_s": res["elapsed_s"],
            }
            for rel, res in all_results.items()
        },
        "three_way_comparison": comparison,
    }

    out_path = OUT_DIR / "metrics.json"
    with open(out_path, "w") as fh:
        json.dump(metrics_out, fh, indent=2, default=str)
    print(f"\\nmetrics.json written -> {out_path}")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""## Cell 7 — Persist outputs (Colab workspace is EPHEMERAL — no Drive)

**WARNING**: all files written to `/content/pfas-gnn/experiments/gnn_hybrid_t1/` are
**lost when the Colab runtime disconnects or times out**. Use one or both options below to
preserve your results. This cell is idempotent; re-run anytime.
"""))
cells.append(code("""import shutil
from pathlib import Path

OUT_DIR = Path("experiments/gnn_hybrid_t1")

# ---- Option A: download a zip archive to your local machine ----
arch = shutil.make_archive("gnn_hybrid_t1_outputs", "zip", str(OUT_DIR))
print(f"Archive created: {arch}")
if IN_COLAB:
    from google.colab import files
    files.download(arch)
    print("Download triggered. Save the zip to keep your results.")
else:
    print(f"(Not in Colab — archive at {arch}; copy manually.)")

# ---- Option B (preferred): commit & push results to the repo ----
# Uncomment and run the lines below to push metrics + incremental checkpoints.
# You will need push access to REPO_URL.
#
# import subprocess
# subprocess.run(["git", "add",
#     "experiments/gnn_hybrid_t1/metrics.json",
#     "experiments/gnn_hybrid_t1/config.yaml",
#     "experiments/gnn_hybrid_t1/run_subbasin_knn/",
#     "experiments/gnn_hybrid_t1/run_spatial/",
# ], check=True)
# subprocess.run(["git", "commit", "-m",
#     f"feat(hybrid-t1): full run SMOKE={SMOKE_TEST} relations={RELATIONS}"], check=True)
# subprocess.run(["git", "push"], check=True)
# print("Results committed and pushed to", REPO_URL)

print("\\nWARNING: without download or git push, ALL outputs are lost on runtime disconnect.")
"""))

# ---------------------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "gnn_hybrid_t1_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
