"""Generator for notebooks/gnn_phase2_colab.ipynb — autonomous Colab notebook (CLAUDE.md §4).
Run: python3 notebooks/gnn_phase2_colab.ipynb.py  -> writes the .ipynb next to it."""
import json
from pathlib import Path

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# PFAS Groundwater — GNN Phase 2 (P0: T1 robust early-stop · P1: T2 bipartite completion)

**AUTONOMOUS** (CLAUDE.md §4): bootstraps `src/` + the versioned dataset via `git clone`
(no Google Drive), installs **PyTorch Geometric** for the Colab torch wheel, then runs:

- **P0 — T1a**: GraphSAGE / GCN with a ROBUST early-stop (validation = several assembled
  spatial micro-blocks of the train wells, + ReduceLROnPlateau + more patience). Reports the
  honest **triplet (random, spatial, Δ)** vs the non-graph WALL (RF spatial 0.601) and vs
  phase-1 (GraphSAGE 0.618 / GCN 0.624).
- **P1 — T2**: matrix completion on the **bipartite wells × analytes** graph (MNAR-aware).
  Inductive link-LABEL prediction; spatial-block CV at the well level + random for Δ;
  per-label measurement mask; the 5 multilabel metrics. Compares to the T2 WALL
  (BinaryRelevance macro-AUROC spatial **0.680**).

All leakage controls of `EVAL_PROTOCOL.md` (C2 group-by-well, C4 no cross-block edges) hold.
Outputs are written **into the cloned workspace** under `experiments/gnn_phase2/` (no Drive);
the last cell offers explicit persistence (download / git push) because Colab is ephemeral.
"""))

cells.append(md("## Cell 0 — User parameters"))
cells.append(code("""SMOKE_TEST = False        # True = fast CPU sanity; False = full GPU run

REPO_URL = "https://github.com/dnwiloic/pfas-gnn.git"
GIT_REF  = "main"
DATA_PATH = "data/CA-PFAS-ASGWS.parquet"

RUN_P0 = True             # T1 robust early-stop
RUN_P1 = True             # T2 bipartite completion
N_BLOCKS = 8              # spatial KMeans blocks (reference CV)
"""))

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
    print("torch   :", torch.__version__, " CUDA avail:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU     :", torch.cuda.get_device_name(0))
except ImportError:
    print("torch not yet installed (Colab base usually has it).")
"""))

cells.append(md("## Cell 2 — Clone repo (code + versioned dataset), no Drive"))
cells.append(code("""import os, subprocess
REPO_DIR = "/content/pfas-gnn" if IN_COLAB else os.getcwd()
if IN_COLAB:
    if not os.path.isdir(REPO_DIR):
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", GIT_REF], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
assert os.path.exists(DATA_PATH), f"dataset missing at {DATA_PATH} — clone failed"
print("workdir:", os.getcwd(), "| dataset present:", os.path.exists(DATA_PATH))
"""))

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
    print("installing PyG from", idx)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch_scatter", "torch_sparse", "-f", idx], check=False)
ensure_pyg()
import torch_geometric; print("PyG:", torch_geometric.__version__)
"""))

cells.append(md("## Cell 4 — Integrity check: dataset shape & socle import"))
cells.append(code("""from src import config as C
from src import data, gnn, gnn_bipartite as GB, graph as G, splits as S, targets as T
df = data.load(smoke=SMOKE_TEST, smoke_n=1500)
n_wells = df[C.WELL_ID].nunique()
print(f"rows={len(df)}  wells={n_wells}")
if not SMOKE_TEST:
    assert len(df) == 46338, f"unexpected row count {len(df)}"
    assert n_wells == 11333, f"unexpected well count {n_wells}"
print("OK — socle imported, dataset integrity verified.")
"""))

cells.append(md("""## Cell 5 — P0: T1a GraphSAGE / GCN with robust early-stop

Validation = several assembled spatial micro-blocks of the TRAIN wells (replaces phase-1's
single-block holdout that under-trained 2-3 folds) + ReduceLROnPlateau + patience 50.
Reports the triplet vs the WALL. C4 audited (0 cross-block edges remaining)."""))
cells.append(code("""import json, time
from pathlib import Path
OUT = Path("experiments/gnn_phase2"); OUT.mkdir(parents=True, exist_ok=True)

if RUN_P0:
    nb = 3 if SMOKE_TEST else N_BLOCKS
    common = dict(k=8, cap_km=1.5, cut_blocks=True, encode="frequency",
                  hidden=(32 if SMOKE_TEST else 64), layers=2, dropout=0.5,
                  lr=5e-3, n_val_micro=(4 if SMOKE_TEST else 8), val_frac=0.18,
                  lr_schedule=True, max_epochs=(50 if SMOKE_TEST else 400),
                  patience=(20 if SMOKE_TEST else 50))
    t0 = time.time(); p0 = {}
    for model in ["graphsage", "gcn"]:
        sp, _ = gnn.run_t1_cv(df, model_name=model, regime="spatial", n_blocks=nb, **common)
        rd, _ = gnn.run_t1_cv(df, model_name=model, regime="random", n_blocks=nb, **common)
        p0[model] = {"auc_spatial": sp["auc_mean"], "auc_spatial_std": sp["auc_std"],
                     "auc_random": rd["auc_mean"],
                     "delta": rd["auc_mean"] - sp["auc_mean"],
                     "per_fold_spatial": sp["per_fold_auc"],
                     "per_fold_best_epoch_spatial": sp["per_fold_best_epoch"],
                     "total_removed_cross_block": sp["total_removed_cross_block"]}
        (OUT / "metrics_p0_incremental.json").write_text(json.dumps(p0, indent=2))
        print(f"{model}: spatial={sp['auc_mean']:.4f}±{sp['auc_std']:.3f} "
              f"random={rd['auc_mean']:.4f} Δ={p0[model]['delta']:.4f} "
              f"best_ep={sp['per_fold_best_epoch']}")
    p0_out = {"task": "T1a", "phase": "P0", "smoke": SMOKE_TEST, "models": p0,
              "wall": {"RF_spatial": 0.601, "phase1_graphsage": 0.618, "phase1_gcn": 0.624},
              "elapsed_min": round((time.time()-t0)/60, 2)}
    (OUT / "metrics_p0.json").write_text(json.dumps(p0_out, indent=2))
    print("wrote metrics_p0.json in", p0_out["elapsed_min"], "min")
"""))

cells.append(md("""## Cell 6 — P1: T2 bipartite matrix completion vs the wall (0.680)

Inductive link-LABEL prediction on the wells × analytes graph; spatial-block CV at the well
level + random for Δ; per-label measurement mask; the 5 multilabel metrics. C4 audited."""))
cells.append(code("""def _strip(res): return {k: v for k, v in res.items()
                          if k not in ("P_row", "Y_row", "M_row")}
if RUN_P1:
    nb = 3 if SMOKE_TEST else N_BLOCKS
    common = dict(emb_dim=(16 if SMOKE_TEST else 32), hidden=(32 if SMOKE_TEST else 64),
                  layers=2, dropout=0.3, lr=5e-3,
                  max_epochs=(40 if SMOKE_TEST else 300), patience=(15 if SMOKE_TEST else 40))
    t0 = time.time()
    sp = GB.run_t2_bipartite_cv(df, regime="spatial", n_blocks=nb, **common)
    rd = GB.run_t2_bipartite_cv(df, regime="random", n_blocks=nb, **common)
    delta = rd["macro_AUROC"] - sp["macro_AUROC"]
    p1_out = {"task": "T2", "phase": "P1", "model": "BipartiteCompletion",
              "smoke": SMOKE_TEST, "spatial": _strip(sp), "random": _strip(rd),
              "delta_macro_AUROC": delta,
              "wall": {"BR_macro_AUROC_spatial": 0.680, "BR_macro_AUROC_random": 0.902,
                       "BR_delta": 0.222},
              "elapsed_min": round((time.time()-t0)/60, 2)}
    (OUT / "metrics_p1.json").write_text(json.dumps(p1_out, indent=2))
    print(f"P1 spatial macro_AUROC={sp['macro_AUROC']:.4f} (wall 0.680) "
          f"random={rd['macro_AUROC']:.4f} Δ={delta:.4f} "
          f"cross_block_edges={sp['n_cross_block_edges']}")
    import pandas as pd
    print(pd.DataFrame(sp["per_label"])[["label","n_measured","prevalence","roc_auc","f1"]]
          .to_string(index=False))
"""))

cells.append(md("## Cell 7 — Persist outputs (Colab is ephemeral — no Drive)"))
cells.append(code("""# Option A: download an archive of experiments/gnn_phase2/
import shutil
arch = shutil.make_archive("gnn_phase2_outputs", "zip", "experiments/gnn_phase2")
print("archived ->", arch)
if IN_COLAB:
    from google.colab import files
    files.download(arch)
# Option B (preferred): commit & push the metrics back to the repo.
# !git -C {REPO_DIR} add experiments/gnn_phase2 && git -C {REPO_DIR} commit -m "phase2 run" && git -C {REPO_DIR} push
print("WARNING: without download or git push, these outputs are lost on disconnect.")
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parent / "gnn_phase2_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
