"""Generator for notebooks/gnn_phase3_colab.ipynb — autonomous Colab notebook (CLAUDE.md §4).
Run: python3 notebooks/gnn_phase3_colab.ipynb.py  -> writes the .ipynb next to it.

P1+ — T2 HETEROGENEOUS matrix completion: bipartite wells<->analytes AUGMENTED with capped
spatial well<->well edges. Sweeps encoders {hetero_sage, hgt, rgcn} x decoders {mlp, vgae}
on the spatial reference CV, runs the random regime for the Δ on the best, audits C4 on every
fold (0 cross-block bipartite AND 0 cross-block well edges)."""
import json
from pathlib import Path

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# PFAS Groundwater — GNN Phase 3 (P1+: T2 heterogeneous matrix completion)

**AUTONOMOUS** (CLAUDE.md §4): bootstraps `src/` + the versioned dataset via `git clone`
(no Google Drive), installs **PyTorch Geometric** for the Colab torch wheel, then runs P1+.

**P1+** extends P1 (bipartite completion, spatial macro-AUROC **0.681** = wall 0.680). It
adds the relational signal HYDRO_CRITIQUE asked for: a **heterogeneous** graph with two node
types (wells with context features + analytes with a learned embedding) and **three
relations** — bipartite `analyte<->well` (measured cells, MNAR mask) and spatial
`well<->well` (k-NN, **hard cap 1.5 km**, symmetrised). So a test well borrows strength from
its *measured spatial neighbours'* analyte patterns, not only from its own cells.

Encoders swept: **hetero_sage** / **HGT** / **R-GCN**. Decoders: **MLP** (P1-style) /
**VGAE** (variational + KL, probabilistic completion). Focal loss (γ=1) targets the rare
PFNA (report the **AP**, not just AUROC).

**Leakage controls (audited, never presumed):** outer spatial-block CV at the well level
(C2/C3); the bipartite edges touch one well = one block (C4 by construction);
the **well<->well edges are CUT per fold** (`cut_cross_block`) and message passing is
**train-only** (inductive) — we ASSERT 0 cross-block bipartite AND 0 cross-block well edges
on every fold. Outputs go to `experiments/gnn_phase3/` (no Drive); the last cell persists.
"""))

cells.append(md("## Cell 0 — User parameters"))
cells.append(code("""SMOKE_TEST = False        # True = fast CPU sanity; False = full GPU run

REPO_URL = "https://github.com/dnwiloic/pfas-gnn.git"
GIT_REF  = "main"
DATA_PATH = "data/CA-PFAS-ASGWS.parquet"

N_BLOCKS  = 8             # spatial KMeans blocks (reference CV) + random folds (Δ)
MAX_EPOCHS = 300          # full-scale completion converged near 290 on CPU
# the sweep: (encoder, decoder) pairs evaluated on the SPATIAL reference CV
SWEEP = [("hetero_sage", "mlp"), ("hgt", "mlp"), ("rgcn", "mlp"), ("hetero_sage", "vgae")]
RANDOM_FOR = ("hetero_sage", "mlp")   # also run the random regime (Δ) for this pair
FOCAL_GAMMA = 1.0
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
import torch_geometric
from torch_geometric.nn import HGTConv, RGCNConv, HeteroConv, SAGEConv  # noqa
print("PyG:", torch_geometric.__version__, "— HGT/RGCN/HeteroConv import OK")
"""))

cells.append(md("## Cell 4 — Integrity check: dataset shape & socle import"))
cells.append(code("""from src import config as C
from src import data, gnn_hetero as GH, gnn_bipartite as GB, graph as G, splits as S
df = data.load(smoke=SMOKE_TEST, smoke_n=1500)
n_wells = df[C.WELL_ID].nunique()
print(f"rows={len(df)}  wells={n_wells}")
if not SMOKE_TEST:
    assert len(df) == 46338, f"unexpected row count {len(df)}"
    assert n_wells == 11333, f"unexpected well count {n_wells}"
print("OK — socle imported, dataset integrity verified.")
"""))

cells.append(md("""## Cell 5 — P1+ sweep on the SPATIAL reference CV (the honest metric)

Each (encoder, decoder) is trained with the per-label measurement mask, focal γ, and the
well<->well edges CUT per fold. We report macro-AUROC + micro-F1 + the C4 audit, and write an
incremental checkpoint after each pair so a disconnect never loses finished work."""))
cells.append(code("""import json, time
import pandas as pd
from pathlib import Path
OUT = Path("experiments/gnn_phase3"); OUT.mkdir(parents=True, exist_ok=True)

def _strip(res): return {k: v for k, v in res.items()
                          if k not in ("P_row", "Y_row", "M_row")}

nb = 3 if SMOKE_TEST else N_BLOCKS
common = dict(emb_dim=(16 if SMOKE_TEST else 32), hidden=(32 if SMOKE_TEST else 64),
              layers=2, dropout=0.3, heads=2, k=8, cap_km=1.5, lr=5e-3,
              gamma=FOCAL_GAMMA, max_epochs=(40 if SMOKE_TEST else MAX_EPOCHS),
              patience=(15 if SMOKE_TEST else 40))

t0 = time.time(); sweep = {}
for enc, dec in SWEEP:
    beta = 1e-3 if dec == "vgae" else 0.0
    sp = GH.run_t2_hetero_cv(df, regime="spatial", n_blocks=nb, encoder=enc, decoder=dec,
                             beta_kl=beta, **common)
    key = f"{enc}/{dec}"
    sweep[key] = _strip(sp)
    (OUT / "metrics_p1plus_incremental.json").write_text(json.dumps(sweep, indent=2))
    print(f"[spatial] {key:18s} macro_AUROC={sp['macro_AUROC']:.4f} micro_F1={sp['micro_F1']:.4f} "
          f"cross_bip={sp['n_cross_block_bipartite']} cross_well={sp['n_cross_block_well']} "
          f"removed_well={sp['n_removed_well_cross_total']}")
print(f"sweep done in {(time.time()-t0)/60:.1f} min")
"""))

cells.append(md("""## Cell 6 — Random regime (Δ) for the reference pair + per-label table

The random-minus-spatial Δ is the spatial-leakage guard (a variant that lifts random without
spatial is leakage to reject). Per-label AUROC **and AP** vs the wall and vs P1 bipartite."""))
cells.append(code("""enc, dec = RANDOM_FOR
beta = 1e-3 if dec == "vgae" else 0.0
sp = GH.run_t2_hetero_cv(df, regime="spatial", n_blocks=nb, encoder=enc, decoder=dec,
                         beta_kl=beta, **common)
rd = GH.run_t2_hetero_cv(df, regime="random", n_blocks=nb, encoder=enc, decoder=dec,
                         beta_kl=beta, **common)
delta = rd["macro_AUROC"] - sp["macro_AUROC"]

WALL_PL = {"PFOS":(0.588,0.450),"PFBS":(0.632,0.487),"PFHxA":(0.656,0.519),"PFOA":(0.665,0.443),
           "PFHpA":(0.634,0.379),"PFBA":(0.728,0.592),"PFPeA":(0.689,0.559),"PFHxS":(0.660,0.288),
           "PFPeS":(0.721,0.367),"PFNA":(0.831,0.169)}
P1_PL = {"PFOS":0.638,"PFBS":0.641,"PFHxA":0.669,"PFOA":0.663,"PFHpA":0.672,"PFBA":0.706,
         "PFPeA":0.698,"PFHxS":0.657,"PFPeS":0.702,"PFNA":0.766}
pl = pd.DataFrame(sp["per_label"])
pl["wall_AUROC"] = pl["label"].map(lambda a: WALL_PL[a][0])
pl["wall_AP"]    = pl["label"].map(lambda a: WALL_PL[a][1])
pl["p1_AUROC"]   = pl["label"].map(P1_PL)
pl["d_vs_wall"]  = pl["roc_auc"] - pl["wall_AUROC"]
pl["d_vs_p1"]    = pl["roc_auc"] - pl["p1_AUROC"]
print(f"P1+ [{enc}/{dec}] TRIPLET: spatial={sp['macro_AUROC']:.4f}  random={rd['macro_AUROC']:.4f}"
      f"  Δ={delta:.4f}  | wall 0.680 / P1 0.681 ; wall Δ 0.222 / P1 Δ 0.162")
print(pl[["label","prevalence","roc_auc","pr_auc","wall_AUROC","wall_AP","p1_AUROC",
          "d_vs_wall","d_vs_p1"]].round(3).to_string(index=False))

out = {"task":"T2","phase":"P1+","smoke":SMOKE_TEST,"seed":C.SEED,"n_blocks":nb,
       "focal_gamma":FOCAL_GAMMA,
       "reference_pair":f"{enc}/{dec}",
       "spatial":_strip(sp),"random":_strip(rd),
       "delta_random_minus_spatial_macro_AUROC":delta,
       "sweep_spatial":sweep,
       "wall":{"BR_macro_AUROC_spatial":0.680,"BR_macro_AUROC_random":0.902,"BR_delta":0.222,
               "BR_micro_F1_spatial":0.542},
       "p1_bipartite":{"macro_AUROC_spatial":0.681,"macro_AUROC_random":0.843,"delta":0.162},
       "elapsed_min":round((time.time()-t0)/60,2)}
(OUT/"metrics_p1plus.json").write_text(json.dumps(out, indent=2))
print("\\nwrote experiments/gnn_phase3/metrics_p1plus.json")
"""))

cells.append(md("## Cell 7 — Persist outputs (Colab is ephemeral — no Drive)"))
cells.append(code("""import shutil
arch = shutil.make_archive("gnn_phase3_outputs", "zip", "experiments/gnn_phase3")
print("archived ->", arch)
if IN_COLAB:
    from google.colab import files
    files.download(arch)
# Preferred: commit & push the metrics back to the repo.
# !git -C {REPO_DIR} add experiments/gnn_phase3 && git -C {REPO_DIR} commit -m "phase3 P1+ run" && git -C {REPO_DIR} push
print("WARNING: without download or git push, these outputs are lost on disconnect.")
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parent / "gnn_phase3_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
