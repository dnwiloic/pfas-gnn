"""T1 by MULTI-RELATIONAL encoders (HGT / R-GCN) on a HOMOGENEOUS well-well graph
with TWO edge types.

Why this module exists (and how it differs from src/gnn.py)
----------------------------------------------------------
`src/gnn.py` runs SINGLE-relation homogeneous GNNs (GraphSAGE / GCN / GraphConv):
ONE edge type at a time (either the bare spatial k-NN, OR the mechanistic intra-
sub-basin k-NN), so a layer mixes one neighbourhood. This module keeps the SAME node
set (one real node type, the WELL) but feeds the model BOTH relations at once as
DISTINCT edge types, and uses MULTI-RELATIONAL encoders that route messages per
relation:

  * relation R_NEAR     = ('well', 'near', 'well')          — bare spatial k-NN, cap 1.5 km
  * relation R_SUBBASIN = ('well', 'same_subbasin_knn', 'well')
                                                            — k-NN RESTRICTED to wells that
                                                              share `sgma_subbasin_name`, cap 2 km

The two relations live on the SAME well nodes (no second node type), so this is a
homogeneous multigraph, encoded with relational architectures:

  * name='rgcn'        — RGCNConv(num_relations=2): one weight matrix per relation,
                         summed. The two relations share the well node space, so no node
                         re-indexing is needed (unlike gnn_hetero.py's bipartite RGCN view).
  * name='hgt'         — HGTConv with metadata=(['well'], [R_NEAR, R_SUBBASIN]): typed
                         per-relation attention. Valid on ONE real node type — the
                         "hetero" machinery is used purely for the two edge types.
  * name='hetero_sage' — HeteroConv({R_NEAR: SAGEConv, R_SUBBASIN: SAGEConv}, aggr='sum'):
                         per-relation SAGE message passing, ablation/comparison baseline.

This is the eval-methodologist's APPROVED design (experiments/hgt_rgcn_t1/
eval_validation.md): the rejected "source/installation" node type does NOT exist in the
data (C-NODE.1/2); HGT and R-GCN are admissible only as RELATIONAL encoders over a
single real node type (C-NODE.3).

Methodological conditions enforced here (eval_validation.md, all asserted/reported)
-----------------------------------------------------------------------------------
  C-SPAT.1  spatial-block CV at the WELL level (splits.spatial_block_folds, k=8).
  C-SPAT.2  cross-block edges CUT per fold + assert 0 residual.
  C-SPAT.3  hard distance cap per relation (1.5 km spatial / 2 km sub-basin).
  C-SPAT.4  INDUCTIVE on labels: message passing uses TRAIN-TRAIN edges only during
            training (a test well never injects messages while we fit). The full
            (train+test, cross-block-free) edge set is used only at scoring time so a
            test well aggregates from its TRAIN neighbours through cut edges.
  C-SPAT.5  the inter-block cut is applied and asserted SEPARATELY PER RELATION — a
            single un-cut relation reopens the leak (HeteroConv/HGT route per edge type).
  C-SPAT.6  the runner reports the random-minus-spatial AUC Δ when both regimes run.
  C-LOC.1   feature_cols exclude lat/lon by default (include_location=False).
  C-THR     the F1 threshold is fit on VAL/OOF probabilities only, never on test.
  C-CAL     Brier + ECE + a reliability curve are reported on the aggregated OOF probas.
  C-MET     same metric set / k blocks / caps as experiments/gnn_phase1 and the wall.
  Seed fixed everywhere (config.SEED).

Evaluation, exactly like src/gnn.py, is at the SAMPLING (row) level for strict
comparability with the non-graph wall: a well-node probability is broadcast to every
sampling row of that well, then row-level binary metrics use the row target. Grouping
(C2) is automatic (one well = one node = one block).

Torch is lazily imported so the module imports without a GPU; it is CPU smoke-testable
on a few hundred wells with a single fold (`run_t1_multirel_cv(..., smoke=True)`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import graph as G
from . import metrics as M
from . import splits as S
from . import targets as T

# Canonical relation triples (one real node type 'well', two edge types).
R_NEAR = ("well", "near", "well")
R_SUBBASIN = ("well", "same_subbasin_knn", "well")
RELATIONS = (R_NEAR, R_SUBBASIN)
REL_NAMES = ("near", "same_subbasin_knn")


# ------------------------------------------------------------------ torch helpers
def set_seed(seed: int = C.SEED):
    import random
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================================
# 1. MULTI-RELATIONAL GRAPH CONSTRUCTION (homogeneous well-well, two edge types)
# =====================================================================================
@dataclass
class MultiRelGraph:
    """Two-relation homogeneous well graph (numpy/pandas only — torch-free, cheap import).

    `rel` maps each relation triple -> (edge_index[2,E] int64, edge_dist_km[E]), already
    cross-block-cut (C-SPAT.2) and symmetrised (bidirectional). `audit` carries the per-
    relation cross-block counts (all 0 after the cut) for the assertions / report.
    """
    well_ids: np.ndarray
    coords: np.ndarray
    node_block: np.ndarray                 # CV block id per well node
    rel: dict                              # {triple: (edge_index, edge_dist)}
    row_to_node: np.ndarray               # [n_rows] node index of each sampling row
    audit: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def build_multirel_graph(df, well_ids, coords, subbasin, node_block, *,
                         k_spatial=8, cap_km_spatial=1.5,
                         k_subbasin=8, cap_km_subbasin=2.0):
    """Build the two-relation well graph and CUT each relation across CV blocks SEPARATELY.

    Parameters
    ----------
    df          : row-level frame (only used for the row->node map via WELL_ID).
    well_ids    : per-well id array (node order); from graph.well_table.
    coords      : per-well [lat, lon] in well_ids order.
    subbasin    : per-well SGMA sub-basin label (object array) in well_ids order; from
                  graph.well_subbasin. Wells with a missing sub-basin get NO R_SUBBASIN edge.
    node_block  : per-well CV block id (int) in well_ids order.

    Returns a `MultiRelGraph`. Both relations are cut by `graph.cut_cross_block` and
    asserted to have 0 cross-block edges (C-SPAT.2/C-SPAT.5), then symmetrised.
    """
    # relation 1: bare spatial k-NN, cap 1.5 km (the gnn.py 'spatial' relation)
    ei_near, ed_near = G.knn_edges_km(coords, k=k_spatial, cap_km=cap_km_spatial)
    # relation 2: intra-sub-basin k-NN, cap 2 km (the gnn.py 'subbasin_knn' relation)
    ei_sub, ed_sub = G.knn_edges_intra_subbasin(coords, subbasin, k=k_subbasin,
                                                cap_km=cap_km_subbasin)

    # C-SPAT.5: cut EACH relation separately across blocks, BEFORE symmetrising.
    ei_near, ed_near, rm_near = G.cut_cross_block(ei_near, ed_near, node_block)
    ei_sub, ed_sub, rm_sub = G.cut_cross_block(ei_sub, ed_sub, node_block)

    # C-SPAT.2: assert 0 residual cross-block edge per relation (before symmetrise; the
    # symmetrise step only mirrors surviving edges, so 0 stays 0).
    n_cross_near = int((node_block[ei_near[0]] != node_block[ei_near[1]]).sum())
    n_cross_sub = int((node_block[ei_sub[0]] != node_block[ei_sub[1]]).sum())
    assert n_cross_near == 0, f"{n_cross_near} cross-block edges remain in R_NEAR (leak)"
    assert n_cross_sub == 0, f"{n_cross_sub} cross-block edges remain in R_SUBBASIN (leak)"

    ei_near, ed_near = G.symmetrise(ei_near, ed_near)
    ei_sub, ed_sub = G.symmetrise(ei_sub, ed_sub)

    well_to_node = {w: i for i, w in enumerate(well_ids)}
    row_to_node = df[C.WELL_ID].map(well_to_node).to_numpy().astype(np.int64)

    audit = {
        "n_removed_cross_block_near": int(rm_near),
        "n_removed_cross_block_subbasin": int(rm_sub),
        "n_cross_block_near": n_cross_near,            # MUST be 0
        "n_cross_block_subbasin": n_cross_sub,         # MUST be 0
        "n_edges_near_directed": int(ei_near.shape[1]),
        "n_edges_subbasin_directed": int(ei_sub.shape[1]),
        "n_wells_missing_subbasin": int(pd.isna(subbasin).sum()),
    }
    meta = {"k_spatial": k_spatial, "cap_km_spatial": cap_km_spatial,
            "k_subbasin": k_subbasin, "cap_km_subbasin": cap_km_subbasin,
            "n_nodes": int(len(well_ids))}
    return MultiRelGraph(well_ids=well_ids, coords=coords, node_block=node_block,
                         rel={R_NEAR: (ei_near, ed_near), R_SUBBASIN: (ei_sub, ed_sub)},
                         row_to_node=row_to_node, audit=audit, meta=meta)


# =====================================================================================
# 2. MODELS (multi-relational encoders over a single real node type)
# =====================================================================================
def build_model_t1(name, in_dim, *, hidden=64, layers=2, dropout=0.3, heads=4):
    """Build a multi-relational node classifier returning [n_nodes] logits.

    name in {'hgt','rgcn','hetero_sage'}. All share: an input Linear to `hidden`, `layers`
    relational conv blocks (LayerNorm + ReLU + dropout, skip connection when in_dim==hidden
    on the first block / always between hidden blocks), and a Linear(hidden,1) head.

    forward signatures:
      * hgt / hetero_sage : forward(x_dict, edge_index_dict) -> logits[n_nodes]
        where x_dict = {'well': X}, edge_index_dict = {R_NEAR: ei, R_SUBBASIN: ei}.
      * rgcn              : forward(x, edge_index, edge_type) -> logits[n_nodes]
        (merged edge list of both relations with a 0/1 relation type vector).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    from torch_geometric.nn import HeteroConv, HGTConv, RGCNConv, SAGEConv

    name = name.lower()

    class MultiRelGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.name = name
            self.hidden = hidden
            self.dropout = dropout
            self.in_proj = nn.Linear(in_dim, hidden)
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(layers):
                if name == "hgt":
                    meta = (["well"], list(RELATIONS))
                    self.convs.append(HGTConv(hidden, hidden, meta, heads=heads))
                elif name == "rgcn":
                    self.convs.append(RGCNConv(hidden, hidden, num_relations=len(RELATIONS),
                                               aggr="mean"))
                elif name == "hetero_sage":
                    self.convs.append(HeteroConv({
                        R_NEAR: SAGEConv(hidden, hidden, aggr="mean"),
                        R_SUBBASIN: SAGEConv(hidden, hidden, aggr="mean"),
                    }, aggr="sum"))
                else:
                    raise ValueError(f"unknown model {name!r}")
                self.norms.append(nn.LayerNorm(hidden))
            self.head = nn.Linear(hidden, 1)

        # ----------------------------------------------------------- hetero path
        def _embed_hetero(self, x_dict, edge_index_dict):
            h = Fn.relu(self.in_proj(x_dict["well"]))
            for conv, norm in zip(self.convs, self.norms):
                xd = {"well": h}
                out = conv(xd, edge_index_dict)["well"]
                out = Fn.relu(norm(out))
                out = out + h                       # skip connection (hidden==hidden)
                h = Fn.dropout(out, p=self.dropout, training=self.training)
            return h

        # ----------------------------------------------------------- rgcn path
        def _embed_rgcn(self, x, edge_index, edge_type):
            h = Fn.relu(self.in_proj(x))
            for conv, norm in zip(self.convs, self.norms):
                out = conv(h, edge_index, edge_type)
                out = Fn.relu(norm(out))
                out = out + h
                h = Fn.dropout(out, p=self.dropout, training=self.training)
            return h

        def embed(self, *args):
            if self.name == "rgcn":
                return self._embed_rgcn(*args)
            return self._embed_hetero(*args)

        def forward(self, *args, embed=False):
            h = self.embed(*args)
            if embed:
                return h
            return self.head(h).squeeze(-1)

        @property
        def embed_dim(self):
            return self.head.in_features

    return MultiRelGNN()


# =====================================================================================
# calibration / threshold / CI helpers (self-contained, OOF-only)
# =====================================================================================
def _f1_threshold(y, p):
    from sklearn.metrics import f1_score
    if len(np.unique(y)) < 2:
        return 0.5
    grid = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def _ece(y_true, proba, n_bins=10):
    """Expected Calibration Error (uniform-width bins), same definition as hybrid._ece."""
    y_true = np.asarray(y_true); proba = np.asarray(proba)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(float(y_true[mask].mean()) - float(proba[mask].mean()))
    return float(ece)


def _reliability_curve(y_true, proba, n_bins=10):
    """Return per-bin (mean_conf, frac_pos, count) for the reliability diagram (C-CAL)."""
    y_true = np.asarray(y_true); proba = np.asarray(proba)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() == 0:
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": 0,
                         "mean_conf": float("nan"), "frac_pos": float("nan")})
        else:
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(mask.sum()),
                         "mean_conf": float(proba[mask].mean()),
                         "frac_pos": float(y_true[mask].mean())})
    return rows


def _bootstrap_ci_auc(y_true, proba, group, *, n_boot=1000, alpha=0.05, seed=C.SEED):
    """IC95% bootstrap on the OOF AUC, RESAMPLED BY GROUP (well) so pseudo-replicates of
    a well stay together (avoids falsely tight intervals). Matches hybrid._bootstrap_ci_by_well.
    """
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true); proba = np.asarray(proba); group = np.asarray(group)
    rng = np.random.RandomState(seed)
    uniq = np.unique(group)
    idx_by_g = {g: np.where(group == g)[0] for g in uniq}
    boot = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in gs])
        yb, pb = y_true[rows], proba[rows]
        if len(np.unique(yb)) < 2:
            continue
        try:
            boot.append(float(roc_auc_score(yb, pb)))
        except Exception:
            pass
    if not boot:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    return {"ci_low": float(np.percentile(boot, 100 * alpha / 2)),
            "ci_high": float(np.percentile(boot, 100 * (1 - alpha / 2))),
            "n_boot": len(boot)}


# =====================================================================================
# robust spatial validation (mirrors gnn._robust_val_mask_coords)
# =====================================================================================
def _robust_val_mask_coords(coords, train_nodes_all, *, n_micro=6, val_frac=0.18,
                            seed=C.SEED):
    """Hold out a stratified slice of several spatial micro-blocks of the TRAIN wells as the
    early-stop validation set (P0; same construction as gnn.py)."""
    from sklearn.cluster import KMeans

    train_idx = np.where(train_nodes_all)[0]
    rng = np.random.RandomState(seed)
    if len(train_idx) < max(2 * n_micro, 10):           # tiny smoke fallback
        val_idx = rng.choice(train_idx, size=max(1, int(val_frac * len(train_idx))),
                             replace=False)
        val = np.zeros_like(train_nodes_all); val[val_idx] = True
        return val, 1
    n_micro = int(min(n_micro, max(2, len(train_idx) // 30)))
    km = KMeans(n_clusters=n_micro, random_state=seed, n_init=10)
    micro = km.fit_predict(coords[train_idx])
    val = np.zeros_like(train_nodes_all)
    for b in range(n_micro):
        members = train_idx[micro == b]
        if len(members) == 0:
            continue
        n_hold = max(1, int(round(val_frac * len(members))))
        val[rng.choice(members, size=min(n_hold, len(members)), replace=False)] = True
    return val, n_micro


# =====================================================================================
# 3. ONE FOLD
# =====================================================================================
@dataclass
class FoldResult:
    fold: int
    metrics_spatial: dict
    best_epoch: int
    n_val_micro: int
    n_val_nodes: int
    audit: dict                          # per-relation cross-block audit (must be 0)
    n_edges_near: int
    n_edges_subbasin: int


def _build_edge_tensors(mrg, dev, *, train_only_mask=None):
    """Build the torch edge structures for both relations.

    If `train_only_mask` (bool per node) is given, keep only TRAIN-TRAIN edges per relation
    (C-SPAT.4 inductive message passing). Returns (edge_index_dict, rgcn_edge_index,
    rgcn_edge_type).
    """
    import torch
    edge_index_dict = {}
    merged_ei = []
    merged_et = []
    for rid, triple in enumerate(RELATIONS):
        ei, _ = mrg.rel[triple]
        if ei.shape[1] and train_only_mask is not None:
            keep = train_only_mask[ei[0]] & train_only_mask[ei[1]]
            ei = ei[:, keep]
        t = torch.tensor(ei, dtype=torch.long, device=dev)
        edge_index_dict[triple] = t
        if t.shape[1]:
            merged_ei.append(t)
            merged_et.append(torch.full((t.shape[1],), rid, dtype=torch.long, device=dev))
    if merged_ei:
        rgcn_ei = torch.cat(merged_ei, dim=1)
        rgcn_et = torch.cat(merged_et, dim=0)
    else:
        rgcn_ei = torch.zeros((2, 0), dtype=torch.long, device=dev)
        rgcn_et = torch.zeros((0,), dtype=torch.long, device=dev)
    return edge_index_dict, rgcn_ei, rgcn_et


def train_eval_fold(df, well_ids, y_well, node_block, test_block, feature_cols, *,
                    name="hgt", hidden=64, layers=2, dropout=0.3, heads=4,
                    k_spatial=8, cap_km_spatial=1.5, k_subbasin=8, cap_km_subbasin=2.0,
                    lr=5e-3, weight_decay=5e-4, max_epochs=400, patience=50,
                    val_frac=0.18, n_val_micro=6, lr_schedule=True, inductive=True,
                    coords=None, subbasin=None, y_row=None, seed=C.SEED, verbose=False):
    """Train on one outer fold (test = nodes with block == test_block), early-stopped on a
    robust spatial VAL set carved from TRAIN, score test nodes once, return per-fold metrics.

    Inductive (default): the message-passing edges used DURING TRAINING are restricted to
    TRAIN-TRAIN pairs per relation (C-SPAT.4); at SCORING time the full cross-block-free
    edge set is used so a test well aggregates from its TRAIN neighbours through cut edges.

    `y_row` (row-level T1a) is required to compute row-level test metrics (broadcast).
    `coords`/`subbasin` are per-well arrays (well_ids order); recomputed from df if None.
    Returns (FoldResult, proba_node[n_nodes]).
    """
    import torch
    import torch.nn.functional as Fn

    set_seed(seed)
    dev = device()

    if coords is None or subbasin is None:
        _, coords2, _ = G.well_table(df)
        coords = coords if coords is not None else coords2
        subbasin = subbasin if subbasin is not None else G.well_subbasin(df, well_ids)

    test_nodes = node_block == test_block
    train_nodes_all = ~test_nodes

    # robust spatial validation carved from TRAIN nodes only (anti-leak, P0)
    val_nodes, n_micro_used = _robust_val_mask_coords(
        coords, train_nodes_all, n_micro=n_val_micro, val_frac=val_frac, seed=seed)
    fit_nodes = train_nodes_all & ~val_nodes
    if val_nodes.sum() == 0 or fit_nodes.sum() == 0:        # degenerate guard
        rng = np.random.RandomState(seed)
        idx = np.where(train_nodes_all)[0]
        vi = rng.choice(idx, size=max(1, int(val_frac * len(idx))), replace=False)
        val_nodes = np.zeros_like(train_nodes_all); val_nodes[vi] = True
        fit_nodes = train_nodes_all & ~val_nodes

    # graph (both relations), cross-block cut + asserted per relation
    mrg = build_multirel_graph(df, well_ids, coords, subbasin, node_block,
                               k_spatial=k_spatial, cap_km_spatial=cap_km_spatial,
                               k_subbasin=k_subbasin, cap_km_subbasin=cap_km_subbasin)

    # node features fit on FIT nodes only (anti-leak; frequency encode = no y needed)
    X, _, _ = G.node_features(df, well_ids, feature_cols, train_node_mask=fit_nodes,
                              y_node=y_well, encode="frequency")

    x = torch.tensor(X, dtype=torch.float32, device=dev)
    y = torch.tensor(y_well, dtype=torch.float32, device=dev)
    m_fit = torch.tensor(fit_nodes, dtype=torch.bool, device=dev)
    m_val = torch.tensor(val_nodes, dtype=torch.bool, device=dev)

    # TRAIN-side message-passing edges (C-SPAT.4 inductive): keep TRAIN-TRAIN pairs only.
    train_mp_mask = train_nodes_all if inductive else np.ones_like(train_nodes_all)
    eid_train, rgcn_ei_train, rgcn_et_train = _build_edge_tensors(
        mrg, dev, train_only_mask=train_mp_mask)
    # SCORING edges: full cross-block-free edge set (test well attaches to TRAIN neighbours).
    eid_all, rgcn_ei_all, rgcn_et_all = _build_edge_tensors(mrg, dev, train_only_mask=None)

    def _fwd(model, edge_pack, *, embed=False):
        eid, rei, ret = edge_pack
        if name.lower() == "rgcn":
            return model(x, rei, ret, embed=embed)
        return model({"well": x}, eid, embed=embed)

    train_pack = (eid_train, rgcn_ei_train, rgcn_et_train)
    score_pack = (eid_all, rgcn_ei_all, rgcn_et_all)

    model = build_model_t1(name, in_dim=X.shape[1], hidden=hidden, layers=layers,
                           dropout=dropout, heads=heads).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = None
    if lr_schedule:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=max(5, patience // 4), min_lr=1e-5)
    pos = float(y[m_fit].sum()); neg = float(m_fit.sum().item() - pos)
    pos_weight = torch.tensor([neg / pos if pos > 0 else 1.0], device=dev)

    best_val, best_state, best_epoch, bad = -np.inf, None, 0, 0
    for epoch in range(max_epochs):
        model.train(); opt.zero_grad()
        out = _fwd(model, train_pack)
        loss = Fn.binary_cross_entropy_with_logits(out[m_fit], y[m_fit],
                                                   pos_weight=pos_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            # validation also uses TRAIN-side edges (val nodes ARE train-side, inductive)
            p = torch.sigmoid(_fwd(model, train_pack))
            pv = p[m_val].cpu().numpy(); yv = y[m_val].cpu().numpy()
            try:
                from sklearn.metrics import roc_auc_score
                vauc = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else 0.0
            except Exception:
                vauc = 0.0
        if sched is not None:
            sched.step(vauc)
        if vauc > best_val + 1e-4:
            best_val, best_epoch, bad = vauc, epoch, 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and epoch % 25 == 0:
            print(f"  ep{epoch} loss={float(loss.detach()):.4f} val_auc={vauc:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.1e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        proba_node = torch.sigmoid(_fwd(model, score_pack)).cpu().numpy()
        # per-node pre-head embedding (for the fusion / stacking arms, arch #2/#3).
        # Scored with the SAME cross-block-free edge set as the probas so a test well's
        # embedding aggregates ONLY from its TRAIN neighbours (C-SPAT.4 inductive).
        emb_node = _fwd(model, score_pack, embed=True).cpu().numpy().astype(np.float32)

    # OOF threshold from VAL nodes (C-THR: never from test)
    thr = _f1_threshold(y_well[val_nodes], proba_node[val_nodes])

    # row-level test metrics (broadcast node proba -> sampling rows), comparable to the wall
    if y_row is None:
        y_row = T.build_T1a(df).to_numpy()
    proba_row = proba_node[mrg.row_to_node]
    test_row_mask = test_nodes[mrg.row_to_node]
    yt = np.asarray(y_row)[test_row_mask]
    pt = proba_row[test_row_mask]
    mets = M.binary_metrics(yt, pt, threshold=thr)
    mets["ece"] = _ece(yt, pt)

    fr = FoldResult(
        fold=int(test_block), metrics_spatial=mets, best_epoch=int(best_epoch),
        n_val_micro=int(n_micro_used), n_val_nodes=int(val_nodes.sum()),
        audit=dict(mrg.audit),
        n_edges_near=int(mrg.rel[R_NEAR][0].shape[1]),
        n_edges_subbasin=int(mrg.rel[R_SUBBASIN][0].shape[1]))
    return fr, proba_node, emb_node


# =====================================================================================
# 4. CV RUNNER
# =====================================================================================
def _run_one_regime(df, *, name, regime, feature_cols, n_blocks, well_ids, coords,
                    subbasin, y_well, y_row, well_to_node, model_kw, train_kw, seed,
                    verbose):
    """Leave-one-block-out CV for one regime ('spatial' or 'random'). Returns a dict with
    aggregated ROW-LEVEL OOF metrics + per-fold list + audit."""
    if regime == "spatial":
        fold_block_row = S.spatial_block_folds(df, k=n_blocks, seed=seed)
    else:
        fold_block_row = S.group_random_folds(df, k=n_blocks, seed=seed)

    # per-well block id (a well lives in one block)
    bdf = pd.DataFrame({"w": df[C.WELL_ID].to_numpy(), "b": fold_block_row})
    if int((bdf.groupby("w")["b"].nunique() > 1).sum()):
        raise AssertionError("a well straddles >1 block — fold_block must be well-consistent")
    per_well = bdf.groupby("w")["b"].agg(lambda s: int(s.iloc[0]))
    node_block = per_well.reindex(well_ids).to_numpy().astype(int)

    blocks = sorted(set(node_block.tolist()))
    row_to_node = df[C.WELL_ID].map(well_to_node).to_numpy().astype(np.int64)
    proba_node_oof = np.full(len(well_ids), np.nan, dtype=np.float64)
    folds = []
    for b in blocks:
        fr, proba_node, _emb_node = train_eval_fold(
            df, well_ids, y_well, node_block, b, feature_cols,
            name=name, coords=coords, subbasin=subbasin, y_row=y_row, seed=seed,
            verbose=verbose, **model_kw, **train_kw)
        test_nodes = node_block == b
        proba_node_oof[test_nodes] = proba_node[test_nodes]
        folds.append(fr)
        if verbose:
            a = fr.audit
            print(f"[{regime}] block {b}: AUC={fr.metrics_spatial['roc_auc']:.4f} "
                  f"xblock(near={a['n_cross_block_near']},sub={a['n_cross_block_subbasin']}) "
                  f"removed(near={a['n_removed_cross_block_near']},"
                  f"sub={a['n_removed_cross_block_subbasin']}) best_ep={fr.best_epoch}")

    # aggregate OOF at the ROW level (broadcast well -> samplings), score like the wall
    proba_row = proba_node_oof[row_to_node]
    valid = ~np.isnan(proba_row)
    yv = np.asarray(y_row)[valid]
    pv = proba_row[valid]
    wells_row = df[C.WELL_ID].to_numpy()[valid]
    thr = _f1_threshold(yv, pv)
    global_mets = M.binary_metrics(yv, pv, threshold=thr)
    global_mets["ece"] = _ece(yv, pv)
    ci = _bootstrap_ci_auc(yv, pv, wells_row, seed=seed)
    reliability = _reliability_curve(yv, pv)

    aucs = [f.metrics_spatial["roc_auc"] for f in folds
            if not np.isnan(f.metrics_spatial["roc_auc"])]
    total_cross = int(sum(f.audit["n_cross_block_near"] + f.audit["n_cross_block_subbasin"]
                          for f in folds))
    return {
        "regime": regime, "model": name, "n_blocks": len(blocks),
        "auc_oof_global": float(global_mets["roc_auc"]),
        "auc_oof_ci95": ci,
        "per_fold_auc": [float(a) for a in aucs],
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "global_metrics": global_mets,
        "reliability_curve": reliability,
        "per_fold": [
            {"fold": f.fold, "metrics": f.metrics_spatial, "best_epoch": f.best_epoch,
             "n_val_nodes": f.n_val_nodes, "n_val_micro": f.n_val_micro,
             "audit": f.audit, "n_edges_near": f.n_edges_near,
             "n_edges_subbasin": f.n_edges_subbasin}
            for f in folds],
        "n_cross_block_total": total_cross,            # MUST be 0 (C-SPAT.2/5)
    }


def run_t1_multirel_cv(df, *, name="hgt", regime="spatial", feature_cols=None,
                       n_blocks=None, hidden=64, layers=2, dropout=0.3, heads=4,
                       k_spatial=8, cap_km_spatial=1.5, k_subbasin=8, cap_km_subbasin=2.0,
                       lr=5e-3, weight_decay=5e-4, max_epochs=400, patience=50,
                       val_frac=0.18, n_val_micro=6, lr_schedule=True, inductive=True,
                       compute_delta=False, smoke=False, write=True,
                       exp_dir=None, seed=C.SEED, verbose=False):
    """Full leave-one-block-out CV for T1a with a multi-relational encoder (HGT/R-GCN/
    hetero_sage) on the two-relation well graph.

    `regime`        : 'spatial' (reference, C-SPAT.1) or 'random' (the optimistic Δ arm).
    `compute_delta` : if True, ALSO runs the 'random' regime and reports
                      Δ(random − spatial) AUC (C-SPAT.6).
    `smoke`         : subsample ~400 wells, 1 fold, few epochs — CPU < ~3 min (CLAUDE.md §5).
    `write`         : write metrics.json + REPORT.md to `exp_dir`
                      (default experiments/hgt_rgcn_t1).

    Returns a dict {regime: regime_result, ..., 'delta_random_minus_spatial': float?}.
    """
    feature_cols = feature_cols or C.feature_columns(include_location=False, cocontam="core")

    if smoke:
        # subsample wells (and their rows) for a fast CPU end-to-end check
        rng = np.random.RandomState(seed)
        all_wells = df[C.WELL_ID].unique()
        keep = set(rng.choice(all_wells, size=min(400, len(all_wells)), replace=False))
        df = df[df[C.WELL_ID].isin(keep)].reset_index(drop=True)
        n_blocks = n_blocks or 3
        max_epochs = min(max_epochs, 15)
        patience = min(patience, 6)
        n_val_micro = min(n_val_micro, 3)
    n_blocks = n_blocks or (C.N_SPATIAL_BLOCKS if regime == "spatial" else C.N_RANDOM_FOLDS)

    well_ids, coords, well_to_node = G.well_table(df)
    subbasin = G.well_subbasin(df, well_ids)
    y_row = T.build_T1a(df).to_numpy()
    y_well = G.well_majority_target(df, y_row, well_ids)

    model_kw = dict(hidden=hidden, layers=layers, dropout=dropout, heads=heads)
    train_kw = dict(k_spatial=k_spatial, cap_km_spatial=cap_km_spatial,
                    k_subbasin=k_subbasin, cap_km_subbasin=cap_km_subbasin,
                    lr=lr, weight_decay=weight_decay, max_epochs=max_epochs,
                    patience=patience, val_frac=val_frac, n_val_micro=n_val_micro,
                    lr_schedule=lr_schedule, inductive=inductive)

    out = {}
    spatial_res = _run_one_regime(
        df, name=name, regime="spatial" if regime == "spatial" else regime,
        feature_cols=feature_cols, n_blocks=n_blocks, well_ids=well_ids, coords=coords,
        subbasin=subbasin, y_well=y_well, y_row=y_row, well_to_node=well_to_node,
        model_kw=model_kw, train_kw=train_kw, seed=seed, verbose=verbose) \
        if regime == "spatial" else None

    if regime != "spatial":
        out[regime] = _run_one_regime(
            df, name=name, regime=regime, feature_cols=feature_cols, n_blocks=n_blocks,
            well_ids=well_ids, coords=coords, subbasin=subbasin, y_well=y_well, y_row=y_row,
            well_to_node=well_to_node, model_kw=model_kw, train_kw=train_kw, seed=seed,
            verbose=verbose)
    else:
        out["spatial"] = spatial_res

    if compute_delta:
        other = "random" if regime == "spatial" else "spatial"
        nb_other = n_blocks
        out[other] = _run_one_regime(
            df, name=name, regime=other, feature_cols=feature_cols, n_blocks=nb_other,
            well_ids=well_ids, coords=coords, subbasin=subbasin, y_well=y_well, y_row=y_row,
            well_to_node=well_to_node, model_kw=model_kw, train_kw=train_kw, seed=seed,
            verbose=verbose)
        if "spatial" in out and "random" in out:
            out["delta_random_minus_spatial"] = float(
                out["random"]["auc_oof_global"] - out["spatial"]["auc_oof_global"])

    out["meta"] = {
        "task": "T1a", "model": name, "smoke": bool(smoke), "seed": int(seed),
        "feature_cols": list(feature_cols), "include_location": False,
        "n_features": len(feature_cols), "inductive": bool(inductive),
        "k_spatial": k_spatial, "cap_km_spatial": cap_km_spatial,
        "k_subbasin": k_subbasin, "cap_km_subbasin": cap_km_subbasin,
        "hidden": hidden, "layers": layers, "dropout": dropout, "heads": heads,
        "relations": list(REL_NAMES),
    }

    if write:
        _write_outputs(out, name=name, exp_dir=exp_dir)
    return out


# =====================================================================================
# 5. OUTPUTS (metrics.json + REPORT.md)
# =====================================================================================
def _write_outputs(out, *, name, exp_dir=None):
    exp_dir = Path(exp_dir) if exp_dir is not None else (C.EXPERIMENTS_DIR / "hgt_rgcn_t1")
    exp_dir.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    (exp_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=_default))

    meta = out.get("meta", {})
    lines = [
        f"# HGT / R-GCN multi-relational T1 — {name}", "",
        "Multi-relational encoders over a HOMOGENEOUS well-well graph with TWO edge types "
        "(`near` spatial k-NN cap 1.5 km, `same_subbasin_knn` intra-sub-basin k-NN cap 2 km). "
        "Eval-validated design (experiments/hgt_rgcn_t1/eval_validation.md): no fabricated "
        "source node type; HGT/R-GCN used purely as relational encoders.", "",
        f"- model: **{meta.get('model')}**  smoke={meta.get('smoke')}  seed={meta.get('seed')}",
        f"- features: {meta.get('n_features')} cols, include_location={meta.get('include_location')} "
        "(C-LOC.1: lat/lon NOT node features)",
        f"- inductive (C-SPAT.4): {meta.get('inductive')}",
        f"- relations: {meta.get('relations')}", "",
        "## Results (row-level OOF, comparable to the non-graph wall and gnn_phase1)", "",
        "| regime | AUC OOF | AUC 95% CI | AUC mean±std (folds) | F1@OOF | PR-AUC | "
        "bal.acc | Brier | ECE | xblock |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for reg in ("spatial", "random"):
        if reg not in out:
            continue
        r = out[reg]
        gm = r["global_metrics"]
        ci = r["auc_oof_ci95"]
        lines.append(
            f"| {reg} | {r['auc_oof_global']:.4f} | "
            f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] | "
            f"{r['auc_mean']:.4f}±{r['auc_std']:.4f} | {gm['f1']:.4f} | "
            f"{gm['pr_auc']:.4f} | {gm['balanced_accuracy']:.4f} | {gm['brier']:.4f} | "
            f"{r['global_metrics'].get('ece', float('nan')):.4f} | "
            f"{r['n_cross_block_total']} |")
    if "delta_random_minus_spatial" in out:
        lines += ["", f"**Δ(random − spatial) AUC = {out['delta_random_minus_spatial']:.4f}** "
                  "(C-SPAT.6: the spatial-leakage inflation; a large Δ confirms random split "
                  "is an optimistic artefact, not a real generalisation gain)."]
    lines += [
        "", "## Leakage guard (C-SPAT.2 / C-SPAT.5)",
        "Cross-block edges are cut SEPARATELY per relation and asserted to 0. "
        f"Total residual cross-block edges across all folds/regimes: "
        f"{sum(out[r]['n_cross_block_total'] for r in ('spatial','random') if r in out)} "
        "(must be 0).",
        "", "## Calibration (C-CAL)",
        "Brier + ECE reported above; per-bin reliability curve stored in metrics.json "
        "under `reliability_curve`.",
        "", "## Positioning",
        "gnn_phase1 single-relation spatial AUC: GraphSAGE 0.618±0.067, GCN 0.624±0.074. "
        "The honest comparison is THIS spatial AUC vs that wall; gains below the inter-fold "
        "σ (~0.06–0.07) are within noise and not claimed (eval C-CMP).",
    ]
    (exp_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return exp_dir
