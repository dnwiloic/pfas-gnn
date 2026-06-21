"""First GNN models + training loop for T1 (binary) on the well-level spatial graph.

Reuses the frozen socle: targets (T1a), splits (spatial-block reference + group-random Δ),
graph.py (distance-capped k-NN, C4 inter-block cut), metrics.binary_metrics.

Training is TRANSDUCTIVE with masks: all nodes (wells) sit in one graph; the loss is
computed on TRAIN nodes only, validation drives early stopping, test nodes are scored once
per fold. Because edges that cross a CV block boundary are removed (C4), message passing
never carries information from test wells into train wells — the spatial CV stays valid.
The same code path runs inductively in spirit (a test well's prediction depends only on its
train-side neighbourhood, all cross-block links being cut), which we report as the
generalisation-to-unseen-regions number.

Evaluation is at the SAMPLING level (rows) for strict comparability with the non-graph WALL:
a node (well) probability is broadcast to every sampling of that well, then row-level
metrics use the row target. Grouping (C2) is automatic (one well = one node = one block).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C
from . import graph as G
from . import metrics as M
from . import splits as S
from . import targets as T


# ------------------------------------------------------------------ torch helpers
def _torch():
    import torch
    return torch


def set_seed(seed: int = C.SEED):
    import random
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- models
def build_model(name: str, in_dim: int, hidden: int = 64, layers: int = 2,
                dropout: float = 0.5):
    """One of {'graphsage','gcn','graphconv'}: 2-layer (default) node classifier with
    edge-weight support where the conv allows it. Returns an nn.Module."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    from torch_geometric.nn import GraphConv, GCNConv, SAGEConv

    name = name.lower()

    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.name = name
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            dims = [in_dim] + [hidden] * layers
            for a, b in zip(dims[:-1], dims[1:]):
                if name == "graphsage":
                    self.convs.append(SAGEConv(a, b, aggr="mean"))
                elif name == "gcn":
                    self.convs.append(GCNConv(a, b, add_self_loops=True))
                elif name == "graphconv":
                    self.convs.append(GraphConv(a, b, aggr="mean"))
                else:
                    raise ValueError(f"unknown model {name}")
                self.norms.append(nn.LayerNorm(b))
            self.head = nn.Linear(hidden, 1)
            self.dropout = dropout
            self._uses_weight = name in ("gcn", "graphconv")

        def forward(self, x, edge_index, edge_weight=None):
            for conv, norm in zip(self.convs, self.norms):
                if self._uses_weight and edge_weight is not None:
                    x = conv(x, edge_index, edge_weight)
                else:
                    x = conv(x, edge_index)
                x = norm(x)
                x = Fn.relu(x)
                x = Fn.dropout(x, p=self.dropout, training=self.training)
            return self.head(x).squeeze(-1)

    return GNN()


def edge_weight_from_dist(edge_dist_km: np.ndarray, cap_km: float):
    """Map edge length to a (0,1] weight: closer = stronger (1 at 0 km, ->small at cap).
    Gaussian-like decay with the cap as the scale, so the cap is a soft physical prior."""
    import torch
    w = np.exp(-(edge_dist_km / max(cap_km, 1e-6)) ** 2)
    return torch.tensor(w, dtype=torch.float32)


# --------------------------------------------------------------------------- run
@dataclass
class FoldResult:
    fold: int
    metrics_spatial: dict
    n_removed_cross_block: int
    n_edges: int
    best_epoch: int
    n_val_micro: int = 0
    n_val_nodes: int = 0


def _make_data(df, y_row, feature_cols, fold_block, *, k, cap_km, cut_blocks, encode):
    """Build the WellGraph + node features + node labels for one configuration.
    Train mask for feature fitting/loss is derived later per fold (here we just assemble
    the static parts: topology depends on fold_block via the inter-block cut)."""
    wg = G.build_well_graph(df, fold_block=fold_block, k=k, cap_km=cap_km,
                            cut_blocks=cut_blocks)
    y_node = G.well_majority_target(df, y_row, wg.well_ids)
    return wg, y_node


def _robust_val_mask(node_block, train_nodes_all, *, n_micro=6, val_frac=0.18,
                     seed=C.SEED):
    """P0: spatial validation = several MICRO-BLOCKS assembled, not a single spatial block.

    Phase-1 took the single train block with the largest id as validation; that holdout is
    one compact region, so its AUC is noisy and early-stop fired too soon on 2-3 folds
    (EVAL_PROTOCOL §2.4 warns a single block is statistically under-powered). Here we split
    the TRAIN wells into `n_micro` spatial micro-blocks (KMeans on their coords) and hold
    out a `val_frac` slice of EACH micro-block, so validation spans the whole train extent
    (more representative, less variance) while staying spatial WITHIN the train side. The
    held-out wells are still TRAIN-side, so no test leakage; the FIT graph keeps all train
    edges (we only mask the loss/early-stop nodes, transductive)."""
    from sklearn.cluster import KMeans

    train_idx = np.where(train_nodes_all)[0]
    if len(train_idx) < 2 * n_micro:               # tiny smoke fallback: plain random slice
        rng = np.random.RandomState(seed)
        val_idx = rng.choice(train_idx, size=max(1, int(val_frac * len(train_idx))),
                             replace=False)
        val = np.zeros_like(train_nodes_all); val[val_idx] = True
        return val
    # we need coords; reuse node_block only as a fallback key. Caller passes coords below.
    raise RuntimeError("call _robust_val_mask_coords")  # pragma: no cover


def _robust_val_mask_coords(coords, train_nodes_all, *, n_micro=6, val_frac=0.18,
                            seed=C.SEED):
    """Assemble several spatial micro-blocks of the TRAIN wells and hold out a stratified
    slice of each as the early-stop validation set (see _robust_val_mask docstring)."""
    from sklearn.cluster import KMeans

    train_idx = np.where(train_nodes_all)[0]
    rng = np.random.RandomState(seed)
    if len(train_idx) < max(2 * n_micro, 10):      # tiny smoke fallback
        val_idx = rng.choice(train_idx, size=max(1, int(val_frac * len(train_idx))),
                             replace=False)
        val = np.zeros_like(train_nodes_all); val[val_idx] = True
        return val, 1
    n_micro = int(min(n_micro, max(2, len(train_idx) // 30)))
    km = KMeans(n_clusters=n_micro, random_state=seed, n_init=10)
    micro = km.fit_predict(coords[train_idx])
    val = np.zeros_like(train_nodes_all)
    for b in range(n_micro):                       # stratified hold-out per micro-block
        members = train_idx[micro == b]
        if len(members) == 0:
            continue
        n_hold = max(1, int(round(val_frac * len(members))))
        val[rng.choice(members, size=min(n_hold, len(members)), replace=False)] = True
    return val, n_micro


def train_eval_fold(df, y_row, feature_cols, fold_block, test_block, *,
                    model_name="graphsage", k=8, cap_km=1.5, cut_blocks=True,
                    encode="frequency", hidden=64, layers=2, dropout=0.5,
                    lr=5e-3, weight_decay=5e-4, max_epochs=400, patience=50,
                    val_frac=0.18, n_val_micro=6, lr_schedule=True,
                    seed=C.SEED, verbose=False):
    """Train on one outer fold (test = nodes whose block == test_block), early-stopped on a
    ROBUST spatial validation (several train micro-blocks assembled, P0), with an optional
    ReduceLROnPlateau schedule, score test nodes once, broadcast to sampling rows and
    compute row-level binary metrics. Returns (FoldResult, proba_row, test_row_mask)."""
    import torch
    import torch.nn.functional as Fn

    set_seed(seed)
    dev = device()

    wg, y_node = _make_data(df, y_row, feature_cols, fold_block,
                            k=k, cap_km=cap_km, cut_blocks=cut_blocks, encode=encode)
    node_block = wg.node_block
    test_nodes = node_block == test_block
    train_nodes_all = ~test_nodes

    # P0: robust spatial validation = several assembled micro-blocks of the TRAIN wells
    # (replaces phase-1's single-block holdout that under-trained 2-3 folds).
    val_nodes, n_micro_used = _robust_val_mask_coords(
        wg.coords, train_nodes_all, n_micro=n_val_micro, val_frac=val_frac, seed=seed)
    fit_nodes = train_nodes_all & ~val_nodes
    if val_nodes.sum() == 0 or fit_nodes.sum() == 0:   # degenerate guard
        rng = np.random.RandomState(seed)
        idx = np.where(train_nodes_all)[0]
        val_idx = rng.choice(idx, size=max(1, int(val_frac * len(idx))), replace=False)
        val_nodes = np.zeros_like(train_nodes_all); val_nodes[val_idx] = True
        fit_nodes = train_nodes_all & ~val_nodes

    # node features fit on FIT nodes only (anti-leak); frequency encode = no y needed
    X, names, _ = G.node_features(df, wg.well_ids, feature_cols,
                                  train_node_mask=fit_nodes, y_node=y_node, encode=encode)

    x = torch.tensor(X, dtype=torch.float32, device=dev)
    ei = torch.tensor(wg.edge_index, dtype=torch.long, device=dev)
    ew = edge_weight_from_dist(wg.edge_dist, cap_km).to(dev)
    y = torch.tensor(y_node, dtype=torch.float32, device=dev)
    m_fit = torch.tensor(fit_nodes, dtype=torch.bool, device=dev)
    m_val = torch.tensor(val_nodes, dtype=torch.bool, device=dev)

    model = build_model(model_name, in_dim=X.shape[1], hidden=hidden,
                        layers=layers, dropout=dropout).to(dev)
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
        out = model(x, ei, ew)
        loss = Fn.binary_cross_entropy_with_logits(out[m_fit], y[m_fit], pos_weight=pos_weight)
        loss.backward(); opt.step()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(x, ei, ew))
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
        if verbose and epoch % 20 == 0:
            print(f"  ep{epoch} loss={float(loss):.4f} val_auc={vauc:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.1e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        proba_node = torch.sigmoid(model(x, ei, ew)).cpu().numpy()

    # broadcast node proba -> sampling rows; threshold from OOF (val) proba, F1-optimal
    proba_row = proba_node[wg.row_to_node]
    test_row_mask = test_nodes[wg.row_to_node]

    # OOF threshold on VAL nodes (never on test)
    val_p = proba_node[val_nodes]; val_y = y_node[val_nodes]
    thr = _f1_threshold(val_y, val_p)

    yt_row = np.asarray(y_row)[test_row_mask]
    pt_row = proba_row[test_row_mask]
    mets = M.binary_metrics(yt_row, pt_row, threshold=thr)

    fr = FoldResult(fold=int(test_block), metrics_spatial=mets,
                    n_removed_cross_block=wg.n_removed_cross_block,
                    n_edges=wg.edge_index.shape[1], best_epoch=best_epoch,
                    n_val_micro=int(n_micro_used), n_val_nodes=int(val_nodes.sum()))
    return fr, proba_row, test_row_mask


def _f1_threshold(y, p):
    from sklearn.metrics import f1_score
    if len(np.unique(y)) < 2:
        return 0.5
    grid = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def run_t1_cv(df, *, model_name="graphsage", feature_cols=None, regime="spatial",
              k=8, cap_km=1.5, cut_blocks=True, encode="frequency", n_blocks=None,
              hidden=64, layers=2, dropout=0.5, max_epochs=400, patience=50,
              lr=5e-3, n_val_micro=6, val_frac=0.18, lr_schedule=True,
              seed=C.SEED, verbose=False):
    """Full leave-one-block-out CV for T1a on the well graph. `regime`:
       'spatial' -> spatial_block_folds (reference);  'random' -> group_random_folds (Δ).
    Returns (summary_dict, per_fold_list). Row-level metrics, comparable to the WALL.
    """
    feature_cols = feature_cols or C.feature_columns(include_location=False, cocontam="core")
    y_row = T.build_T1a(df).to_numpy()
    n_blocks = n_blocks or (C.N_SPATIAL_BLOCKS if regime == "spatial" else C.N_RANDOM_FOLDS)
    if regime == "spatial":
        fold_block = S.spatial_block_folds(df, k=n_blocks, seed=seed)
        do_cut = cut_blocks
    else:
        fold_block = S.group_random_folds(df, k=n_blocks, seed=seed)
        do_cut = cut_blocks                       # also cut for random (honest per-fold graph)

    blocks = sorted(set(fold_block.tolist()))
    results, oof_p, oof_y, oof_mask = [], [], [], []
    for b in blocks:
        fr, proba_row, test_mask = train_eval_fold(
            df, y_row, feature_cols, fold_block, b,
            model_name=model_name, k=k, cap_km=cap_km, cut_blocks=do_cut,
            encode=encode, hidden=hidden, layers=layers, dropout=dropout,
            lr=lr, max_epochs=max_epochs, patience=patience, n_val_micro=n_val_micro,
            val_frac=val_frac, lr_schedule=lr_schedule, seed=seed, verbose=verbose)
        results.append(fr)
        if verbose:
            print(f"[{regime}] block {b}: AUC={fr.metrics_spatial['roc_auc']:.4f} "
                  f"removed_xblock={fr.n_removed_cross_block} edges={fr.n_edges} "
                  f"best_ep={fr.best_epoch} val_micro={fr.n_val_micro}")

    aucs = [r.metrics_spatial["roc_auc"] for r in results if not np.isnan(r.metrics_spatial["roc_auc"])]
    summary = {
        "regime": regime, "model": model_name,
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "n_blocks": len(blocks),
        "total_removed_cross_block": int(sum(r.n_removed_cross_block for r in results)),
        "cap_km": cap_km, "k": k, "cut_blocks": do_cut, "encode": encode,
        "per_fold_auc": [float(a) for a in aucs],
        "per_fold_best_epoch": [int(r.best_epoch) for r in results],
        "per_fold_n_val_micro": [int(r.n_val_micro) for r in results],
    }
    return summary, results
