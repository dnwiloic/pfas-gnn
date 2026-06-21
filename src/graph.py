"""Graph construction for the PFAS GNN phase — reuses the frozen src/ socle.

Design decisions (justified in experiments/gnn_phase1/REPORT.md):

* NODE granularity = WELL (gm_well_id, 11 333 nodes), not sampling event. Rationale:
  coordinates, spatial blocks (splits.spatial_block_*) and the group key (C2) all live at
  the well level; a per-event graph would explode the spatial edge count (cross-product of
  samplings, ~1.6M edges) and merely duplicate identical coordinates. Node features are
  the per-well aggregate of the non-leaking context (mean for numeric, mode for low-card
  categoricals). The training label per node is the well-majority target.

* EVALUATION stays at the SAMPLING level (46 338 rows) for strict comparability with the
  non-graph WALL (RF/XGB spatial AUC ~0.60 are row-level). The node probability is
  broadcast back to every sampling of that well; row-level metrics use the row target.
  Grouping (C2) is automatically honoured (one well = one node = one block).

* EDGES = spatial k-NN at the WELL level with a HARD DISTANCE CAP (~1-2 km, C4): beyond
  the measured autocorrelation range (2-5 km) "proximity" only re-encodes the map = spatial
  leakage. Haversine distance (great-circle) so the cap is a true physical km.

* C4 inter-block cut: when a fold-block vector is given, every edge whose two endpoints
  fall in DIFFERENT CV blocks is REMOVED, so message passing never carries train<->test
  information. The graph is therefore (re)built per fold. With KMeans blocks the eval audit
  measured 0 wells <1 km across a block boundary, so the cut removes very few edges yet is
  mandatory for a valid spatial CV.

* Node features are fit on TRAIN wells only (FeaturePipeline, anti-leakage). lat/lon are
  NOT node features (C6): geography enters ONLY through the distance-capped k-NN edges.

* T2 bipartite track: build_bipartite_well_analyte() turns the MNAR measurement matrix into
  a wells<->analytes bipartite graph (reuses baselines_t2.measurement_mask) for link
  prediction / matrix completion (HYDRO_CRITIQUE recommendation over a label graph).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import baselines_t2 as B2
from . import config as C
from . import features as F
from . import splits as S

EARTH_R_KM = 6371.0088


# --------------------------------------------------------------------------- coords
def well_table(df: pd.DataFrame):
    """One row per well: ids, coords, and the integer node index map.

    Returns (well_ids, coords[lat,lon], well_to_node dict). Coords are the per-well mean
    (= splits.well_coordinates), so a well = one node = one (lat,lon) = one block."""
    well_ids, coords = S.well_coordinates(df)
    well_to_node = {w: i for i, w in enumerate(well_ids)}
    return well_ids, coords, well_to_node


# ------------------------------------------------------------------ spatial k-NN
def knn_edges_km(coords: np.ndarray, k: int = 8, cap_km: float = 1.5):
    """Undirected spatial k-NN edges between wells, capped at `cap_km` (great-circle).

    Returns (edge_index[2,E] int64, edge_dist_km[E]). Each undirected edge appears once
    here; callers symmetrise. Self-loops excluded. An edge survives only if its haversine
    distance <= cap_km — the hard physical cap of C4.
    """
    from sklearn.neighbors import BallTree

    n = coords.shape[0]
    kq = min(k + 1, n)
    tree = BallTree(np.radians(coords), metric="haversine")
    dist, idx = tree.query(np.radians(coords), k=kq)
    dist_km = dist * EARTH_R_KM

    src, dst, dkm = [], [], []
    for i in range(n):
        for j in range(1, kq):                 # skip self (column 0)
            d = dist_km[i, j]
            if d <= cap_km:
                a, b = i, int(idx[i, j])
                if a < b:                       # dedup undirected (BallTree is symmetric-ish)
                    src.append(a); dst.append(b); dkm.append(d)
                elif a > b:
                    src.append(b); dst.append(a); dkm.append(d)
    if not src:
        return (np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float64))
    # collapse possible duplicate undirected pairs (i in j's knn and j in i's knn)
    key = np.array(src, dtype=np.int64) * n + np.array(dst, dtype=np.int64)
    _, uniq = np.unique(key, return_index=True)
    ei = np.vstack([np.array(src)[uniq], np.array(dst)[uniq]]).astype(np.int64)
    ed = np.array(dkm)[uniq].astype(np.float64)
    return ei, ed


def cut_cross_block(edge_index: np.ndarray, edge_dist: np.ndarray,
                    node_block: np.ndarray):
    """C4: drop every edge whose endpoints sit in DIFFERENT CV blocks.

    `node_block[i]` = CV block id of node i. Returns the filtered (edge_index, edge_dist)
    and the number of edges removed (reported as the spatial-leakage guard)."""
    a, b = edge_index[0], edge_index[1]
    same = node_block[a] == node_block[b]
    removed = int((~same).sum())
    return edge_index[:, same], edge_dist[same], removed


def symmetrise(edge_index: np.ndarray, edge_dist: np.ndarray):
    """Make the edge set bidirectional (PyG message passing expects both directions)."""
    if edge_index.shape[1] == 0:
        return edge_index, edge_dist
    ei = np.hstack([edge_index, edge_index[[1, 0]]])
    ed = np.hstack([edge_dist, edge_dist])
    return ei, ed


# ------------------------------------------------------------ node features / labels
_MODE = object()


def aggregate_to_wells(df: pd.DataFrame, well_ids: np.ndarray, cols):
    """Per-well aggregate frame over `cols`: mean for numeric, mode for categorical.

    Index = well_ids order (so row r corresponds to node r). Categorical low/high-card
    columns keep their raw category (encoded later by FeaturePipeline, train-fit)."""
    cat = set(C.CATEGORICAL_LOW_CARD) | set(C.CATEGORICAL_HIGH_CARD)
    present = [c for c in cols if c in df.columns]
    g = df.groupby(C.WELL_ID)
    out = {}
    for c in present:
        if c in cat:
            out[c] = g[c].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        else:
            out[c] = pd.to_numeric(g[c].mean(), errors="coerce")
    # keep date column as the mean date (for temporal derivation) if present
    if C.DATE_COL in df.columns:
        out[C.DATE_COL] = g[C.DATE_COL].agg(lambda s: s.iloc[len(s) // 2])
    wf = pd.DataFrame(out)
    return wf.reindex(well_ids)


def well_majority_target(df: pd.DataFrame, y_row: np.ndarray, well_ids: np.ndarray):
    """Well-level training label = majority of the sampling targets (ties -> positive
    only if >= 0.5). Returns an int array aligned with well_ids/node order."""
    t = pd.DataFrame({"w": df[C.WELL_ID].to_numpy(), "y": np.asarray(y_row)})
    m = t.groupby("w")["y"].mean().reindex(well_ids)
    return (m.to_numpy() >= 0.5).astype(int)


# --------------------------------------------------------------------------- bundle
@dataclass
class WellGraph:
    """Container the GNN runner consumes. Tensors are built in src/gnn.py to keep this
    module torch-free for cheap CPU import; here everything is numpy/pandas."""
    well_ids: np.ndarray
    coords: np.ndarray
    node_block: np.ndarray            # CV block id per node (well)
    edge_index: np.ndarray            # [2, E] symmetrised, post inter-block cut
    edge_dist: np.ndarray             # [E] km (edge weight source)
    row_to_node: np.ndarray           # [n_rows] node index of each sampling row
    n_removed_cross_block: int = 0
    meta: dict = field(default_factory=dict)


def build_well_graph(df: pd.DataFrame, *, fold_block: np.ndarray | None = None,
                     k: int = 8, cap_km: float = 1.5, cut_blocks: bool = True) -> WellGraph:
    """Assemble the well-level spatial graph (topology + block ids + row->node map).

    `fold_block` is a per-ROW block id (e.g. splits.spatial_block_folds(df)); it is reduced
    to a per-WELL block (a well lives in exactly one block by construction) and used both as
    node_block and, when cut_blocks, to cut cross-block edges (C4).
    """
    well_ids, coords, well_to_node = well_table(df)
    n = len(well_ids)

    if fold_block is not None:
        bdf = pd.DataFrame({"w": df[C.WELL_ID].to_numpy(), "b": np.asarray(fold_block)})
        per_well = bdf.groupby("w")["b"].agg(lambda s: int(s.iloc[0]))
        # guard: a well must not straddle blocks
        nun = bdf.groupby("w")["b"].nunique()
        if int((nun > 1).sum()):
            raise AssertionError("a well straddles >1 block — fold_block must be well-consistent")
        node_block = per_well.reindex(well_ids).to_numpy().astype(int)
    else:
        node_block = np.zeros(n, dtype=int)

    ei, ed = knn_edges_km(coords, k=k, cap_km=cap_km)
    removed = 0
    if cut_blocks and fold_block is not None:
        ei, ed, removed = cut_cross_block(ei, ed, node_block)
    ei, ed = symmetrise(ei, ed)

    row_to_node = df[C.WELL_ID].map(well_to_node).to_numpy().astype(np.int64)

    return WellGraph(
        well_ids=well_ids, coords=coords, node_block=node_block,
        edge_index=ei, edge_dist=ed, row_to_node=row_to_node,
        n_removed_cross_block=removed,
        meta={"n_nodes": n, "n_edges_undirected": ei.shape[1] // 2,
              "k": k, "cap_km": cap_km, "cut_blocks": bool(cut_blocks and fold_block is not None)},
    )


# ------------------------------------------------------- node feature construction
def node_features(df: pd.DataFrame, well_ids: np.ndarray, feature_cols,
                  train_node_mask: np.ndarray, y_node: np.ndarray | None = None,
                  encode: str = "frequency"):
    """Per-well node feature matrix, FeaturePipeline fit on TRAIN nodes only (anti-leak).

    Aggregates df to wells (mean/mode), then fits the frozen FeaturePipeline on the train
    wells and transforms all wells. `encode='frequency'` is leak-free without y (default,
    matches the T2 baseline); `encode='target'` needs y_node (binary) and is OOF-encoded
    internally by the pipeline. Returns (X[n_nodes, d], feature_names, pipe).
    """
    wf = aggregate_to_wells(df, well_ids, feature_cols)
    pipe = F.FeaturePipeline(feature_cols, encode=encode)
    wf_tr = wf.iloc[train_node_mask] if train_node_mask.dtype == bool else wf.iloc[train_node_mask]
    y_tr = None
    if encode == "target":
        if y_node is None:
            raise ValueError("encode='target' needs y_node")
        y_tr = np.asarray(y_node)[train_node_mask]
    pipe.fit_transform(wf_tr, y_tr)
    X, names = pipe.transform(wf)
    return X.astype(np.float32), names, pipe


# =====================================================================================
# T2 — bipartite wells x analytes graph (matrix completion / link prediction track)
# =====================================================================================
@dataclass
class BipartiteGraph:
    """wells (left) x analytes (right) bipartite graph for T2 matrix completion.

    edge_index[2,E] indexes wells [0..n_wells) and analytes [0..n_analytes); an edge exists
    where the analyte was MEASURED at the well (measurement_mask, MNAR-aware) and carries the
    binary exceedance label (build_T2). Train/test edge masks let a link-prediction / GAE
    model complete the matrix. Wells also carry the same context node features as WellGraph.
    """
    well_ids: np.ndarray
    analytes: list
    edge_well: np.ndarray             # [E] well node index
    edge_analyte: np.ndarray          # [E] analyte index
    edge_label: np.ndarray            # [E] 0/1 exceedance on a measured cell
    well_block: np.ndarray            # [n_wells] CV block per well (for spatial split)


def build_bipartite_well_analyte(df: pd.DataFrame, *, labels=None,
                                 fold_block: np.ndarray | None = None) -> BipartiteGraph:
    """Reformulate the lacunar wells x analyte exceedance matrix as a bipartite graph.

    Aggregates to the WELL level: a (well, analyte) edge exists iff the analyte was measured
    in at least one sampling of that well; its label is the well-majority exceedance. Reuses
    baselines_t2.measurement_mask (availability only -> no leakage) and targets.build_T2.
    """
    labels = labels or C.T2_LABELS
    well_ids, _, well_to_node = well_table(df)

    Y, M = B2.masked_targets(df, labels=labels)        # row-level 0/1 and bool mask
    wcol = df[C.WELL_ID].to_numpy()
    edge_well, edge_analyte, edge_label = [], [], []
    for j, a in enumerate(labels):
        col = f"label_{a}"
        meas = M[col].to_numpy()
        if not meas.any():
            continue
        sub = pd.DataFrame({"w": wcol[meas], "y": Y[col].to_numpy()[meas]})
        agg = sub.groupby("w")["y"].mean()
        for w, frac in agg.items():
            edge_well.append(well_to_node[w])
            edge_analyte.append(j)
            edge_label.append(int(frac >= 0.5))

    if fold_block is not None:
        bdf = pd.DataFrame({"w": wcol, "b": np.asarray(fold_block)})
        per_well = bdf.groupby("w")["b"].agg(lambda s: int(s.iloc[0]))
        well_block = per_well.reindex(well_ids).to_numpy().astype(int)
    else:
        well_block = np.zeros(len(well_ids), dtype=int)

    return BipartiteGraph(
        well_ids=well_ids, analytes=list(labels),
        edge_well=np.array(edge_well, dtype=np.int64),
        edge_analyte=np.array(edge_analyte, dtype=np.int64),
        edge_label=np.array(edge_label, dtype=np.int64),
        well_block=well_block,
    )
