"""CPU smoke-test for the GNN phase (must stay < ~3 min on CPU).

Verifies end-to-end on a tiny well subsample:
  * PyTorch Geometric imports on CPU;
  * the well graph builds with sane node/edge counts and a real distance cap;
  * C4 inter-block cut leaves EXACTLY 0 edges crossing a CV block boundary;
  * the forward pass runs, the loss is finite and decreases;
  * row-level binary metrics are computed (comparable to the WALL);
  * the bipartite wells x analyte graph (T2 track) builds with measured-cell edges.

Run directly:  PFAS_FORCE_CPU=1 python3 tests/test_gnn_smoke.py
or via pytest.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("PFAS_FORCE_CPU", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import config as C
from src import data, graph as G, gnn, splits as S, targets as T


SMOKE_WELLS = 1500   # dense enough that some near wells fall in different blocks (C4)
SMOKE_BLOCKS = 6


def _load():
    df = data.load(smoke=True, smoke_n=SMOKE_WELLS)
    return df


def test_pyg_imports_cpu():
    import torch
    import torch_geometric  # noqa
    assert torch.__version__
    print(f"torch {torch.__version__}, pyg {torch_geometric.__version__}, cuda={torch.cuda.is_available()}")


def test_graph_builds_and_caps_distance():
    df = _load()
    fold_block = S.spatial_block_folds(df, k=SMOKE_BLOCKS)
    wg = G.build_well_graph(df, fold_block=fold_block, k=8, cap_km=1.5, cut_blocks=False)
    assert wg.meta["n_nodes"] == df[C.WELL_ID].nunique()
    assert wg.edge_index.shape[0] == 2
    # every edge length <= cap
    assert wg.edge_dist.max() <= 1.5 + 1e-6
    # row->node map covers all rows
    assert wg.row_to_node.shape[0] == len(df)
    assert wg.row_to_node.max() < wg.meta["n_nodes"]
    print(f"nodes={wg.meta['n_nodes']} undirected_edges={wg.meta['n_edges_undirected']} "
          f"max_edge_km={wg.edge_dist.max():.3f}")


def test_c4_zero_cross_block_edges():
    df = _load()
    fold_block = S.spatial_block_folds(df, k=SMOKE_BLOCKS)
    wg = G.build_well_graph(df, fold_block=fold_block, k=8, cap_km=1.5, cut_blocks=True)
    a, b = wg.edge_index[0], wg.edge_index[1]
    cross = int((wg.node_block[a] != wg.node_block[b]).sum())
    assert cross == 0, f"{cross} edges cross a CV block boundary — C4 violated"
    # and the uncut graph DID have some cross-block edges (so the cut is meaningful)
    wg_uncut = G.build_well_graph(df, fold_block=fold_block, k=8, cap_km=1.5, cut_blocks=False)
    au, bu = wg_uncut.edge_index[0], wg_uncut.edge_index[1]
    cross_uncut = int((wg_uncut.node_block[au] != wg_uncut.node_block[bu]).sum())
    print(f"C4: cross-block edges cut={wg.n_removed_cross_block}, "
          f"remaining cross-block={cross}, would-be cross-block (uncut)={cross_uncut}")
    # the uncut graph must have HAD cross-block edges, and the cut must remove exactly them.
    # n_removed counts UNDIRECTED edges (pre-symmetrise); cross_uncut counts DIRECTED edges
    # on the symmetrised graph -> cross_uncut == 2 * n_removed.
    assert cross_uncut > 0, "smoke too sparse to exercise C4 — increase SMOKE_WELLS/blocks"
    assert cross_uncut == 2 * wg.n_removed_cross_block


def test_train_one_fold_loss_finite_and_metrics():
    df = _load()
    y_row = T.build_T1a(df).to_numpy()
    fold_block = S.spatial_block_folds(df, k=SMOKE_BLOCKS)
    cols = C.feature_columns(include_location=False, cocontam="core")
    test_block = int(np.bincount(fold_block).argmax())  # a non-empty block

    # capture loss trajectory via a short run
    fr, proba_row, test_mask = gnn.train_eval_fold(
        df, y_row, cols, fold_block, test_block,
        model_name="graphsage", k=8, cap_km=1.5, cut_blocks=True,
        encode="frequency", hidden=32, layers=2, dropout=0.3,
        max_epochs=40, patience=15, seed=C.SEED, verbose=False)

    assert fr.n_edges >= 0
    m = fr.metrics_spatial
    for key in ("roc_auc", "f1", "accuracy", "recall", "precision"):
        assert key in m
    assert 0.0 <= m["accuracy"] <= 1.0
    assert np.isfinite(proba_row).all()
    assert test_mask.sum() > 0
    print(f"fold {test_block}: AUC={m['roc_auc']:.3f} F1={m['f1']:.3f} "
          f"acc={m['accuracy']:.3f} removed_xblock={fr.n_removed_cross_block} "
          f"best_epoch={fr.best_epoch}")


def test_loss_decreases():
    """Explicit check that the training loss goes down over a few epochs."""
    import torch
    import torch.nn.functional as Fn
    df = _load()
    y_row = T.build_T1a(df).to_numpy()
    fold_block = S.spatial_block_folds(df, k=SMOKE_BLOCKS)
    cols = C.feature_columns(include_location=False, cocontam="core")
    wg = G.build_well_graph(df, fold_block=fold_block, k=8, cap_km=1.5, cut_blocks=True)
    y_node = G.well_majority_target(df, y_row, wg.well_ids)
    fit = np.ones(len(wg.well_ids), dtype=bool)
    X, _, _ = G.node_features(df, wg.well_ids, cols, train_node_mask=fit,
                              y_node=y_node, encode="frequency")
    gnn.set_seed(C.SEED)
    model = gnn.build_model("graphsage", in_dim=X.shape[1], hidden=32, layers=2, dropout=0.0)
    x = torch.tensor(X); ei = torch.tensor(wg.edge_index, dtype=torch.long)
    ew = gnn.edge_weight_from_dist(wg.edge_dist, 1.5)
    y = torch.tensor(y_node, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    model.train()
    for _ in range(30):
        opt.zero_grad()
        out = model(x, ei, ew)
        loss = Fn.binary_cross_entropy_with_logits(out, y)
        losses.append(float(loss.detach()))
        loss.backward(); opt.step()
        assert torch.isfinite(loss)
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "loss did not decrease"


def test_bipartite_graph_builds():
    df = _load()
    fold_block = S.spatial_block_folds(df, k=SMOKE_BLOCKS)
    bg = G.build_bipartite_well_analyte(df, fold_block=fold_block)
    assert len(bg.analytes) == len(C.T2_LABELS)
    assert bg.edge_well.shape == bg.edge_analyte.shape == bg.edge_label.shape
    assert bg.edge_well.shape[0] > 0
    assert set(np.unique(bg.edge_label)).issubset({0, 1})
    print(f"bipartite: {len(bg.well_ids)} wells x {len(bg.analytes)} analytes, "
          f"{bg.edge_well.shape[0]} measured-cell edges, "
          f"positive rate={bg.edge_label.mean():.3f}")


if __name__ == "__main__":
    t0 = time.time()
    test_pyg_imports_cpu()
    test_graph_builds_and_caps_distance()
    test_c4_zero_cross_block_edges()
    test_loss_decreases()
    test_train_one_fold_loss_finite_and_metrics()
    test_bipartite_graph_builds()
    dt = time.time() - t0
    print(f"\nSMOKE OK in {dt:.1f}s")
