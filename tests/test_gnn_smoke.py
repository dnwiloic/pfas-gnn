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
from src import data, graph as G, gnn, gnn_bipartite as GB, gnn_hetero as GH, splits as S, targets as T


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


# --------------------------------------------------------------- P0: robust early-stop
def test_p0_robust_val_is_multiblock():
    """P0: the early-stop validation spans SEVERAL train micro-blocks (not a single block)
    and one fold trains to finite metrics."""
    df = _load()
    y_row = T.build_T1a(df).to_numpy()
    fold_block = S.spatial_block_folds(df, k=SMOKE_BLOCKS)
    cols = C.feature_columns(include_location=False, cocontam="core")
    test_block = int(np.bincount(fold_block).argmax())
    fr, proba_row, test_mask = gnn.train_eval_fold(
        df, y_row, cols, fold_block, test_block,
        model_name="graphsage", k=8, cap_km=1.5, cut_blocks=True,
        encode="frequency", hidden=32, layers=2, dropout=0.3,
        max_epochs=60, patience=20, n_val_micro=5, seed=C.SEED, verbose=False)
    assert fr.n_val_micro >= 2, "validation must assemble >=2 micro-blocks (P0)"
    assert fr.n_val_nodes > 0
    m = fr.metrics_spatial
    assert np.isfinite(m["accuracy"]) and 0.0 <= m["accuracy"] <= 1.0
    print(f"P0 fold {test_block}: val_micro={fr.n_val_micro} val_nodes={fr.n_val_nodes} "
          f"AUC={m['roc_auc']:.3f} best_epoch={fr.best_epoch}")


# --------------------------------------------------------------- P1: bipartite completion
def test_p1_bipartite_label_matrix():
    df = _load()
    well_ids, Yw, Mw, w2n = GB.well_label_matrix(df, C.T2_LABELS)
    assert Yw.shape == Mw.shape == (len(well_ids), len(C.T2_LABELS))
    assert Mw.any(), "no measured cells in the well x analyte matrix"
    assert set(np.unique(Yw[Mw])).issubset({0, 1})
    print(f"P1 matrix: {len(well_ids)} wells x {len(C.T2_LABELS)} analytes, "
          f"{int(Mw.sum())} measured cells, well-level positive rate={Yw[Mw].mean():.3f}")


def test_p1_bipartite_loss_decreases_and_no_cross_block():
    """P1: bipartite completion trains end-to-end, loss is finite/decreasing, and the
    measured-cell edges never cross a CV block (C4 by construction)."""
    import torch
    import torch.nn.functional as Fn
    df = _load()
    cols = C.feature_columns(include_location=False, cocontam="core")
    labels = C.T2_LABELS
    well_ids, Yw, Mw, w2n = GB.well_label_matrix(df, labels)
    fold_block = S.spatial_block_folds(df, k=SMOKE_BLOCKS)
    bdf = __import__("pandas").DataFrame({"w": df[C.WELL_ID].to_numpy(), "b": fold_block})
    per_well = bdf.groupby("w")["b"].agg(lambda s: int(s.iloc[0]))
    well_block = per_well.reindex(well_ids).to_numpy().astype(int)
    test_block = int(np.bincount(well_block).argmax())

    # explicit short loss-trajectory check on the fit edges
    GB.set_seed(C.SEED)
    train_wells = well_block != test_block
    X, _, _ = G.node_features(df, well_ids, cols, train_node_mask=train_wells,
                              encode="frequency")
    n_lab = len(labels)
    tr_w, tr_a, tr_y = [], [], []
    for w in range(len(well_ids)):
        for j in range(n_lab):
            if Mw[w, j] and train_wells[w]:
                tr_w.append(w); tr_a.append(j); tr_y.append(int(Yw[w, j]))
    tr_w = np.array(tr_w); tr_a = np.array(tr_a); tr_y = np.array(tr_y, dtype=np.float32)
    x = torch.tensor(X)
    aid = torch.arange(n_lab)
    ea2w = torch.tensor(np.vstack([tr_a, tr_w]), dtype=torch.long)
    ew2a = torch.tensor(np.vstack([tr_w, tr_a]), dtype=torch.long)
    model = GB.build_completion_model(X.shape[1], n_lab, emb_dim=16, hidden=32, layers=2,
                                      dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    pw = torch.tensor(tr_w, dtype=torch.long); pa = torch.tensor(tr_a, dtype=torch.long)
    yt = torch.tensor(tr_y)
    losses = []
    model.train()
    for _ in range(25):
        opt.zero_grad()
        out = model(x, aid, ea2w, ew2a, pw, pa)
        loss = Fn.binary_cross_entropy_with_logits(out, yt)
        losses.append(float(loss.detach())); loss.backward(); opt.step()
        assert torch.isfinite(loss)
    print(f"P1 bipartite loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "bipartite loss did not decrease"
    # C4: every measured-cell edge touches one well = one block -> 0 cross-block.
    cross = int((well_block[tr_w] == test_block).sum())  # all train edges -> none in test
    assert cross == 0, "train bipartite edges touch a test-block well — leakage"


def test_p1_bipartite_cv_masked_metrics():
    """P1 end-to-end: a 3-block spatial CV produces the 5 masked multilabel metrics
    (comparable to the T2 wall) with macro-AUROC a finite number in (0,1)."""
    df = _load()
    res = GB.run_t2_bipartite_cv(
        df, regime="spatial", n_blocks=3, emb_dim=16, hidden=32, layers=2,
        dropout=0.2, max_epochs=40, patience=15, seed=C.SEED, verbose=False)
    for key in ("macro_AUROC", "micro_F1", "micro_recall", "micro_precision"):
        assert key in res and np.isfinite(res[key])
    assert 0.0 < res["macro_AUROC"] < 1.0
    assert res["n_cross_block_edges"] == 0, "C4 violated in bipartite CV"
    pl = res["per_label"]
    assert len(pl) == len(C.T2_LABELS)
    print(f"P1 CV(3 blk): macro_AUROC={res['macro_AUROC']:.3f} "
          f"micro_F1={res['micro_F1']:.3f} cross_block_edges={res['n_cross_block_edges']}")


# -------------------------------------- HYBRID T1: mechanistic edges + embed primitive
def test_mechanistic_subbasin_graph_builds_and_intra():
    """Mechanistic intra-sub-basin k-NN graph (eval §2.4): builds, every edge within the
    distance cap, every edge connects wells SHARING the sub-basin, and wells with a missing
    sub-basin carry no mechanistic edge."""
    df = _load()
    fold_block = S.spatial_block_folds(df, k=SMOKE_BLOCKS)
    wg = G.build_well_graph(df, fold_block=fold_block, relation="subbasin_knn",
                            k=8, cap_km=2.0, cut_blocks=False)
    assert wg.meta["relation"] == "subbasin_knn"
    assert wg.edge_index.shape[0] == 2
    if wg.edge_index.shape[1] > 0:
        assert wg.edge_dist.max() <= 2.0 + 1e-6, "mechanistic edge exceeds the 2 km cap"
        sub = G.well_subbasin(df, wg.well_ids)
        a, b = wg.edge_index[0], wg.edge_index[1]
        same = sub[a] == sub[b]
        assert bool(same.all()), "a mechanistic edge connects two DIFFERENT sub-basins"
    n_miss = wg.meta["n_wells_missing_subbasin"]
    # missing-subbasin wells are isolated under this relation (no edge touches them)
    if n_miss > 0 and wg.edge_index.shape[1] > 0:
        sub = G.well_subbasin(df, wg.well_ids)
        miss_nodes = set(np.where(_isna(sub))[0].tolist())
        touched = set(wg.edge_index.ravel().tolist())
        assert miss_nodes.isdisjoint(touched), "a missing-subbasin well got a mechanistic edge"
    print(f"mechanistic: nodes={wg.meta['n_nodes']} undirected_edges={wg.meta['n_edges_undirected']} "
          f"max_km={wg.edge_dist.max() if wg.edge_index.shape[1] else 0:.3f} "
          f"missing_subbasin={n_miss}")


def _isna(arr):
    import pandas as pd
    return pd.isna(arr)


def test_c4_zero_cross_block_per_relation():
    """C4 applies to BOTH relations: after the cut, 0 edge crosses a CV block boundary for
    the bare spatial AND the mechanistic graph, and the uncut graph DID have some (so the
    cut is meaningful)."""
    df = _load()
    fold_block = S.spatial_block_folds(df, k=SMOKE_BLOCKS)
    for rel, cap in [("spatial", 1.5), ("subbasin_knn", 2.0)]:
        wg = G.build_well_graph(df, fold_block=fold_block, relation=rel, k=8,
                                cap_km=cap, cut_blocks=True)
        a, b = wg.edge_index[0], wg.edge_index[1]
        cross = int((wg.node_block[a] != wg.node_block[b]).sum())
        assert cross == 0, f"{rel}: {cross} cross-block edges after C4 — violated"
        wgu = G.build_well_graph(df, fold_block=fold_block, relation=rel, k=8,
                                 cap_km=cap, cut_blocks=False)
        au, bu = wgu.edge_index[0], wgu.edge_index[1]
        cross_uncut = int((wgu.node_block[au] != wgu.node_block[bu]).sum())
        assert cross_uncut == 2 * wg.n_removed_cross_block
        print(f"C4[{rel}]: removed(undir)={wg.n_removed_cross_block} remaining={cross} "
              f"would-be(dir,uncut)={cross_uncut}")


def test_embed_returns_prehead_shape():
    """build_model(...).forward(embed=True) returns the PRE-HEAD hidden embedding
    [n_nodes, hidden], distinct from the [n_nodes] logits."""
    import torch
    df = _load()
    y_row = T.build_T1a(df).to_numpy()
    fold_block = S.spatial_block_folds(df, k=SMOKE_BLOCKS)
    cols = C.feature_columns(include_location=False, cocontam="core")
    wg = G.build_well_graph(df, fold_block=fold_block, relation="subbasin_knn",
                            k=8, cap_km=2.0, cut_blocks=True)
    y_node = G.well_majority_target(df, y_row, wg.well_ids)
    fit = np.ones(len(wg.well_ids), dtype=bool)
    X, _, _ = G.node_features(df, wg.well_ids, cols, train_node_mask=fit,
                              y_node=y_node, encode="frequency")
    gnn.set_seed(C.SEED)
    HID = 48
    model = gnn.build_model("graphsage", in_dim=X.shape[1], hidden=HID, layers=2, dropout=0.0)
    model.eval()
    x = torch.tensor(X); ei = torch.tensor(wg.edge_index, dtype=torch.long)
    ew = gnn.edge_weight_from_dist(wg.edge_dist, 2.0)
    with torch.no_grad():
        emb = model(x, ei, ew, embed=True)
        logit = model(x, ei, ew)
    assert tuple(emb.shape) == (len(wg.well_ids), HID), "embedding is not [n_nodes, hidden]"
    assert tuple(logit.shape) == (len(wg.well_ids),), "logit is not [n_nodes]"
    assert model.embed_dim == HID
    print(f"embed(): emb={tuple(emb.shape)} logit={tuple(logit.shape)} embed_dim={model.embed_dim}")


def test_train_gnn_and_embed_primitive():
    """The hybrid primitive: trains GraphSAGE on FIT blocks only, returns pre-head embeddings
    for the EMBED block — with 0 cross-block edges (C4, mechanistic relation), finite loss,
    and embed nodes that live exactly in the embed block (not the fit blocks)."""
    df = _load()
    y_row = T.build_T1a(df).to_numpy()
    fold_block = S.spatial_block_folds(df, k=SMOKE_BLOCKS)
    cols = C.feature_columns(include_location=False, cocontam="core")
    blocks = sorted(set(fold_block.tolist()))
    embed_b = [blocks[0]]
    fit_b = blocks[1:]
    emb, info = gnn.train_gnn_and_embed(
        df, y_row, cols, fold_block, fit_blocks=fit_b, embed_blocks=embed_b,
        relation="subbasin_knn", k=8, cap_km=2.0, model_name="graphsage",
        hidden=32, layers=2, dropout=0.3, max_epochs=30, patience=12,
        n_val_micro=4, seed=C.SEED)
    assert emb.shape[1] == 32 == info.embed_dim, "embedding width != hidden"
    assert emb.shape[0] == info.n_embed_nodes > 0, "no embed-node embeddings produced"
    assert info.n_cross_block_remaining == 0, "C4 violated in the embed graph"
    assert np.isfinite(emb).all(), "non-finite embedding"
    assert np.isfinite(info.final_loss), "non-finite final loss"
    # embed nodes belong ONLY to the embed block (never trained on their label)
    nb = G.build_well_graph(df, fold_block=fold_block, relation="subbasin_knn",
                            k=8, cap_km=2.0).node_block
    assert set(nb[info.embed_node_idx].tolist()) == set(embed_b), \
        "embed nodes leak into fit blocks"
    print(f"primitive: emb={emb.shape} fit_nodes={info.n_fit_nodes} embed_nodes={info.n_embed_nodes} "
          f"removed_xblock={info.n_removed_cross_block} cross_remaining={info.n_cross_block_remaining} "
          f"best_epoch={info.best_epoch} loss={info.final_loss:.4f}")


# ----------------------------------------------- P1+: heterogeneous bipartite + well edges
def _hetero_setup(test_block_only=True):
    """Shared fixture: labels matrix, per-well block, one test block."""
    df = _load()
    cols = C.feature_columns(include_location=False, cocontam="core")
    labels = C.T2_LABELS
    well_ids, Yw, Mw, w2n = GH.GB.well_label_matrix(df, labels)
    import pandas as pd
    fold_block = S.spatial_block_folds(df, k=SMOKE_BLOCKS)
    bdf = pd.DataFrame({"w": df[C.WELL_ID].to_numpy(), "b": fold_block})
    per_well = bdf.groupby("w")["b"].agg(lambda s: int(s.iloc[0]))
    well_block = per_well.reindex(well_ids).to_numpy().astype(int)
    test_block = int(np.bincount(well_block).argmax())
    return df, cols, labels, well_ids, Yw, Mw, w2n, well_block, test_block


def test_p1plus_hetero_fold_c4_audited():
    """P1+: a heterogeneous fold trains end-to-end; the C4 audit shows 0 cross-block
    bipartite edges AND 0 cross-block well<->well edges remaining (the well edges DID
    cross before the cut, so the cut is meaningful), and metrics are finite."""
    df, cols, labels, well_ids, Yw, Mw, w2n, well_block, test_block = _hetero_setup()
    fr, P_well, test_mask = GH.train_eval_hetero_fold(
        df, well_ids, Yw, Mw, w2n, cols, well_block, test_block,
        encoder="hetero_sage", decoder="mlp", emb_dim=16, hidden=32, layers=2,
        dropout=0.2, k=8, cap_km=1.5, max_epochs=40, patience=15, seed=C.SEED)
    assert fr.n_cross_block_bip == 0, "bipartite edge crosses a block — C4 violated"
    assert fr.n_cross_block_well == 0, "well<->well edge crosses a block after cut — C4 violated"
    assert fr.n_well_edges_train > 0, "no spatial well edges survived — graph too sparse"
    assert test_mask.sum() > 0
    # at least some test cells got a prediction
    assert np.isfinite(P_well[test_mask]).any()
    print(f"P1+ fold {test_block}: tr_edges={fr.n_train_edges} te_edges={fr.n_test_edges} "
          f"well_edges={fr.n_well_edges_train} removed_well_xblock={fr.n_removed_well_cross} "
          f"cross_well={fr.n_cross_block_well} cross_bip={fr.n_cross_block_bip} "
          f"best_epoch={fr.best_epoch}")


def test_p1plus_well_edges_did_cross_before_cut():
    """The C4 cut is MEANINGFUL: the uncut capped k-NN well graph HAS cross-block edges,
    and the cut removes exactly them (so n_removed_well_cross > 0 in a dense smoke)."""
    df, cols, labels, well_ids, Yw, Mw, w2n, well_block, test_block = _hetero_setup()
    _, coords, _ = G.well_table(df)
    ei_full, ed_full = G.knn_edges_km(coords, k=8, cap_km=1.5)
    cross_before = int((well_block[ei_full[0]] != well_block[ei_full[1]]).sum())
    _, _, removed = G.cut_cross_block(ei_full, ed_full, well_block)
    assert cross_before > 0, "smoke too sparse to exercise the well-edge C4 cut"
    assert removed == cross_before
    print(f"P1+ well-edge C4: cross_before_cut={cross_before} removed={removed}")


def test_p1plus_hetero_loss_decreases_all_encoders():
    """P1+: hetero_sage / hgt / rgcn encoders and mlp / vgae decoders all train with a
    finite, decreasing loss on the fit edges (a few epochs)."""
    import torch
    df, cols, labels, well_ids, Yw, Mw, w2n, well_block, test_block = _hetero_setup()
    train_wells = well_block != test_block
    X, _, _ = G.node_features(df, well_ids, cols, train_node_mask=train_wells,
                              encode="frequency")
    n_lab = len(labels)
    from torch_geometric.utils import to_undirected
    # build the train edge_dict once
    tr_w, tr_a, tr_y = [], [], []
    for w in range(len(well_ids)):
        for j in range(n_lab):
            if Mw[w, j] and train_wells[w]:
                tr_w.append(w); tr_a.append(j); tr_y.append(int(Yw[w, j]))
    tr_w = np.array(tr_w); tr_a = np.array(tr_a); tr_y = np.array(tr_y, np.float32)
    _, coords, _ = G.well_table(df)
    ei_full, ed_full = G.knn_edges_km(coords, k=8, cap_km=1.5)
    ei_cut, _, _ = G.cut_cross_block(ei_full, ed_full, well_block)
    keep = train_wells[ei_cut[0]] & train_wells[ei_cut[1]]
    ew2w = to_undirected(torch.tensor(ei_cut[:, keep], dtype=torch.long), num_nodes=len(well_ids))
    edge_dict = {
        ("analyte", "measured_by", "well"): torch.tensor(np.vstack([tr_a, tr_w]), dtype=torch.long),
        ("well", "measures", "analyte"): torch.tensor(np.vstack([tr_w, tr_a]), dtype=torch.long),
        ("well", "near", "well"): ew2w,
    }
    x = torch.tensor(X); aid = torch.arange(n_lab)
    pw = torch.tensor(tr_w, dtype=torch.long); pa = torch.tensor(tr_a, dtype=torch.long)
    yt = torch.tensor(tr_y)
    for enc, dec in [("hetero_sage", "mlp"), ("hgt", "mlp"), ("rgcn", "mlp"),
                     ("hetero_sage", "vgae")]:
        GH.set_seed(C.SEED)
        model = GH.build_hetero_model(X.shape[1], n_lab, encoder=enc, decoder=dec,
                                      emb_dim=16, hidden=32, layers=2, dropout=0.0, heads=2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        model.train()
        for _ in range(25):
            opt.zero_grad()
            logit, kl = model(x, aid, edge_dict, pw, pa)
            loss = GH._bce_focal(logit, yt, None, gamma=0.0) + 1e-3 * kl
            losses.append(float(loss.detach())); loss.backward(); opt.step()
            assert torch.isfinite(loss)
        print(f"P1+ {enc}/{dec} loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
        assert losses[-1] < losses[0], f"{enc}/{dec} loss did not decrease"


def test_p1plus_hetero_cv_masked_metrics_and_c4():
    """P1+ end-to-end: a 3-block spatial CV produces the 5 masked multilabel metrics
    (comparable to the wall AND P1), macro-AUROC finite in (0,1), AND the C4 audit is
    exactly 0 for BOTH the bipartite and the well<->well relations."""
    df, *_ = _hetero_setup()
    res = GH.run_t2_hetero_cv(
        df, regime="spatial", n_blocks=3, encoder="hetero_sage", decoder="mlp",
        emb_dim=16, hidden=32, layers=2, dropout=0.2, k=8, cap_km=1.5,
        max_epochs=40, patience=15, gamma=1.0, seed=C.SEED)
    for key in ("macro_AUROC", "micro_F1", "micro_recall", "micro_precision"):
        assert key in res and np.isfinite(res[key])
    assert 0.0 < res["macro_AUROC"] < 1.0
    assert res["n_cross_block_bipartite"] == 0, "C4 bipartite violated"
    assert res["n_cross_block_well"] == 0, "C4 well<->well violated"
    assert res["n_cross_block_edges"] == 0
    assert res["n_removed_well_cross_total"] >= 0
    pl = res["per_label"]
    assert len(pl) == len(C.T2_LABELS)
    # per-label AP present (pr_auc) for the PFNA-vs-AP comparison
    assert all("pr_auc" in r for r in pl)
    print(f"P1+ CV(3 blk): macro_AUROC={res['macro_AUROC']:.3f} micro_F1={res['micro_F1']:.3f} "
          f"cross_bip={res['n_cross_block_bipartite']} cross_well={res['n_cross_block_well']} "
          f"removed_well={res['n_removed_well_cross_total']}")


if __name__ == "__main__":
    t0 = time.time()
    test_pyg_imports_cpu()
    test_graph_builds_and_caps_distance()
    test_c4_zero_cross_block_edges()
    test_loss_decreases()
    test_train_one_fold_loss_finite_and_metrics()
    test_bipartite_graph_builds()
    test_p0_robust_val_is_multiblock()
    test_p1_bipartite_label_matrix()
    test_p1_bipartite_loss_decreases_and_no_cross_block()
    test_p1_bipartite_cv_masked_metrics()
    test_mechanistic_subbasin_graph_builds_and_intra()
    test_c4_zero_cross_block_per_relation()
    test_embed_returns_prehead_shape()
    test_train_gnn_and_embed_primitive()
    test_p1plus_hetero_fold_c4_audited()
    test_p1plus_well_edges_did_cross_before_cut()
    test_p1plus_hetero_loss_decreases_all_encoders()
    test_p1plus_hetero_cv_masked_metrics_and_c4()
    dt = time.time() - t0
    print(f"\nSMOKE OK in {dt:.1f}s")
