"""T2 by MATRIX COMPLETION on a bipartite wells x analytes graph (P1, MNAR-aware).

Why this track (HYDRO_CRITIQUE / baseline_t2 §5): the wells x analyte exceedance matrix is
lacunar and MNAR (a reduced lab panel measures only some analytes). The non-graph wall
(BinaryRelevance) trains one model per label independently and ignores the matrix structure;
classifier chains did NOT help (the co-occurrence is already mediated by shared context).
A bipartite completion model instead learns ONE shared encoder and an analyte embedding, so
a well's prediction for analyte a borrows strength from (i) its context features, (ii) the
other analytes measured at that well, and (iii) other wells' analyte patterns — the GAE /
link-prediction / IGMC idea, here as inductive link-LABEL prediction.

Graph
-----
* LEFT nodes  = wells (gm_well_id), feature = FeaturePipeline context (anti-leak, frequency
  encoded, fit on TRAIN wells only). lat/lon are NOT node features (C6).
* RIGHT nodes = the 10 T2 analytes, feature = a learned embedding (id -> vector).
* EDGES = measured (well, analyte) cells (baselines_t2.measurement_mask — availability only,
  no value, no leakage). Each edge carries the binary well-majority exceedance label
  (targets.build_T2 aggregated to the well, identical contract to the wall).

Encoder / decoder
-----------------
* A bipartite SAGE encoder: wells aggregate from their measured analytes and analytes from
  their wells (heterogeneous message passing). MESSAGE PASSING USES ONLY TRAIN EDGES, so a
  test well's representation is built inductively from the train-side structure + its own
  features — no test label ever enters the graph (transductive-free on labels).
* Decoder = a small MLP on [well_emb ; analyte_emb ; hadamard] -> exceedance logit. Trained
  with BCE on TRAIN edges only; the per-label class imbalance is handled by a per-label
  pos_weight (mirrors the wall's class_weight='balanced').

Leakage controls (eval C2/C4)
-----------------------------
* OUTER CV is spatial-block at the WELL level (splits.spatial_block_folds), so train and
  test share NO well and NO spatial block. The well-analyte edges never cross a block (an
  edge touches exactly one well = one block), so C4 holds by construction; we assert it.
* Random CV (group_random_folds) is run only to report the random-minus-spatial Delta.
* Evaluation is row-level for strict comparability with the wall: the (well, analyte)
  probability is broadcast to every sampling row of that well, then the 5 metrics are
  computed with the per-label measurement mask via metrics.multilabel_metrics.

This module is torch-importing but smoke-testable on CPU (tiny well subsample).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import baselines_t2 as B2
from . import config as C
from . import features as F
from . import graph as G
from . import metrics as MM
from . import splits as S


# --------------------------------------------------------------------------- torch
def set_seed(seed: int = C.SEED):
    import random
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------- well-level label matrix
def well_label_matrix(df, labels):
    """Aggregate the row-level masked T2 matrix to the WELL level.

    Returns (well_ids, Yw[n_wells, n_labels] int, Mw[n_wells, n_labels] bool):
      * Mw[w, j] = analyte j was measured in >=1 sampling of well w (the bipartite edge set);
      * Yw[w, j] = well-MAJORITY exceedance of analyte j over the measured samplings (the
        same well-majority contract used for T1 nodes). Y on a not-measured cell is masked.
    """
    well_ids, _, well_to_node = G.well_table(df)
    n = len(well_ids)
    Y, M = B2.masked_targets(df, labels=labels)
    wcol = df[C.WELL_ID].to_numpy()
    Yw = np.zeros((n, len(labels)), dtype=np.int64)
    Mw = np.zeros((n, len(labels)), dtype=bool)
    for j, a in enumerate(labels):
        col = f"label_{a}"
        meas = M[col].to_numpy()
        if not meas.any():
            continue
        sub = pd.DataFrame({"w": wcol[meas], "y": Y[col].to_numpy()[meas]})
        agg = sub.groupby("w")["y"].mean()
        for w, frac in agg.items():
            i = well_to_node[w]
            Mw[i, j] = True
            Yw[i, j] = int(frac >= 0.5)
    return well_ids, Yw, Mw, well_to_node


# --------------------------------------------------------------------------- model
def build_completion_model(in_dim, n_analytes, *, emb_dim=32, hidden=64, layers=2,
                           dropout=0.3):
    """Bipartite SAGE encoder (wells<->analytes) + bilinear/MLP edge-label decoder.

    The encoder runs `layers` rounds of heterogeneous mean aggregation over the TRAIN
    bipartite edges; the decoder scores a (well, analyte) pair from the concatenation of
    their encoded embeddings and their hadamard product. Returns an nn.Module whose
    forward(x_well, analyte_ids, edge_w2a, edge_a2w, pair_well, pair_analyte) -> logits."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    from torch_geometric.nn import SAGEConv

    class BipartiteCompletion(nn.Module):
        def __init__(self):
            super().__init__()
            self.analyte_emb = nn.Embedding(n_analytes, emb_dim)
            self.well_in = nn.Linear(in_dim, hidden)
            self.analyte_in = nn.Linear(emb_dim, hidden)
            # two directed SAGE convs per layer: analyte->well and well->analyte
            self.conv_a2w = nn.ModuleList([SAGEConv((hidden, hidden), hidden, aggr="mean")
                                           for _ in range(layers)])
            self.conv_w2a = nn.ModuleList([SAGEConv((hidden, hidden), hidden, aggr="mean")
                                           for _ in range(layers)])
            self.norm_w = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.norm_a = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.dec = nn.Sequential(
                nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1))
            self.dropout = dropout

        def encode(self, x_well, analyte_ids, edge_a2w, edge_w2a):
            hw = Fn.relu(self.well_in(x_well))
            ha = Fn.relu(self.analyte_in(self.analyte_emb(analyte_ids)))
            for i in range(len(self.conv_a2w)):
                # analytes -> wells : src=analyte, dst=well
                hw_new = self.conv_a2w[i]((ha, hw), edge_a2w)
                # wells -> analytes : src=well, dst=analyte
                ha_new = self.conv_w2a[i]((hw, ha), edge_w2a)
                hw = Fn.dropout(Fn.relu(self.norm_w[i](hw_new)), p=self.dropout,
                                training=self.training)
                ha = Fn.dropout(Fn.relu(self.norm_a[i](ha_new)), p=self.dropout,
                                training=self.training)
            return hw, ha

        def decode(self, hw, ha, pair_well, pair_analyte):
            zw = hw[pair_well]; za = ha[pair_analyte]
            feat = torch.cat([zw, za, zw * za], dim=-1)
            return self.dec(feat).squeeze(-1)

        def forward(self, x_well, analyte_ids, edge_a2w, edge_w2a, pair_well, pair_analyte):
            hw, ha = self.encode(x_well, analyte_ids, edge_a2w, edge_w2a)
            return self.decode(hw, ha, pair_well, pair_analyte)

    return BipartiteCompletion()


# --------------------------------------------------------------------------- one fold
@dataclass
class BipFold:
    fold: int
    n_train_edges: int
    n_test_edges: int
    best_epoch: int
    n_cross_block_edges: int


def train_eval_bipartite_fold(df, well_ids, Yw, Mw, well_to_node, feature_cols,
                              well_block, test_block, *, emb_dim=32, hidden=64, layers=2,
                              dropout=0.3, lr=5e-3, weight_decay=5e-4, max_epochs=300,
                              patience=40, val_frac=0.15, seed=C.SEED, labels=None,
                              verbose=False):
    """Train the completion model on TRAIN-block wells' measured cells, predict the
    exceedance on TEST-block wells' measured cells. Returns (BipFold, P_well[n_wells,
    n_labels] proba on test wells filled, test_well_mask)."""
    import torch
    import torch.nn.functional as Fn

    labels = labels or C.T2_LABELS
    set_seed(seed)
    dev = device()
    n_wells = len(well_ids)
    n_lab = len(labels)

    test_wells = well_block == test_block
    train_wells = ~test_wells

    # node features fit on TRAIN wells only (anti-leak), frequency encode (no y needed)
    fit_mask = train_wells.copy()
    X, _, _ = G.node_features(df, well_ids, feature_cols, train_node_mask=fit_mask,
                              encode="frequency")
    x = torch.tensor(X, dtype=torch.float32, device=dev)
    analyte_ids = torch.arange(n_lab, dtype=torch.long, device=dev)

    # ---- edge sets. An edge = a measured cell (well w, analyte j). TRAIN edges = measured
    # cells of TRAIN wells; TEST edges = measured cells of TEST wells. By construction an
    # edge touches one well = one block, so no edge crosses a block (assert C4).
    tr_w, tr_a, tr_y = [], [], []
    te_w, te_a = [], []
    n_cross = 0
    for w in range(n_wells):
        for j in range(n_lab):
            if not Mw[w, j]:
                continue
            if train_wells[w]:
                tr_w.append(w); tr_a.append(j); tr_y.append(int(Yw[w, j]))
            else:
                te_w.append(w); te_a.append(j)
            # cross-block check is trivially 0 (one well per edge) but we audit it anyway
    tr_w = np.asarray(tr_w); tr_a = np.asarray(tr_a); tr_y = np.asarray(tr_y, dtype=np.float32)
    te_w = np.asarray(te_w); te_a = np.asarray(te_a)

    # message-passing edges (TRAIN only -> inductive): analyte->well and well->analyte.
    # bipartite SAGE expects edge_index[2,E] with src in the first node set, dst in second.
    ea2w = torch.tensor(np.vstack([tr_a, tr_w]), dtype=torch.long, device=dev)  # (analyte,well)
    ew2a = torch.tensor(np.vstack([tr_w, tr_a]), dtype=torch.long, device=dev)  # (well,analyte)

    # validation split: hold out a random slice of TRAIN edges (for early stop). The wells
    # stay train-side, so no test leakage; the held-out edges are masked in the loss only.
    rng = np.random.RandomState(seed)
    n_tr = len(tr_w)
    val_n = max(1, int(val_frac * n_tr))
    perm = rng.permutation(n_tr)
    val_idx = perm[:val_n]; fit_idx = perm[val_n:]
    fit_w = torch.tensor(tr_w[fit_idx], dtype=torch.long, device=dev)
    fit_a = torch.tensor(tr_a[fit_idx], dtype=torch.long, device=dev)
    fit_y = torch.tensor(tr_y[fit_idx], dtype=torch.float32, device=dev)
    val_w = torch.tensor(tr_w[val_idx], dtype=torch.long, device=dev)
    val_a = torch.tensor(tr_a[val_idx], dtype=torch.long, device=dev)
    val_y_np = tr_y[val_idx]

    # per-label pos_weight (imbalance handling, like class_weight='balanced')
    pw = np.ones(n_lab, dtype=np.float32)
    for j in range(n_lab):
        yj = tr_y[tr_a == j]
        pos = float(yj.sum()); neg = float(len(yj) - pos)
        pw[j] = (neg / pos) if pos > 0 else 1.0
    pos_weight_edge = torch.tensor(pw[tr_a[fit_idx]], dtype=torch.float32, device=dev)

    model = build_completion_model(X.shape[1], n_lab, emb_dim=emb_dim, hidden=hidden,
                                   layers=layers, dropout=dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=max(5, patience // 4),
                                                       min_lr=1e-5)

    best_val, best_state, best_epoch, bad = -np.inf, None, 0, 0
    for epoch in range(max_epochs):
        model.train(); opt.zero_grad()
        logit = model(x, analyte_ids, ea2w, ew2a, fit_w, fit_a)
        loss = Fn.binary_cross_entropy_with_logits(logit, fit_y, weight=pos_weight_edge)
        loss.backward(); opt.step()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite bipartite loss")

        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(x, analyte_ids, ea2w, ew2a, val_w, val_a)).cpu().numpy()
            try:
                from sklearn.metrics import roc_auc_score
                vauc = roc_auc_score(val_y_np, vp) if len(np.unique(val_y_np)) > 1 else 0.0
            except Exception:
                vauc = 0.0
        sched.step(vauc)
        if vauc > best_val + 1e-4:
            best_val, best_epoch, bad = vauc, epoch, 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and epoch % 25 == 0:
            print(f"    ep{epoch} loss={float(loss):.4f} val_auc={vauc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    P_well = np.full((n_wells, n_lab), np.nan, dtype=np.float64)
    with torch.no_grad():
        if len(te_w):
            tw = torch.tensor(te_w, dtype=torch.long, device=dev)
            ta = torch.tensor(te_a, dtype=torch.long, device=dev)
            pp = torch.sigmoid(model(x, analyte_ids, ea2w, ew2a, tw, ta)).cpu().numpy()
            P_well[te_w, te_a] = pp

    fr = BipFold(fold=int(test_block), n_train_edges=int(n_tr), n_test_edges=int(len(te_w)),
                 best_epoch=int(best_epoch), n_cross_block_edges=int(n_cross))
    return fr, P_well, test_wells


# --------------------------------------------------------------------------- CV runner
def run_t2_bipartite_cv(df, *, feature_cols=None, regime="spatial", labels=None,
                        n_blocks=None, emb_dim=32, hidden=64, layers=2, dropout=0.3,
                        lr=5e-3, max_epochs=300, patience=40, seed=C.SEED, verbose=False):
    """Leave-one-block-out (spatial) or group-random CV for the bipartite completion model.

    Returns (result_dict) with the 5 multilabel metrics computed ROW-LEVEL via
    metrics.multilabel_metrics (strict comparability with the T2 wall), per-label table,
    OOF thresholds, and per-fold diagnostics. `regime`: 'spatial' (reference) | 'random'.
    """
    labels = labels or C.T2_LABELS
    feature_cols = feature_cols or C.feature_columns(include_location=False, cocontam="core")
    n_blocks = n_blocks or (C.N_SPATIAL_BLOCKS if regime == "spatial" else C.N_RANDOM_FOLDS)

    if regime == "spatial":
        fold_block = S.spatial_block_folds(df, k=n_blocks, seed=seed)
    else:
        fold_block = S.group_random_folds(df, k=n_blocks, seed=seed)

    well_ids, Yw, Mw, well_to_node = well_label_matrix(df, labels)
    # per-well block id (a well lives in exactly one block by construction)
    bdf = pd.DataFrame({"w": df[C.WELL_ID].to_numpy(), "b": fold_block})
    nun = bdf.groupby("w")["b"].nunique()
    if int((nun > 1).sum()):
        raise AssertionError("a well straddles >1 block — fold_block must be well-consistent")
    per_well = bdf.groupby("w")["b"].agg(lambda s: int(s.iloc[0]))
    well_block = per_well.reindex(well_ids).to_numpy().astype(int)

    blocks = sorted(set(well_block.tolist()))
    P_well_oof = np.full((len(well_ids), len(labels)), np.nan, dtype=np.float64)
    folds_info = []
    for b in blocks:
        fr, P_well, test_wells = train_eval_bipartite_fold(
            df, well_ids, Yw, Mw, well_to_node, feature_cols, well_block, b,
            emb_dim=emb_dim, hidden=hidden, layers=layers, dropout=dropout, lr=lr,
            max_epochs=max_epochs, patience=patience, seed=seed, labels=labels,
            verbose=verbose)
        # fill OOF predictions for this fold's test wells (measured cells only)
        fill = test_wells[:, None] & Mw
        P_well_oof[fill] = P_well[fill]
        folds_info.append(fr)
        if verbose:
            print(f"[{regime}] block {b}: train_edges={fr.n_train_edges} "
                  f"test_edges={fr.n_test_edges} best_ep={fr.best_epoch}")

    # ---- broadcast well-level OOF predictions to ROWS and score like the wall.
    row_node = df[C.WELL_ID].map(well_to_node).to_numpy().astype(np.int64)
    Y_row, M_row = B2.masked_targets(df, labels=labels)
    n_rows = len(df)
    P_row = np.full((n_rows, len(labels)), np.nan, dtype=np.float64)
    for j in range(len(labels)):
        P_row[:, j] = P_well_oof[row_node, j]
    # where a cell is measured at row level but the well prediction is NaN (well never had a
    # measured majority for that analyte -> shouldn't happen, but guard) fill prevalence
    for j, a in enumerate(labels):
        col = f"label_{a}"
        m = M_row[col].to_numpy()
        nanmask = m & np.isnan(P_row[:, j])
        if nanmask.any():
            prev = float(Y_row[col].to_numpy()[m & ~np.isnan(P_row[:, j])].mean()) \
                if (m & ~np.isnan(P_row[:, j])).any() else 0.0
            P_row[nanmask, j] = prev
    P_row = np.nan_to_num(P_row, nan=0.0)

    # OOF thresholds per label (on the row-level OOF probabilities = the only predictions
    # we have; these ARE out-of-fold because each well was scored only as a test well).
    thr = B2.best_thresholds_oof(Y_row, P_row, M_row, labels)
    full5 = MM.multilabel_metrics(Y_row, P_row, M_row, labels, thr)

    # audit: 0 cross-block bipartite edges (C4 holds by construction)
    total_cross = int(sum(f.n_cross_block_edges for f in folds_info))

    return {
        "regime": regime, "n_blocks": len(blocks),
        "macro_AUROC": float(full5["macro"]["roc_auc"]),
        "micro_AUROC": float(full5["micro"]["roc_auc"]),
        "micro_F1": float(full5["micro"]["f1"]),
        "macro_F1": float(full5["macro"]["f1"]),
        "micro_recall": float(full5["micro"]["recall"]),
        "micro_precision": float(full5["micro"]["precision"]),
        "subset_accuracy": float(full5["subset_accuracy"]),
        "per_label": full5["per_label"].to_dict(orient="records"),
        "thresholds": thr,
        "n_cross_block_edges": total_cross,
        "per_fold": [vars(f) for f in folds_info],
        "P_row": P_row, "Y_row": Y_row, "M_row": M_row,
    }
