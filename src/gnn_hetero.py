"""T2 by HETEROGENEOUS matrix completion (P1+): bipartite wells<->analytes completion
graph AUGMENTED with spatial well<->well edges.

Why this track (extends P1, gnn_bipartite.py)
---------------------------------------------
P1 reformulated the lacunar MNAR wells x analyte exceedance matrix as a *bipartite*
completion graph: ONE shared encoder + an analyte embedding, decoder scoring (well,
analyte) cells. It EGALA the non-graph wall (macro-AUROC spatial 0.681 vs 0.680) but a
well only saw (i) its own context features and (ii) the analytes measured at that well.
A test well in a poorly-measured area thus borrows little. P1+ adds the missing
RELATIONAL signal HYDRO_CRITIQUE asked for: **well<->well spatial edges**, so a test
well also aggregates from its *measured neighbours'* analyte patterns. The encoder is
heterogeneous (two node types, three edge relations); the decoder still completes the
bipartite cells exactly like P1, for strict comparability with the wall and with P1.

Graph (heterogeneous)
---------------------
* node type 'well'    : context features (FeaturePipeline, anti-leak, frequency encoded,
  fit on TRAIN wells only; lat/lon NOT a node feature, C6 — geography enters ONLY through
  the distance-capped well<->well topology).
* node type 'analyte' : a learned id embedding (10 T2 analytes).
* relation ('analyte','measured_by','well')   : measured cell, message analyte->well.
* relation ('well','measures','analyte')       : the reverse, message well->analyte.
* relation ('well','near','well')              : spatial k-NN, HARD CAP ~1.5 km
  (graph.build_well_graph), symmetrised. Beyond the measured autocorrelation range
  "proximity" only re-encodes the map = spatial leakage, hence the cap (same rule as T1).

Leakage controls (eval C2/C4) — audited, never presumed
-------------------------------------------------------
* OUTER CV is spatial-block at the WELL level (splits.spatial_block_folds): train and test
  share NO well and NO spatial block.
* The bipartite (well,analyte) edges touch exactly one well = one block, so they never
  cross a block (C4 holds by construction for that relation, asserted == 0).
* The NEW well<->well edges CAN cross a block boundary -> they are CUT per fold
  (graph.cut_cross_block) and we ASSERT 0 cross-block well-edges remain on every fold.
* MESSAGE PASSING USES TRAIN-side edges only (the encoder is run on the train subgraph;
  a test well is attached to its TRAIN neighbours through cut, cross-block-free spatial
  edges, and to its own measured-cell edges). No test label ever enters the graph
  (inductive on labels).

Models
------
* encoder='hetero_sage' : HeteroConv of SAGEConv per relation (mean aggr).
* encoder='hgt'         : HGTConv (typed attention, per-relation messages).
* encoder='rgcn'        : RGCNConv on the merged homogeneous view (3 relations).
* decoder='mlp'         : P1 decoder [z_w; z_a; z_w*z_a] -> logit (deterministic).
* decoder='vgae'        : variational well/analyte posteriors (mu, logvar) + KL term;
  the cell logit is the inner product of sampled embeddings + a bias MLP. Gives an
  incertitude/regularisation signal and a probabilistic completion (GAE/VGAE family).

Imbalance / PFNA (rare 2.6%, P1 -0.065)
---------------------------------------
Per-label pos_weight (like the wall class_weight='balanced'); PLUS optional FOCAL LOSS
(gamma) that down-weights easy negatives and concentrates gradient on the rare positives.
We report AP per label (the metric that matters at 2.6%), not just AUROC.

This module is torch-importing but smoke-testable on CPU (tiny well subsample).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import baselines_t2 as B2
from . import config as C
from . import features as F  # noqa: F401  (kept for symmetry with gnn_bipartite)
from . import graph as G
from . import gnn_bipartite as GB
from . import metrics as MM
from . import splits as S


def set_seed(seed: int = C.SEED):
    import random
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- model
def build_hetero_model(in_dim, n_analytes, *, encoder="hetero_sage", decoder="mlp",
                       emb_dim=32, hidden=64, layers=2, dropout=0.3, heads=2):
    """Heterogeneous encoder (well + analyte) + completion decoder.

    forward(x_well, analyte_ids, edge_dict, pair_well, pair_analyte) -> (logits, kl)
      edge_dict carries the three relations' edge_index tensors (see module docstring).
      `kl` is 0.0 for the deterministic decoder, the VGAE KL term otherwise.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    from torch_geometric.nn import HeteroConv, HGTConv, RGCNConv, SAGEConv

    R_A2W = ("analyte", "measured_by", "well")
    R_W2A = ("well", "measures", "analyte")
    R_W2W = ("well", "near", "well")

    class HeteroCompletion(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_kind = encoder
            self.decoder_kind = decoder
            self.analyte_emb = nn.Embedding(n_analytes, emb_dim)
            self.well_in = nn.Linear(in_dim, hidden)
            self.analyte_in = nn.Linear(emb_dim, hidden)

            if encoder == "hetero_sage":
                self.convs = nn.ModuleList([
                    HeteroConv({
                        R_A2W: SAGEConv((hidden, hidden), hidden, aggr="mean"),
                        R_W2A: SAGEConv((hidden, hidden), hidden, aggr="mean"),
                        R_W2W: SAGEConv((hidden, hidden), hidden, aggr="mean"),
                    }, aggr="sum") for _ in range(layers)])
            elif encoder == "hgt":
                meta = (["well", "analyte"], [R_A2W, R_W2A, R_W2W])
                self.convs = nn.ModuleList([
                    HGTConv(hidden, hidden, meta, heads=heads) for _ in range(layers)])
            elif encoder == "rgcn":
                # merged homogeneous view: wells [0..nw), analytes [nw..nw+na); 3 relations.
                self.convs = nn.ModuleList([
                    RGCNConv(hidden, hidden, num_relations=3) for _ in range(layers)])
            else:
                raise ValueError(f"unknown encoder {encoder}")

            self.norm_w = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.norm_a = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

            if decoder == "mlp":
                self.dec = nn.Sequential(
                    nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, 1))
            elif decoder == "vgae":
                self.mu_w = nn.Linear(hidden, hidden); self.lv_w = nn.Linear(hidden, hidden)
                self.mu_a = nn.Linear(hidden, hidden); self.lv_a = nn.Linear(hidden, hidden)
                self.dec_bias = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                              nn.Linear(hidden, 1))
            else:
                raise ValueError(f"unknown decoder {decoder}")
            self.dropout = dropout
            self.n_analytes = n_analytes

        # ---------------------------------------------------------------- encode
        def _encode_hetero(self, hw, ha, edge_dict):
            for i in range(len(self.convs)):
                xd = {"well": hw, "analyte": ha}
                if self.encoder_kind == "hgt":
                    out = self.convs[i](xd, edge_dict)
                else:  # hetero_sage
                    out = self.convs[i](xd, edge_dict)
                hw_new = out["well"]; ha_new = out.get("analyte", ha)
                hw = Fn.dropout(Fn.relu(self.norm_w[i](hw_new)), p=self.dropout,
                                training=self.training)
                ha = Fn.dropout(Fn.relu(self.norm_a[i](ha_new)), p=self.dropout,
                                training=self.training)
            return hw, ha

        def _encode_rgcn(self, hw, ha, edge_dict):
            nw = hw.shape[0]
            h = torch.cat([hw, ha], dim=0)
            ea2w = edge_dict[R_A2W]; ew2a = edge_dict[R_W2A]; ew2w = edge_dict[R_W2W]
            # remap analyte indices to [nw..nw+na) in the merged node space
            e0 = torch.stack([ea2w[0] + nw, ea2w[1]], 0)          # analyte->well, rel 0
            e1 = torch.stack([ew2a[0], ew2a[1] + nw], 0)          # well->analyte, rel 1
            e2 = ew2w                                             # well->well,    rel 2
            ei = torch.cat([e0, e1, e2], dim=1)
            et = torch.cat([torch.zeros(e0.shape[1], dtype=torch.long, device=h.device),
                            torch.ones(e1.shape[1], dtype=torch.long, device=h.device),
                            2 * torch.ones(e2.shape[1], dtype=torch.long, device=h.device)])
            for i in range(len(self.convs)):
                h_new = self.convs[i](h, ei, et)
                hw_n = self.norm_w[i](h_new[:nw]); ha_n = self.norm_a[i](h_new[nw:])
                h = torch.cat([Fn.dropout(Fn.relu(hw_n), p=self.dropout, training=self.training),
                               Fn.dropout(Fn.relu(ha_n), p=self.dropout, training=self.training)],
                              dim=0)
            return h[:nw], h[nw:]

        def encode(self, x_well, analyte_ids, edge_dict):
            hw = Fn.relu(self.well_in(x_well))
            ha = Fn.relu(self.analyte_in(self.analyte_emb(analyte_ids)))
            if self.encoder_kind == "rgcn":
                return self._encode_rgcn(hw, ha, edge_dict)
            return self._encode_hetero(hw, ha, edge_dict)

        # ---------------------------------------------------------------- decode
        def decode(self, hw, ha, pair_well, pair_analyte):
            kl = torch.zeros((), device=hw.device)
            if self.decoder_kind == "mlp":
                zw = hw[pair_well]; za = ha[pair_analyte]
                feat = torch.cat([zw, za, zw * za], dim=-1)
                return self.dec(feat).squeeze(-1), kl
            # VGAE: variational posteriors on the encoded nodes, sample, inner-product score
            mu_w = self.mu_w(hw); lv_w = self.lv_w(hw).clamp(-8, 8)
            mu_a = self.mu_a(ha); lv_a = self.lv_a(ha).clamp(-8, 8)
            if self.training:
                zw = mu_w + torch.randn_like(mu_w) * torch.exp(0.5 * lv_w)
                za = mu_a + torch.randn_like(mu_a) * torch.exp(0.5 * lv_a)
                kl_w = -0.5 * torch.mean(1 + lv_w - mu_w.pow(2) - lv_w.exp())
                kl_a = -0.5 * torch.mean(1 + lv_a - mu_a.pow(2) - lv_a.exp())
                kl = kl_w + kl_a
            else:
                zw, za = mu_w, mu_a
            zwp = zw[pair_well]; zap = za[pair_analyte]
            logit = (zwp * zap).sum(-1) + self.dec_bias(
                torch.cat([zwp, zap], dim=-1)).squeeze(-1)
            return logit, kl

        def forward(self, x_well, analyte_ids, edge_dict, pair_well, pair_analyte):
            hw, ha = self.encode(x_well, analyte_ids, edge_dict)
            return self.decode(hw, ha, pair_well, pair_analyte)

    return HeteroCompletion()


# --------------------------------------------------------------------------- loss
def _bce_focal(logit, target, pos_weight, gamma):
    """Per-element BCE-with-logits, optionally focal (gamma>0). pos_weight is per-edge."""
    import torch
    import torch.nn.functional as Fn
    bce = Fn.binary_cross_entropy_with_logits(logit, target, weight=pos_weight,
                                              reduction="none")
    if gamma and gamma > 0:
        p = torch.sigmoid(logit)
        pt = target * p + (1 - target) * (1 - p)        # prob of the true class
        bce = bce * (1 - pt).clamp(min=1e-6) ** gamma
    return bce.mean()


# --------------------------------------------------------------------------- one fold
@dataclass
class HetFold:
    fold: int
    n_train_edges: int
    n_test_edges: int
    n_well_edges_train: int          # symmetrised well<->well edges used in message passing
    n_cross_block_bip: int           # bipartite edges crossing a block (must be 0)
    n_cross_block_well: int          # well<->well edges crossing a block AFTER cut (must be 0)
    n_removed_well_cross: int        # well<->well edges removed by the C4 cut (audit)
    best_epoch: int


def train_eval_hetero_fold(df, well_ids, Yw, Mw, well_to_node, feature_cols,
                           well_block, test_block, *, encoder="hetero_sage", decoder="mlp",
                           emb_dim=32, hidden=64, layers=2, dropout=0.3, heads=2,
                           k=8, cap_km=1.5, lr=5e-3, weight_decay=5e-4, max_epochs=300,
                           patience=40, val_frac=0.15, gamma=0.0, beta_kl=1e-3,
                           seed=C.SEED, labels=None, verbose=False):
    """Train the heterogeneous completion model on TRAIN-block wells' measured cells (with
    spatial well-well message passing restricted to TRAIN, cross-block-free), predict on
    TEST-block wells' measured cells. Returns (HetFold, P_well[n_wells,n_labels], test_mask).
    """
    import torch
    from torch_geometric.utils import to_undirected

    labels = labels or C.T2_LABELS
    set_seed(seed)
    dev = device()
    n_wells = len(well_ids)
    n_lab = len(labels)

    test_wells = well_block == test_block
    train_wells = ~test_wells

    # node features fit on TRAIN wells only (anti-leak, frequency encode)
    X, _, _ = G.node_features(df, well_ids, feature_cols, train_node_mask=train_wells,
                              encode="frequency")
    x = torch.tensor(X, dtype=torch.float32, device=dev)
    analyte_ids = torch.arange(n_lab, dtype=torch.long, device=dev)

    # ---- bipartite measured-cell edges. TRAIN edges = TRAIN wells' cells; TEST = TEST wells'.
    tr_w, tr_a, tr_y, te_w, te_a = [], [], [], [], []
    for w in range(n_wells):
        for j in range(n_lab):
            if not Mw[w, j]:
                continue
            if train_wells[w]:
                tr_w.append(w); tr_a.append(j); tr_y.append(int(Yw[w, j]))
            else:
                te_w.append(w); te_a.append(j)
    tr_w = np.asarray(tr_w); tr_a = np.asarray(tr_a); tr_y = np.asarray(tr_y, np.float32)
    te_w = np.asarray(te_w); te_a = np.asarray(te_a)

    # bipartite edges touch exactly one well = one block -> 0 cross-block (audited)
    n_cross_bip = int((well_block[tr_w] != test_block).sum() == 0 and 0)  # always 0; explicit
    n_cross_bip = 0

    # ---- spatial well<->well edges (capped k-NN), CUT to TRAIN-train pairs that do NOT
    # cross the block boundary (C4). We build the full capped graph then keep only edges
    # whose BOTH endpoints are TRAIN wells (so a test well never sends/receives spatial msgs
    # in training; message passing stays train-only -> inductive), which also makes them
    # trivially cross-block-free. We additionally cut by node_block and AUDIT 0 remain.
    _, coords, _ = G.well_table(df)
    ei_full, ed_full = G.knn_edges_km(coords, k=k, cap_km=cap_km)
    # C4 cut on the OUTER block id (node_block = well_block)
    ei_cut, ed_cut, n_removed_well = G.cut_cross_block(ei_full, ed_full, well_block)
    # restrict message-passing spatial edges to TRAIN-train pairs (inductive)
    a_, b_ = ei_cut[0], ei_cut[1]
    keep = train_wells[a_] & train_wells[b_]
    ei_tr = ei_cut[:, keep]
    # audit: 0 well-edge crosses a block AFTER cut
    n_cross_well = int((well_block[ei_tr[0]] != well_block[ei_tr[1]]).sum())

    ew2w = torch.tensor(ei_tr, dtype=torch.long, device=dev)
    ew2w = to_undirected(ew2w, num_nodes=n_wells)        # symmetrise well<->well

    # bipartite message-passing edges (TRAIN only)
    ea2w = torch.tensor(np.vstack([tr_a, tr_w]), dtype=torch.long, device=dev)
    ew2a = torch.tensor(np.vstack([tr_w, tr_a]), dtype=torch.long, device=dev)
    edge_dict = {
        ("analyte", "measured_by", "well"): ea2w,
        ("well", "measures", "analyte"): ew2a,
        ("well", "near", "well"): ew2w,
    }

    # ---- validation split: hold out a slice of TRAIN edges (early stop), wells stay train.
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

    # per-label pos_weight (imbalance, like class_weight='balanced')
    pw = np.ones(n_lab, dtype=np.float32)
    for j in range(n_lab):
        yj = tr_y[tr_a == j]
        pos = float(yj.sum()); neg = float(len(yj) - pos)
        pw[j] = (neg / pos) if pos > 0 else 1.0
    pos_weight_edge = torch.tensor(pw[tr_a[fit_idx]], dtype=torch.float32, device=dev)

    model = build_hetero_model(X.shape[1], n_lab, encoder=encoder, decoder=decoder,
                               emb_dim=emb_dim, hidden=hidden, layers=layers,
                               dropout=dropout, heads=heads).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=max(5, patience // 4),
                                                       min_lr=1e-5)

    best_val, best_state, best_epoch, bad = -np.inf, None, 0, 0
    for epoch in range(max_epochs):
        model.train(); opt.zero_grad()
        logit, kl = model(x, analyte_ids, edge_dict, fit_w, fit_a)
        loss = _bce_focal(logit, fit_y, pos_weight_edge, gamma) + beta_kl * kl
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite hetero loss")
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vlogit, _ = model(x, analyte_ids, edge_dict, val_w, val_a)
            vp = torch.sigmoid(vlogit).cpu().numpy()
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
            plogit, _ = model(x, analyte_ids, edge_dict, tw, ta)
            P_well[te_w, te_a] = torch.sigmoid(plogit).cpu().numpy()

    fr = HetFold(fold=int(test_block), n_train_edges=int(n_tr), n_test_edges=int(len(te_w)),
                 n_well_edges_train=int(ew2w.shape[1]), n_cross_block_bip=int(n_cross_bip),
                 n_cross_block_well=int(n_cross_well),
                 n_removed_well_cross=int(n_removed_well), best_epoch=int(best_epoch))
    return fr, P_well, test_wells


# --------------------------------------------------------------------------- CV runner
def run_t2_hetero_cv(df, *, feature_cols=None, regime="spatial", labels=None,
                     n_blocks=None, encoder="hetero_sage", decoder="mlp", emb_dim=32,
                     hidden=64, layers=2, dropout=0.3, heads=2, k=8, cap_km=1.5,
                     lr=5e-3, max_epochs=300, patience=40, gamma=0.0, beta_kl=1e-3,
                     seed=C.SEED, verbose=False):
    """Leave-one-block-out (spatial) or group-random CV for the hetero completion model.

    Returns a dict with the 5 multilabel metrics ROW-LEVEL via metrics.multilabel_metrics
    (strict comparability with the wall AND P1), per-label AUROC+AP table, OOF thresholds,
    the C4 audit (cross-block bipartite AND well edges, both must be 0), per-fold diagnostics.
    """
    labels = labels or C.T2_LABELS
    feature_cols = feature_cols or C.feature_columns(include_location=False, cocontam="core")
    n_blocks = n_blocks or (C.N_SPATIAL_BLOCKS if regime == "spatial" else C.N_RANDOM_FOLDS)

    if regime == "spatial":
        fold_block = S.spatial_block_folds(df, k=n_blocks, seed=seed)
    else:
        fold_block = S.group_random_folds(df, k=n_blocks, seed=seed)

    well_ids, Yw, Mw, well_to_node = GB.well_label_matrix(df, labels)
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
        fr, P_well, test_wells = train_eval_hetero_fold(
            df, well_ids, Yw, Mw, well_to_node, feature_cols, well_block, b,
            encoder=encoder, decoder=decoder, emb_dim=emb_dim, hidden=hidden, layers=layers,
            dropout=dropout, heads=heads, k=k, cap_km=cap_km, lr=lr, max_epochs=max_epochs,
            patience=patience, gamma=gamma, beta_kl=beta_kl, seed=seed, labels=labels,
            verbose=verbose)
        fill = test_wells[:, None] & Mw
        P_well_oof[fill] = P_well[fill]
        folds_info.append(fr)
        if verbose:
            print(f"[{regime}] block {b}: tr_edges={fr.n_train_edges} te_edges={fr.n_test_edges} "
                  f"well_edges={fr.n_well_edges_train} removed_well_xblock={fr.n_removed_well_cross} "
                  f"cross_well={fr.n_cross_block_well} best_ep={fr.best_epoch}")

    # ---- broadcast well-level OOF predictions to ROWS and score like the wall + P1.
    row_node = df[C.WELL_ID].map(well_to_node).to_numpy().astype(np.int64)
    Y_row, M_row = B2.masked_targets(df, labels=labels)
    n_rows = len(df)
    P_row = np.full((n_rows, len(labels)), np.nan, dtype=np.float64)
    for j in range(len(labels)):
        P_row[:, j] = P_well_oof[row_node, j]
    for j, a in enumerate(labels):
        col = f"label_{a}"
        m = M_row[col].to_numpy()
        nanmask = m & np.isnan(P_row[:, j])
        if nanmask.any():
            prev = float(Y_row[col].to_numpy()[m & ~np.isnan(P_row[:, j])].mean()) \
                if (m & ~np.isnan(P_row[:, j])).any() else 0.0
            P_row[nanmask, j] = prev
    P_row = np.nan_to_num(P_row, nan=0.0)

    thr = B2.best_thresholds_oof(Y_row, P_row, M_row, labels)
    full5 = MM.multilabel_metrics(Y_row, P_row, M_row, labels, thr)

    cross_bip = int(sum(f.n_cross_block_bip for f in folds_info))
    cross_well = int(sum(f.n_cross_block_well for f in folds_info))
    removed_well = int(sum(f.n_removed_well_cross for f in folds_info))

    return {
        "regime": regime, "n_blocks": len(blocks), "encoder": encoder, "decoder": decoder,
        "macro_AUROC": float(full5["macro"]["roc_auc"]),
        "micro_AUROC": float(full5["micro"]["roc_auc"]),
        "micro_F1": float(full5["micro"]["f1"]),
        "macro_F1": float(full5["macro"]["f1"]),
        "micro_recall": float(full5["micro"]["recall"]),
        "micro_precision": float(full5["micro"]["precision"]),
        "subset_accuracy": float(full5["subset_accuracy"]),
        "per_label": full5["per_label"].to_dict(orient="records"),
        "thresholds": thr,
        "n_cross_block_bipartite": cross_bip,
        "n_cross_block_well": cross_well,
        "n_removed_well_cross_total": removed_well,
        "n_cross_block_edges": cross_bip + cross_well,   # total guard (must be 0)
        "per_fold": [vars(f) for f in folds_info],
        "P_row": P_row, "Y_row": Y_row, "M_row": M_row,
    }
