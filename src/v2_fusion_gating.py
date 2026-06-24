"""V2 — fusion variant (c): GATING / ATTENTION head over [tabular | HGT-embedding].

Context
-------
The V0 fusion (`hgt_fusion_stacking_t1.fusion_oof_proba`) compresses the 64-D HGT
embedding with PCA-to-95%-variance, which on THIS dataset collapses to ~1 component and
throws away the graph signal. V2 keeps the SAME leak-free 64-D OOF embedding produced by
`build_oof_backbone` and tests three NON-destructive fusions. Variants (a) full 64-D and
(b) PCA-at-fixed-k are pure-XGBoost runners owned by the tabular-ml-engineer (NEXT step).

THIS module owns variant (c): a small neural GATING head that LEARNS how much to trust the
graph block vs the tabular block per well, instead of concatenating-and-hoping. The gate is

    g = sigmoid(MLP_gate([x_tab, x_emb]))          # scalar in (0,1) per well
    z = concat( g * proj_emb(x_emb), (1 - g) * x_tab )   # fused representation
    p = sigmoid( head(z) )                          # contamination probability

so the head can down-weight the embedding when it is uninformative (recovering the
tabular-only solution at g->0) and up-weight it where the relational context helps.

Anti-leak law (the heart of the project)
-----------------------------------------
The gating head is trained NESTED LEAVE-ONE-BLOCK-OUT over the SAME 8 KMeans spatial blocks
as the socle (`S.spatial_block_folds`). For held-out block b the MLP — and the tabular
standardiser, and any normalisation — are FIT ONLY on the OOF rows of the OTHER 7 blocks;
block b is touched solely at prediction time. This mirrors exactly how PCA+XGB are fit in
`fusion_oof_proba`. An EARLY-STOP / threshold VAL set is carved from the TRAIN blocks by
holding out one whole spatial block (block-aware split), never from the held-out test block.

The input embedding is itself already OOF and leak-free (each well's HGT embedding came from
an HGT that never saw its block — C-SPAT.4 inductive). So no test-block information enters
any fit step at any level.

CLAUDE.md §3.8 — training curves (mandatory)
--------------------------------------------
Each gating-MLP fit logs PER EPOCH: train_loss and val_auc (val = the held-out TRAIN block,
NEVER the test block). Histories are returned per fold and, via `plot_gating_curves`, a PNG
(`figures/gating_training_curves.png`) is written with every fold's val-AUC and train-loss vs
epoch plus the early-stop marker, so under-trained folds (flat val-AUC, stop at epoch ~9) are
visible before any conclusion is drawn.

CPU smoke-testable: torch is imported lazily; tiny config fits in seconds.

Public API (for the tabular-ml-engineer)
-----------------------------------------
    get_fusion_inputs(df=None, oof=None, *, smoke, ...) -> dict
        builds (or reuses) the V0 OOF backbone and returns the aligned arrays
        {well_ids, node_block, y_well, hgt_emb[n,H], tabular[n,d], row_to_node, valid_emb}.
    train_gating_oof(inputs, *, smoke, ...) -> GatingOOFResult
        nested-LOBO gating; returns proba_well[n] (OOF), fused_repr[n, d+H] (OOF),
        per-fold histories, the early-stop diagnostic and the OOF F1 threshold.
    plot_gating_curves(result, exp_dir) -> writes figures/gating_training_curves.png
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config as C
from . import hgt_fusion_stacking_t1 as FB
from .hybrid import _optimal_threshold_f1, _ece

SEED = C.SEED

# ----------------------------------------------------------------- smoke / full params
SMOKE_N_WELLS = FB.SMOKE_N_WELLS
SMOKE_BLOCKS = FB.SMOKE_BLOCKS
SMOKE_GATE_EPOCHS = 40
SMOKE_GATE_PATIENCE = 12

FULL_BLOCKS = C.N_SPATIAL_BLOCKS
FULL_GATE_EPOCHS = 300
FULL_GATE_PATIENCE = 40


# ============================================================= (i) fusion inputs accessor
def get_fusion_inputs(df=None, oof=None, *, smoke=False, seed=SEED,
                      encoder="hgt", hidden=64, layers=2, dropout=0.3, heads=4,
                      k_spatial=8, cap_km_spatial=1.5, k_subbasin=8, cap_km_subbasin=2.0,
                      max_epochs=None, patience=None, lr=5e-3, weight_decay=5e-4,
                      inductive=True, n_blocks=None, verbose=False):
    """Build (or reuse) the V0 OOF backbone and return the aligned per-well arrays.

    Pass `oof` (an `OOFArrays` already built) to skip the expensive backbone rebuild —
    this is the cheap path the tabular-ml-engineer should use so variants (a)/(b)/(c)
    all consume the SAME embedding. Otherwise the backbone is built here.

    `encoder` is kept parameterisable but defaults to "hgt": V1 showed hetero_sage is
    strictly worse, so it is NOT the default. (The encoder choice flows into
    build_oof_backbone via gnn_hetero_t1's `name`; here we always use the HGT trainer the
    socle wires in, so `encoder` is recorded for provenance.)

    Returns the dict documented in `FB.oof_embeddings_and_tabular` plus `meta`.
    """
    if oof is None:
        from . import data as D
        if smoke:
            n_blocks = n_blocks or SMOKE_BLOCKS
            max_epochs = max_epochs or FB.SMOKE_EPOCHS
            patience = patience or FB.SMOKE_PATIENCE
        else:
            n_blocks = n_blocks or FULL_BLOCKS
            max_epochs = max_epochs or FB.FULL_EPOCHS
            patience = patience or FB.FULL_PATIENCE
        if df is None:
            df = D.load(smoke=smoke, smoke_n=SMOKE_N_WELLS if smoke else None)
        if smoke and df[C.WELL_ID].nunique() > SMOKE_N_WELLS:
            rng = np.random.RandomState(seed)
            keep = set(rng.choice(df[C.WELL_ID].unique(), size=SMOKE_N_WELLS, replace=False))
            df = df[df[C.WELL_ID].isin(keep)].reset_index(drop=True)
        feature_cols = C.feature_columns(include_location=False, cocontam="core")
        oof = FB.build_oof_backbone(
            df, feature_cols=feature_cols, n_blocks=n_blocks, regime="spatial",
            hidden=hidden, layers=layers, dropout=dropout, heads=heads,
            k_spatial=k_spatial, cap_km_spatial=cap_km_spatial, k_subbasin=k_subbasin,
            cap_km_subbasin=cap_km_subbasin, max_epochs=max_epochs, patience=patience,
            lr=lr, weight_decay=weight_decay, inductive=inductive, smoke=smoke,
            seed=seed, verbose=verbose)

    inputs = FB.oof_embeddings_and_tabular(oof)
    inputs["meta"] = {"encoder": encoder, "regime": "spatial",
                      "n_wells": len(inputs["well_ids"]),
                      "hidden": inputs["hidden"],
                      "n_tabular_features": inputs["n_tabular_features"]}
    inputs["_oof"] = oof   # keep a handle so callers can reuse it for variants (a)/(b)
    return inputs


# ============================================================= gating network
def _build_gating_net(d_tab, d_emb, *, proj_dim, gate_hidden, dropout, seed):
    """Two-block gating MLP. Returns a torch.nn.Module exposing forward -> (logit, g, z)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class GatingHead(nn.Module):
        def __init__(self):
            super().__init__()
            # project the 64-D embedding to a comparable width (keeps it non-destructive:
            # full 64-D in, learned projection out — no PCA collapse)
            self.proj = nn.Sequential(
                nn.Linear(d_emb, proj_dim), nn.ReLU(), nn.Dropout(dropout))
            # gate sees BOTH raw blocks and emits one scalar weight per well
            self.gate = nn.Sequential(
                nn.Linear(d_tab + d_emb, gate_hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(gate_hidden, 1))
            # classification head over the fused representation [g*proj_emb | (1-g)*x_tab]
            self.head = nn.Sequential(
                nn.Linear(proj_dim + d_tab, gate_hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(gate_hidden, 1))

        def forward(self, x_tab, x_emb):
            pe = self.proj(x_emb)
            g = torch.sigmoid(self.gate(torch.cat([x_tab, x_emb], dim=1)))   # [N,1]
            z = torch.cat([g * pe, (1.0 - g) * x_tab], dim=1)                # fused repr
            logit = self.head(z).squeeze(-1)
            return logit, g.squeeze(-1), z

    return GatingHead()


def _block_aware_val_split(node_block_tr, block_ids_tr, *, seed):
    """Hold out ONE whole TRAIN block as the gating VAL set (early-stop + threshold).

    Picks the block whose size is closest to ~18% of the train rows so the val set is a
    real spatial hold-out, not a random slice. Returns boolean (fit_mask, val_mask) over
    the TRAIN rows. NEVER touches the outer test block.
    """
    blocks = sorted(set(block_ids_tr.tolist()))
    if len(blocks) < 2:
        rng = np.random.RandomState(seed)
        idx = np.arange(len(node_block_tr))
        vi = rng.choice(idx, size=max(1, int(0.18 * len(idx))), replace=False)
        val = np.zeros(len(idx), dtype=bool); val[vi] = True
        return ~val, val
    sizes = {b: int((block_ids_tr == b).sum()) for b in blocks}
    target = 0.18 * len(block_ids_tr)
    vb = min(blocks, key=lambda b: abs(sizes[b] - target))
    val = block_ids_tr == vb
    return ~val, val


@dataclass
class GatingOOFResult:
    proba_well: np.ndarray                 # [n_wells] OOF gating probability (NaN if unseen)
    fused_repr: np.ndarray                 # [n_wells, proj_dim + d_tab] OOF fused repr
    gate_value: np.ndarray                 # [n_wells] OOF gate g (graph weight) per well
    y_well: np.ndarray
    node_block: np.ndarray
    well_ids: np.ndarray
    row_to_node: np.ndarray
    oof_threshold: float                   # F1 threshold from OOF wells (C-THR)
    fold_histories: list = field(default_factory=list)   # §3.8 per-fold curves
    diag: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def train_gating_oof(inputs, *, smoke=False, seed=SEED, proj_dim=16, gate_hidden=32,
                     dropout=0.3, lr=5e-3, weight_decay=1e-4, max_epochs=None,
                     patience=None, verbose=False):
    """Nested-LOBO training of the gating head on the OOF embedding + tabular blocks.

    For each held-out spatial block b:
      * fit StandardScaler on the TABULAR rows of the OTHER blocks only (train-only),
        transform train + held-out;
      * standardise the EMBEDDING with stats from the OTHER blocks only too;
      * carve a block-aware VAL set from the train blocks (early-stop + threshold);
      * train the gating MLP, logging train_loss + val_auc per epoch (§3.8);
      * predict block b -> OOF proba, fused repr, gate value.

    Returns a GatingOOFResult. proba_well / fused_repr / gate_value are aligned to
    inputs['well_ids'] row order (well i <-> row i).
    """
    import torch
    import torch.nn.functional as Fn
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    if smoke:
        max_epochs = max_epochs or SMOKE_GATE_EPOCHS
        patience = patience or SMOKE_GATE_PATIENCE
    else:
        max_epochs = max_epochs or FULL_GATE_EPOCHS
        patience = patience or FULL_GATE_PATIENCE

    well_ids = inputs["well_ids"]
    node_block = inputs["node_block"]
    y_well = inputs["y_well"].astype(np.float32)
    emb = inputs["hgt_emb"].astype(np.float32)
    tab = inputs["tabular"].astype(np.float32)
    valid = inputs["valid_emb"]
    n = len(well_ids)
    d_tab = tab.shape[1]
    d_emb = emb.shape[1]

    proba = np.full(n, np.nan, dtype=np.float64)
    gate_v = np.full(n, np.nan, dtype=np.float64)
    fused = np.full((n, proj_dim + d_tab), np.nan, dtype=np.float32)
    histories = []
    n_epochs_per_fold = []

    blocks = sorted(set(node_block.tolist()))
    dev = torch.device("cpu")

    for b in blocks:
        tr = (node_block != b) & valid
        te = (node_block == b) & valid
        if tr.sum() < 10 or te.sum() < 1 or len(np.unique(y_well[tr])) < 2:
            continue

        # block-aware VAL carved from TRAIN blocks (never the test block b)
        tr_idx = np.where(tr)[0]
        fit_rel, val_rel = _block_aware_val_split(node_block[tr_idx], node_block[tr_idx],
                                                  seed=seed)
        fit_idx = tr_idx[fit_rel]
        val_idx = tr_idx[val_rel]
        if len(val_idx) == 0 or len(np.unique(y_well[val_idx])) < 2:
            # fall back to a random 18% slice if the chosen val block is single-class
            rng = np.random.RandomState(seed)
            perm = rng.permutation(tr_idx)
            cut = max(1, int(0.18 * len(perm)))
            val_idx, fit_idx = perm[:cut], perm[cut:]

        # standardisers fit on FIT rows ONLY (train-only, anti-leak)
        sc_tab = StandardScaler().fit(tab[fit_idx])
        sc_emb = StandardScaler().fit(emb[fit_idx])

        def _xt(idx):
            return (torch.tensor(sc_tab.transform(tab[idx]), dtype=torch.float32, device=dev),
                    torch.tensor(sc_emb.transform(emb[idx]), dtype=torch.float32, device=dev))

        xt_fit, xe_fit = _xt(fit_idx)
        xt_val, xe_val = _xt(val_idx)
        y_fit = torch.tensor(y_well[fit_idx], dtype=torch.float32, device=dev)
        y_val_np = y_well[val_idx].astype(int)

        # class-imbalance weight from FIT prevalence (T1a quasi-balanced -> ~1)
        prev = float(y_well[fit_idx].mean())
        pos_w = torch.tensor([(1 - prev) / max(prev, 1e-6)], dtype=torch.float32, device=dev)

        net = _build_gating_net(d_tab, d_emb, proj_dim=proj_dim, gate_hidden=gate_hidden,
                                dropout=dropout, seed=seed).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        hist_ep, hist_loss, hist_val_auc = [], [], []
        best_val, best_state, best_epoch, bad = -1.0, None, 0, 0
        for epoch in range(1, max_epochs + 1):
            net.train()
            opt.zero_grad()
            logit, _, _ = net(xt_fit, xe_fit)
            loss = Fn.binary_cross_entropy_with_logits(logit, y_fit, pos_weight=pos_w)
            loss.backward()
            opt.step()

            net.eval()
            with torch.no_grad():
                vlogit, _, _ = net(xt_val, xe_val)
                vp = torch.sigmoid(vlogit).cpu().numpy()
            vauc = (float(roc_auc_score(y_val_np, vp))
                    if len(np.unique(y_val_np)) > 1 else float("nan"))
            hist_ep.append(int(epoch))
            hist_loss.append(float(loss.detach()))
            hist_val_auc.append(vauc)

            if np.isfinite(vauc) and vauc > best_val + 1e-5:
                best_val, best_epoch, bad = vauc, epoch, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
            if verbose and epoch % 10 == 0:
                print(f"[gate] block {b} ep{epoch} loss={float(loss.detach()):.4f} "
                      f"val_auc={vauc:.4f}")

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            xt_te, xe_te = _xt(np.where(te)[0])
            tlogit, tg, tz = net(xt_te, xe_te)
            proba[te] = torch.sigmoid(tlogit).cpu().numpy()
            gate_v[te] = tg.cpu().numpy()
            fused[te] = tz.cpu().numpy().astype(np.float32)

        n_epochs_per_fold.append(len(hist_ep))
        histories.append({
            "fold": int(b), "best_epoch": int(best_epoch),
            "n_epochs_ran": len(hist_ep), "max_epochs": int(max_epochs),
            "early_stopped": bool(len(hist_ep) < max_epochs),
            "history_epochs": hist_ep,
            "history_train_loss": hist_loss,
            "history_val_auc": hist_val_auc,
            "n_fit": int(len(fit_idx)), "n_val": int(len(val_idx)),
            "best_val_auc": float(best_val),
        })

    # OOF F1 threshold from the gating OOF wells (C-THR: never from test rows directly)
    seen = ~np.isnan(proba)
    thr = (_optimal_threshold_f1(y_well[seen].astype(int), proba[seen])
           if seen.any() else 0.5)

    # under-training diagnostic (§3.8): flag folds that stop very early with a flat val-AUC
    undertrained = []
    for h in histories:
        va = [v for v in h["history_val_auc"] if np.isfinite(v)]
        flat = (len(va) >= 3 and (max(va) - min(va)) < 0.02)
        if h["early_stopped"] and h["best_epoch"] <= max(5, int(0.15 * h["max_epochs"])) and flat:
            undertrained.append(h["fold"])

    diag = {
        "n_epochs_per_fold": n_epochs_per_fold,
        "mean_epochs": float(np.mean(n_epochs_per_fold)) if n_epochs_per_fold else 0.0,
        "undertrained_folds": undertrained,
        "mean_gate_value": float(np.nanmean(gate_v)) if seen.any() else float("nan"),
        "oof_threshold": float(thr),
    }

    return GatingOOFResult(
        proba_well=proba, fused_repr=fused, gate_value=gate_v, y_well=y_well.astype(int),
        node_block=node_block, well_ids=well_ids, row_to_node=inputs["row_to_node"],
        oof_threshold=float(thr), fold_histories=histories, diag=diag,
        meta={"proj_dim": proj_dim, "gate_hidden": gate_hidden, "d_tab": d_tab,
              "d_emb": d_emb, "fused_dim": proj_dim + d_tab,
              "max_epochs": int(max_epochs), "patience": int(patience)})


# ============================================================= §3.8 curves
def plot_gating_curves(result, exp_dir):
    """Write figures/gating_training_curves.png: per-fold val-AUC and train-loss vs epoch
    with the early-stop marker. Read this BEFORE concluding (§3.8)."""
    fig_dir = Path(exp_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable ({e}); skipping gating curves")
        return None
    fds = result.fold_histories
    if not fds:
        print("[plot] no fold histories; nothing to plot")
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for d in fds:
        ep = d.get("history_epochs", [])
        if not ep:
            continue
        ax1.plot(ep, d.get("history_val_auc", []), lw=1.1, alpha=0.8,
                 label=f"fold {d.get('fold')} (es@{d.get('best_epoch')})")
        be = d.get("best_epoch")
        if be is not None:
            ax1.axvline(be, color="grey", lw=0.5, alpha=0.3)
        ax2.plot(ep, d.get("history_train_loss", []), lw=1.1, alpha=0.8,
                 label=f"fold {d.get('fold')}")
    ax1.set(xlabel="epoch", ylabel="val AUC (held-out TRAIN block)",
            title="V2 gating — val AUC vs epoch (line at early-stop)")
    ax2.set(xlabel="epoch", ylabel="train loss (BCE)", title="V2 gating — train loss")
    ax1.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    out = fig_dir / "gating_training_curves.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    # also drop the raw histories so the smoke test can assert non-empty curves
    (fig_dir.parent / "gating_history.json").write_text(
        json.dumps({"fold_histories": fds, "diag": result.diag},
                   indent=2, default=FB._json_default))
    print(f"[plot] wrote {out} and gating_history.json")
    return out


# ============================================================= mini example / smoke entry
def smoke(df=None, *, seed=SEED, exp_dir=None, verbose=True):
    """End-to-end CPU smoke of the V2 gating head (< ~3 min). Builds a tiny OOF backbone,
    trains the gating head nested-LOBO, writes the §3.8 curve PNG + history, and asserts
    the §3.8 invariants (non-empty per-epoch curves, figure written).

    Returns (result, inputs) so a caller / the tabular-ml-engineer can inspect the API.
    """
    inputs = get_fusion_inputs(df=df, smoke=True, seed=seed, verbose=verbose)
    result = train_gating_oof(inputs, smoke=True, seed=seed, verbose=verbose)

    exp_dir = Path(exp_dir) if exp_dir else (C.EXPERIMENTS_DIR / "v2_fusion_gating_smoke")
    exp_dir.mkdir(parents=True, exist_ok=True)
    plot_gating_curves(result, exp_dir)

    # --- §3.8 / smoke invariants ---
    assert result.fold_histories, "no fold histories produced"
    for h in result.fold_histories:
        ne = h["n_epochs_ran"]
        assert ne > 0, f"fold {h['fold']} ran 0 epochs"
        assert len(h["history_train_loss"]) == ne, "train-loss history wrong length"
        assert len(h["history_val_auc"]) == ne, "val-auc history wrong length"
    assert (exp_dir / "figures" / "gating_training_curves.png").exists(), "curve PNG missing"

    # WELL-level OOF AUC as the smoke signal (row-level metrics belong to the downstream
    # runner; here we only confirm the gating head learns a finite, leak-free OOF signal)
    seen = ~np.isnan(result.proba_well)
    from sklearn.metrics import roc_auc_score
    yw = result.y_well[seen]
    well_auc = (float(roc_auc_score(yw, result.proba_well[seen]))
                if len(np.unique(yw)) > 1 else float("nan"))

    if verbose:
        print(f"[smoke] wells scored OOF: {int(seen.sum())}/{len(seen)}  "
              f"well-OOF AUC={well_auc:.4f}  mean_gate={result.diag['mean_gate_value']:.3f}")
        print(f"[smoke] epochs/fold={result.diag['n_epochs_per_fold']}  "
              f"undertrained_folds={result.diag['undertrained_folds']}")
        print(f"[smoke] fused_repr shape={result.fused_repr.shape}  "
              f"threshold={result.oof_threshold:.3f}")
    return result, inputs


if __name__ == "__main__":
    smoke()
