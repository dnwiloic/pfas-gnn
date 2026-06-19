"""Shared classification metrics.

The project reports the SAME five headline metrics on BOTH tasks — **F1-score,
accuracy, recall, precision, AUC-ROC** — plus a few decision-oriented extras. Defining
them once here keeps T1 (binary) and T2 (multilabel) strictly comparable.

- `binary_metrics`     : the 5 (+ pr_auc, balanced_accuracy, brier) for T1.
- `multilabel_metrics` : the 5 in micro and macro form, per-label, + subset accuracy
  (exact match), all computed ONLY on measured cells (per-label measurement mask).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# The five metrics the project requires on every task.
REQUIRED = ("roc_auc", "f1", "accuracy", "recall", "precision")


def binary_metrics(y_true, proba, threshold: float = 0.5, *, extras: bool = True) -> dict:
    """The 5 required metrics for a binary target at `threshold` (AUC-ROC is
    threshold-free). `extras` adds pr_auc / balanced_accuracy / brier."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    two = len(np.unique(y_true)) > 1
    out = {
        "roc_auc": float(roc_auc_score(y_true, proba)) if two else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
    }
    if extras:
        out.update(
            pr_auc=float(average_precision_score(y_true, proba)) if two else float("nan"),
            balanced_accuracy=float(balanced_accuracy_score(y_true, pred)),
            brier=float(brier_score_loss(y_true, proba)),
            threshold_used=float(threshold),
        )
    return out


def multilabel_metrics(Y_df: pd.DataFrame, P, M_df: pd.DataFrame, labels, thresholds) -> dict:
    """Micro/macro of the 5 metrics + per-label table + subset accuracy (exact match).

    All quantities respect the per-label measurement mask: a label contributes only on
    rows where it was measured. `thresholds` is a per-label dict. Micro = pooled measured
    cells; macro = unweighted mean over labels; subset accuracy = full-vector exact match
    on rows where every label is measured.
    """
    cols = [f"label_{a}" for a in labels]
    thr = np.array([float(thresholds[a]) for a in labels])
    P = np.asarray(P, dtype=float)
    Yhat = (P >= thr[None, :]).astype(int)
    Ymat = Y_df[cols].to_numpy().astype(int)
    Mmat = M_df[cols].to_numpy().astype(bool)

    per = []
    for j, a in enumerate(labels):
        m = Mmat[:, j]
        yt, pp, ph = Ymat[m, j], P[m, j], Yhat[m, j]
        two = len(np.unique(yt)) > 1
        per.append({
            "label": a, "n_measured": int(m.sum()),
            "prevalence": float(yt.mean()) if len(yt) else float("nan"),
            "roc_auc": float(roc_auc_score(yt, pp)) if two else float("nan"),
            "pr_auc": float(average_precision_score(yt, pp)) if two else float("nan"),
            "f1": float(f1_score(yt, ph, zero_division=0)),
            "precision": float(precision_score(yt, ph, zero_division=0)),
            "recall": float(recall_score(yt, ph, zero_division=0)),
            "accuracy": float(accuracy_score(yt, ph)) if len(yt) else float("nan"),
        })
    pl = pd.DataFrame(per)

    flat_t, flat_b, flat_p = Ymat[Mmat], Yhat[Mmat], P[Mmat]
    two_micro = len(np.unique(flat_t)) > 1
    micro = {
        "roc_auc": float(roc_auc_score(flat_t, flat_p)) if two_micro else float("nan"),
        "f1": float(f1_score(flat_t, flat_b, zero_division=0)),
        "accuracy": float(accuracy_score(flat_t, flat_b)),
        "recall": float(recall_score(flat_t, flat_b, zero_division=0)),
        "precision": float(precision_score(flat_t, flat_b, zero_division=0)),
    }
    macro = {k: float(np.nanmean(pl[k])) for k in REQUIRED}

    full = Mmat.all(axis=1)
    subset_acc = (float((Yhat[full] == Ymat[full]).all(axis=1).mean())
                  if full.any() else float("nan"))
    return {"micro": micro, "macro": macro, "subset_accuracy": subset_acc,
            "n_full_panel_rows": int(full.sum()), "per_label": pl}
