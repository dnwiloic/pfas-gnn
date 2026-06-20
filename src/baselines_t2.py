"""T2 multilabel BASELINE WALL (non-graph) — the strong baseline the GNNs must beat.

Task T2 (CLAUDE.md, T2_TARGETS.md): predict, in strict predictive mode (no PFAS
measurement as a feature), which individual PFAS exceed their per-compound threshold.
Targets come ONLY from src.targets.build_T2 (hybrid EPA-MCL/analytical + detection
guard C1). Splits, features, blocklist come ONLY from the frozen socle.

What this module provides
-------------------------
1. A per-label MEASUREMENT MASK: a label is built/scored ONLY on rows where the
   analyte was actually measured (``*_ngL`` not NaN). The measurement matrix is
   lacunar and INFORMATIVE (driven by lab/program, see REPORT): PFBA/PFPeA/PFPeS are
   only measured on ~55% of rows (the "full panel"), the rest is a reduced panel.
   Training/scoring a label on not-measured rows would inject censoring bias, so we
   mask. No leakage: the mask uses only measurement availability, never the value.

2. Models, all sharing the SAME folds / features / preprocessing:
     - PrevalenceBaseline      : predicts the per-label train prevalence (floor).
     - BinaryRelevance         : one independent classifier per label (masked).
     - MaskedClassifierChain   : a chain that feeds EARLIER labels' (out-of-fold at
                                 train, predicted at test) probabilities as features
                                 to LATER labels — exploits co-occurrence. Handles
                                 the per-label mask and not-measured priors.
     - EnsembleClassifierChains: average of several chains with different orders.

3. Imbalance handling per label (class_weight / sample_weight; optional SMOTE for the
   rare regulated label PFNA), justified by the per-label prevalence.

4. Threshold optimisation PER LABEL on OUT-OF-FOLD probabilities only (never on test),
   per EVAL_PROTOCOL §3.

5. Double evaluation: spatial-block CV (reference) AND group-random CV, with the
   random-minus-spatial Delta. Metrics: macro-AUROC, micro/macro-F1, Hamming, EMR,
   plus per-label AUROC/AP and prevalence. Paired comparison BR vs chains on the
   same folds (Wilcoxon).

6. Pseudo-labeling probe (semi-supervision): MEASURE, do not assume, its effect when
   a label is missing on many rows (calibrated self-training restricted to the
   partially-measured labels). Reported as a delta vs the masked baseline.

Heavy code lives here (smoke-testable on CPU); a notebook only orchestrates Colab.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
)

from . import config as C
from . import features as F
from . import progress as P
from . import splits as S
from . import targets as T

RNG = C.SEED
NOT_MEASURED = -1  # sentinel for a not-measured label in the masked target matrix


# --------------------------------------------------------------------------- masks
def measurement_mask(df: pd.DataFrame, labels=None) -> pd.DataFrame:
    """Boolean frame: True where the analyte was MEASURED for that row.

    Measurement is encoded as a non-NaN ``*_ngL`` (the detected flag is False both for
    non-detects and not-measured, so it cannot carry the mask). Uses availability only,
    not the value -> no target leakage."""
    labels = labels or C.T2_LABELS
    return pd.DataFrame({f"label_{a}": df[C.ngl(a)].notna().to_numpy() for a in labels},
                        index=df.index)


def masked_targets(df: pd.DataFrame, labels=None):
    """Return (Y, M): Y = build_T2 (int 0/1), M = measurement mask (bool). Y on
    not-measured rows is meaningless (build_T2 fills 0) and MUST be ignored via M."""
    labels = labels or C.T2_LABELS
    Y = T.build_T2(df, labels=labels)
    M = measurement_mask(df, labels=labels)
    return Y, M


# ------------------------------------------------------------------- base estimator
def default_base_kind() -> str:
    """Base learner for T2: 'xgb' (GPU device='cuda') when a GPU is present, else 'hgb'
    (sklearn HistGradientBoosting, CPU, fast, NaN-robust). So a Colab GPU run actually
    exercises the GPU through every per-label fit; the CPU smoke stays on 'hgb'."""
    return "xgb" if C.gpu_available() else "hgb"


def make_estimator(kind: str = "hgb", *, class_weight=None, small: bool = False):
    """Per-label base classifier. 'hgb' = HistGradientBoosting (CPU, NaN-robust,
    class_weight); 'xgb' = XGBoost on GPU when available (imbalance via scale_pos_weight,
    set in _fit_one); 'logreg' = light fallback for smoke."""
    if kind == "logreg":
        return LogisticRegression(max_iter=1000, C=1.0,
                                  class_weight=class_weight, random_state=RNG)
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(
            random_state=RNG, eval_metric="logloss", verbosity=0,
            learning_rate=0.1, reg_lambda=1.0,
            n_estimators=(60 if small else 300),
            max_depth=(3 if small else 6),
            **C.xgb_device_params(),          # device='cuda' on GPU, else CPU 'hist'
        )                                     # class_weight -> scale_pos_weight in _fit_one
    params = dict(random_state=RNG, class_weight=class_weight,
                  learning_rate=0.1, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=15)
    if small:
        params.update(max_iter=60, max_depth=3, max_leaf_nodes=15)
    else:
        params.update(max_iter=300, max_leaf_nodes=31)
    return HistGradientBoostingClassifier(**params)


def _proba1(clf, X) -> np.ndarray:
    """P(label=1), robust to a degenerate single-class fit."""
    if not hasattr(clf, "classes_") or len(clf.classes_) < 2:
        const = float(getattr(clf, "_const_p", 0.0))
        return np.full(X.shape[0], const)
    j = list(clf.classes_).index(1)
    return clf.predict_proba(X)[:, j]


class _ConstClf:
    """Fallback when a (label, fold) train slice is single-class."""
    def __init__(self, p): self._const_p = float(p); self.classes_ = np.array([0])
    def fit(self, *a, **k): return self
    def predict_proba(self, X): return np.repeat([[1.0]], len(X), axis=0)


# --------------------------------------------------------------------------- models
@dataclass
class BinaryRelevance:
    """One independent classifier per label, each trained only on rows where the label
    is measured. Imbalance handled per label via class_weight (and optional SMOTE for
    the rare label). No label-label dependency exploited (the reference to beat)."""
    kind: str = "hgb"
    labels: list = field(default_factory=lambda: list(C.T2_LABELS))
    class_weight: str | None = "balanced"
    smote_labels: tuple = ()           # labels to oversample (e.g. ("PFNA",))
    small: bool = False
    name: str = "BinaryRelevance"

    def fit(self, X, Y, M):
        self.models_ = {}
        for a in self.labels:
            col = f"label_{a}"
            m = M[col].to_numpy()
            ya = Y[col].to_numpy()[m]
            Xa = X[m]
            self.models_[a] = _fit_one(self.kind, Xa, ya, self.class_weight,
                                       a in self.smote_labels, self.small)
        return self

    def predict_proba(self, X):
        return np.column_stack([_proba1(self.models_[a], X) for a in self.labels])


@dataclass
class MaskedClassifierChain:
    """Classifier chain over a given label ORDER. Each label is predicted from X plus
    the EARLIER labels in the order. At TRAIN time earlier-label inputs are real (0/1)
    values obtained out-of-fold within the train split (so the chain does not see its
    own target); a not-measured prior is passed as its train prevalence plus a
    'measured' indicator. At TEST time, earlier-label inputs are the chain's own
    PREDICTED probabilities (standard chain inference)."""
    kind: str = "hgb"
    order: tuple = ()
    class_weight: str | None = "balanced"
    smote_labels: tuple = ()
    small: bool = False
    inner_k: int = 3                    # inner folds for out-of-fold prior features
    name: str = "ClassifierChain"

    out_labels: tuple = ()              # canonical output order (defaults to T2_LABELS)

    def __post_init__(self):
        self.order = tuple(self.order) or tuple(C.T2_LABELS)
        self.out_labels = tuple(self.out_labels) or tuple(C.T2_LABELS)

    def fit(self, X, Y, M, groups=None):
        from sklearn.model_selection import GroupKFold, KFold
        n = X.shape[0]
        self.models_, self.prior_prev_ = {}, {}
        # out-of-fold prior columns built progressively (one per earlier label)
        oof_prior = np.zeros((n, len(self.order)))
        oof_measured = np.zeros((n, len(self.order)))
        # inner CV indices (grouped if groups given)
        if groups is not None:
            splitter = list(GroupKFold(n_splits=self.inner_k).split(X, groups=groups))
        else:
            splitter = list(KFold(self.inner_k, shuffle=True,
                                  random_state=RNG).split(X))
        for pos, a in enumerate(self.order):
            col = f"label_{a}"
            m = M[col].to_numpy()
            ya_full = Y[col].to_numpy()
            prev = float(ya_full[m].mean()) if m.any() else 0.0
            self.prior_prev_[a] = prev
            # features = X + prior columns of EARLIER labels
            extra = np.column_stack([oof_prior[:, :pos], oof_measured[:, :pos]]) \
                if pos else np.empty((n, 0))
            Xa_all = np.hstack([X, extra]) if pos else X
            # OUT-OF-FOLD prediction of THIS label, on measured rows, to feed later labels
            oof_a = np.full(n, prev)
            for tr, va in splitter:
                tr_m = tr[m[tr]]
                if len(tr_m) == 0 or len(np.unique(ya_full[tr_m])) < 2:
                    continue
                clf = _fit_one(self.kind, Xa_all[tr_m], ya_full[tr_m],
                               self.class_weight, a in self.smote_labels, self.small)
                va_m = va[m[va]]
                if len(va_m):
                    oof_a[va_m] = _proba1(clf, Xa_all[va_m])
            oof_prior[:, pos] = oof_a
            oof_measured[:, pos] = m.astype(float)
            # final per-label model on all measured rows
            self.models_[a] = (_fit_one(self.kind, Xa_all[m], ya_full[m],
                                        self.class_weight, a in self.smote_labels,
                                        self.small), pos)
        self._n_order = len(self.order)
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        prior = np.zeros((n, self._n_order))
        measured = np.ones((n, self._n_order))  # at test, treat priors as 'available'
        out = {}
        for pos, a in enumerate(self.order):
            clf, _ = self.models_[a]
            extra = np.column_stack([prior[:, :pos], measured[:, :pos]]) \
                if pos else np.empty((n, 0))
            Xa = np.hstack([X, extra]) if pos else X
            p = _proba1(clf, Xa)
            prior[:, pos] = p
            out[a] = p
        # reorder to the canonical output-label order
        return np.column_stack([out[a] for a in self.out_labels])


@dataclass
class EnsembleClassifierChains:
    """Average of several MaskedClassifierChains with different random orders (ECC).
    Reduces the order-sensitivity of a single chain."""
    kind: str = "hgb"
    n_chains: int = 5
    labels: list = field(default_factory=lambda: list(C.T2_LABELS))
    class_weight: str | None = "balanced"
    smote_labels: tuple = ()
    small: bool = False
    name: str = "EnsembleChains"

    def fit(self, X, Y, M, groups=None):
        rng = np.random.default_rng(RNG)
        labels = list(self.labels)
        self.chains_ = []
        for _ in range(self.n_chains):
            order = list(rng.permutation(labels))
            ch = MaskedClassifierChain(kind=self.kind, order=tuple(order),
                                       out_labels=tuple(labels),
                                       class_weight=self.class_weight,
                                       smote_labels=self.smote_labels, small=self.small)
            ch.fit(X, Y, M, groups=groups)
            self.chains_.append(ch)
        return self

    def predict_proba(self, X):
        return np.mean([ch.predict_proba(X) for ch in self.chains_], axis=0)


@dataclass
class FrequencyClassChain:
    """Dong-et-al.-2024-style "4 classes + chain per class", applied to OUR targets.

    Labels are grouped into ``n_classes`` classes ORDERED BY FREQUENCY of presence in the
    dataset, from the least rare (highest positive prevalence) to the rarest, and a
    *cascade* classifier chain runs in that frequency order: each label is predicted from
    X plus the (out-of-fold at train / predicted at test) probabilities of the
    more-frequent labels already in the chain — so rarer PFAS are predicted from the
    commoner predicted ones (the paper's "predict a species from previously predicted
    species"). Built on the leak-free MaskedClassifierChain; the per-class grouping is
    recorded in ``classes_`` for reporting. Frequency is measured on the TRAIN split only.
    """
    n_classes: int = 4
    kind: str = "hgb"
    labels: list = field(default_factory=lambda: list(C.T2_LABELS))
    class_weight: str | None = "balanced"
    smote_labels: tuple = ()
    small: bool = False
    inner_k: int = 3
    name: str = "FreqClassChain"

    def fit(self, X, Y, M, groups=None):
        freq = {}
        for a in self.labels:
            col = f"label_{a}"
            m = M[col].to_numpy()
            freq[a] = float(Y[col].to_numpy()[m].mean()) if m.any() else 0.0
        order = sorted(self.labels, key=lambda a: -freq[a])      # least rare first
        self.freq_, self.order_ = freq, tuple(order)
        # 4 contiguous frequency classes (quartile-like split of the ordered labels)
        self.classes_ = [[order[i] for i in idx]
                         for idx in np.array_split(np.arange(len(order)), self.n_classes)
                         if len(idx)]
        # one cascade chain over the full frequency order (per-class blocks are contiguous)
        self.chain_ = MaskedClassifierChain(
            kind=self.kind, order=self.order_, out_labels=tuple(self.labels),
            class_weight=self.class_weight, smote_labels=self.smote_labels,
            small=self.small, inner_k=self.inner_k)
        self.chain_.fit(X, Y, M, groups=groups)
        return self

    def predict_proba(self, X):
        return self.chain_.predict_proba(X)


@dataclass
class PrevalenceBaseline:
    """Floor: predict each label's train prevalence (measured rows) for every test row."""
    labels: list = field(default_factory=lambda: list(C.T2_LABELS))
    name: str = "Prevalence"

    def fit(self, X, Y, M):
        self.prev_ = np.array([Y[f"label_{a}"].to_numpy()[M[f"label_{a}"].to_numpy()].mean()
                               if M[f"label_{a}"].any() else 0.0 for a in self.labels])
        return self

    def predict_proba(self, X):
        return np.repeat(self.prev_[None, :], X.shape[0], axis=0)


# ----------------------------------------------------------------------- fit helper
def _fit_one(kind, X, y, class_weight, use_smote, small):
    """Fit one binary classifier with imbalance handling. Falls back to a constant
    predictor when the slice is single-class."""
    if len(np.unique(y)) < 2:
        return _ConstClf(float(y.mean()) if len(y) else 0.0)
    if use_smote and (y.sum() >= 6) and (len(y) - y.sum() >= 6):
        try:
            from imblearn.over_sampling import SMOTE
            k = int(min(5, y.sum() - 1))
            X, y = SMOTE(random_state=RNG, k_neighbors=max(1, k)).fit_resample(X, y)
        except Exception as e:                       # pragma: no cover
            warnings.warn(f"SMOTE failed ({e}); using class_weight only")
    # XGBoost has no class_weight -> translate "balanced" to scale_pos_weight from y.
    cw = None if kind == "xgb" else class_weight
    clf = make_estimator(kind, class_weight=cw, small=small)
    if kind == "xgb" and class_weight == "balanced":
        pos = float(y.sum()); neg = float(len(y) - pos)
        clf.set_params(scale_pos_weight=(neg / pos if pos > 0 else 1.0))
    clf.fit(X, y)
    return clf


# --------------------------------------------------------------------- thresholds
def best_thresholds_oof(Y_oof_true, P_oof, M_oof, labels):
    """Per-label decision threshold maximising F1 on OUT-OF-FOLD probabilities only
    (never on test). Returns one threshold per label."""
    thr = {}
    grid = np.linspace(0.05, 0.95, 19)
    for j, a in enumerate(labels):
        col = f"label_{a}"
        m = M_oof[col].to_numpy()
        yt, pp = Y_oof_true[col].to_numpy()[m], P_oof[m, j]
        if len(np.unique(yt)) < 2:
            thr[a] = 0.5
            continue
        f1s = [f1_score(yt, (pp >= t).astype(int), zero_division=0) for t in grid]
        thr[a] = float(grid[int(np.argmax(f1s))])
    return thr


# ----------------------------------------------------------------------- metrics
def per_label_metrics(Y_true, P, M, labels):
    """AUROC / AP / prevalence per label, computed ONLY on measured rows."""
    rows = []
    for j, a in enumerate(labels):
        col = f"label_{a}"
        m = M[col].to_numpy()
        yt, pp = Y_true[col].to_numpy()[m], P[m, j]
        auc = roc_auc_score(yt, pp) if len(np.unique(yt)) > 1 else np.nan
        ap = average_precision_score(yt, pp) if len(np.unique(yt)) > 1 else np.nan
        rows.append({"label": a, "n_measured": int(m.sum()),
                     "prevalence": float(yt.mean()) if len(yt) else np.nan,
                     "AUROC": auc, "AP": ap})
    return pd.DataFrame(rows)


def aggregate_metrics(Y_true, P, M, labels, thresholds):
    """The 5 headline metrics (F1, accuracy, recall, precision, AUC-ROC) in micro and
    macro form, + Hamming and EMR (subset accuracy). All respect the per-label
    measurement mask. Per-label table enriched with the 5 at the chosen thresholds.

    Threshold-free AUROC/AP per label come from `per_label_metrics`; the threshold-based
    metrics come from the shared `src.metrics` module (identical definitions to T1)."""
    from . import metrics as MM
    pl = per_label_metrics(Y_true, P, M, labels)              # AUROC / AP / prevalence / n
    full5 = MM.multilabel_metrics(Y_true, P, M, labels, thresholds)
    fpl = full5["per_label"].set_index("label")
    for col in ("f1", "precision", "recall", "accuracy"):     # enrich per-label table
        pl[col] = [float(fpl.loc[a, col]) for a in labels]

    macro_auroc = float(np.nanmean(pl["AUROC"]))
    macro_ap = float(np.nanmean(pl["AP"]))
    thr = np.array([thresholds[a] for a in labels])
    Yhat = (P >= thr[None, :]).astype(int)
    Ymat = Y_true[[f"label_{a}" for a in labels]].to_numpy()
    Mmat = M[[f"label_{a}" for a in labels]].to_numpy()
    flat_t, flat_p = Ymat[Mmat], Yhat[Mmat]
    hamming = float((flat_t != flat_p).mean())
    full = Mmat.all(axis=1)
    emr = float((Yhat[full] == Ymat[full]).all(axis=1).mean()) if full.any() else np.nan

    mic, mac = full5["micro"], full5["macro"]
    return {
        # AUC-ROC
        "macro_AUROC": macro_auroc, "micro_AUROC": mic["roc_auc"], "macro_AP": macro_ap,
        # F1
        "micro_F1": mic["f1"], "macro_F1": mac["f1"],
        # precision / recall / accuracy
        "micro_precision": mic["precision"], "macro_precision": mac["precision"],
        "micro_recall": mic["recall"], "macro_recall": mac["recall"],
        "micro_accuracy": mic["accuracy"], "macro_accuracy": mac["accuracy"],
        # multilabel-specific
        "Hamming": hamming, "EMR": emr, "subset_accuracy": full5["subset_accuracy"],
        "n_full_panel_rows": int(full.sum()),
    }, pl


# --------------------------------------------------------- pipeline-aware CV runner
def _transform_fold(df_tr, df_te, feature_cols):
    """Fold-aware feature transform (frequency encoding -> leak-free for multilabel)."""
    pipe = F.FeaturePipeline(feature_cols, encode="frequency")
    Xtr, _ = pipe.fit_transform(df_tr)
    Xte, _ = pipe.transform(df_te)
    return Xtr, Xte


def run_cv(df, model_factory, fold, feature_cols, labels=None, *, use_groups=False,
          desc=""):
    """Run one model over a CV scheme. Returns (oof_P, oof_true, oof_M, per_fold_metrics).

    model_factory() -> a fresh model each fold. oof_* are concatenated test predictions
    (each row appears once, in its test fold). per_fold_metrics is a list of dicts (the
    aggregate metrics computed fold-by-fold, BEFORE OOF thresholding -> uses 0.5 grid on
    that fold's own oof for an internal threshold; the global thresholds come later)."""
    labels = labels or C.T2_LABELS
    Y, M = masked_targets(df, labels=labels)
    n = len(df)
    oof_P = np.full((n, len(labels)), np.nan)
    groups_all = df[C.WELL_ID].to_numpy()
    per_fold = []
    folds = list(S.iter_folds(fold))
    dev = "GPU" if C.gpu_available() else "CPU"
    for f, tr, te in P.track(folds, total=len(folds),
                             desc=f"{desc or 'model'} [{dev}]"):
        df_tr, df_te = df[tr], df[te]
        Xtr, Xte = _transform_fold(df_tr, df_te, feature_cols)
        model = model_factory()
        if use_groups and hasattr(model, "fit") and "groups" in model.fit.__code__.co_varnames:
            model.fit(Xtr, Y[tr], M[tr], groups=groups_all[tr])
        else:
            model.fit(Xtr, Y[tr], M[tr])
        P_te = model.predict_proba(Xte)
        oof_P[np.where(te)[0], :] = P_te
        # quick per-fold aggregate at threshold 0.5 (diagnostic only)
        thr05 = {a: 0.5 for a in labels}
        agg, _ = aggregate_metrics(Y[te], P_te, M[te], labels, thr05)
        agg["fold"] = int(f)
        per_fold.append(agg)
    oof_M = M
    return oof_P, Y, oof_M, per_fold


def evaluate_model(df, model_factory, fold, feature_cols, labels=None, *,
                   use_groups=False, desc=""):
    """Full evaluation of ONE model on ONE CV scheme: OOF predictions -> per-label OOF
    thresholds -> global aggregate metrics + per-label table + per-fold spread."""
    labels = labels or C.T2_LABELS
    oof_P, Y, M, per_fold = run_cv(df, model_factory, fold, feature_cols,
                                   labels=labels, use_groups=use_groups, desc=desc)
    thr = best_thresholds_oof(Y, oof_P, M, labels)     # thresholds from OOF only
    agg, pl = aggregate_metrics(Y, oof_P, M, labels, thr)
    pf = pd.DataFrame(per_fold)
    spread = {f"{k}_std": float(pf[k].std()) for k in
              ["macro_AUROC", "micro_F1", "Hamming", "EMR"] if k in pf}
    return {"aggregate": agg, "per_label": pl, "thresholds": thr,
            "per_fold": pf, "spread": spread, "oof_P": oof_P, "Y": Y, "M": M}


# ---------------------------------------------------- pseudo-labeling probe (semi-sup)
def pseudo_label_probe(df, fold, feature_cols, target_labels=("PFBA", "PFPeA", "PFPeS"),
                       conf=0.85, small=False):
    """MEASURE (do not assume) self-training apport for partially-measured labels.

    For each target label, a base HGB is trained on measured rows; on NOT-measured rows
    it predicts, isotonic-calibrated on OOF; high-confidence predictions (p>=conf or
    p<=1-conf) become pseudo-labels; the model is retrained on measured + pseudo rows.
    We report AUROC on a held-out spatial fold WITH vs WITHOUT pseudo-labels. Returns a
    per-label delta. (Restricted to the 3 reduced-panel labels where ~45% of rows lack
    the label; the MNAR mechanism — lab/program — is documented as a caveat.)"""
    labels = list(target_labels)
    Y, M = masked_targets(df, labels=labels)
    rows = []
    for j, a in enumerate(labels):
        col = f"label_{a}"
        deltas = []
        for f, tr, te in S.iter_folds(fold):
            df_tr, df_te = df[tr], df[te]
            Xtr, Xte = _transform_fold(df_tr, df_te, feature_cols)
            m_tr = M[col].to_numpy()[tr]
            m_te = M[col].to_numpy()[te]
            ytr = Y[col].to_numpy()[tr]
            yte = Y[col].to_numpy()[te]
            if m_tr.sum() < 50 or m_te.sum() < 20 or len(np.unique(ytr[m_tr])) < 2 \
               or len(np.unique(yte[m_te])) < 2:
                continue
            # baseline: train on measured only
            base = make_estimator("hgb", class_weight="balanced", small=small)
            base.fit(Xtr[m_tr], ytr[m_tr])
            auc_base = roc_auc_score(yte[m_te], _proba1(base, Xte[m_te]))
            # pseudo-label the not-measured train rows (calibrated)
            unl = ~m_tr
            if unl.sum() < 20:
                deltas.append((auc_base, auc_base)); continue
            # calibrate base probabilities via isotonic on measured rows (in-sample OK
            # here: this is a probe, not the reported model)
            p_meas = _proba1(base, Xtr[m_tr])
            iso = IsotonicRegression(out_of_bounds="clip").fit(p_meas, ytr[m_tr])
            p_unl = iso.transform(_proba1(base, Xtr[unl]))
            conf_mask = (p_unl >= conf) | (p_unl <= 1 - conf)
            if conf_mask.sum() < 10:
                deltas.append((auc_base, auc_base)); continue
            pseudo_y = (p_unl[conf_mask] >= 0.5).astype(int)
            X_aug = np.vstack([Xtr[m_tr], Xtr[unl][conf_mask]])
            y_aug = np.concatenate([ytr[m_tr], pseudo_y])
            if len(np.unique(y_aug)) < 2:
                deltas.append((auc_base, auc_base)); continue
            aug = make_estimator("hgb", class_weight="balanced", small=small)
            aug.fit(X_aug, y_aug)
            auc_aug = roc_auc_score(yte[m_te], _proba1(aug, Xte[m_te]))
            deltas.append((auc_base, auc_aug))
        if deltas:
            d = np.array(deltas)
            rows.append({"label": a, "n_folds": len(d),
                         "AUROC_base": float(d[:, 0].mean()),
                         "AUROC_pseudo": float(d[:, 1].mean()),
                         "delta": float((d[:, 1] - d[:, 0]).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------- paired BR-vs-chain comparison
def paired_compare(per_fold_a: pd.DataFrame, per_fold_b: pd.DataFrame, metric="macro_AUROC"):
    """Wilcoxon signed-rank on per-fold metric (same folds). Returns (mean_diff, pvalue)."""
    from scipy.stats import wilcoxon
    a = per_fold_a.sort_values("fold")[metric].to_numpy()
    b = per_fold_b.sort_values("fold")[metric].to_numpy()
    diff = b - a
    if len(diff) < 3 or np.allclose(diff, 0):
        return float(diff.mean()), np.nan
    try:
        _, p = wilcoxon(a, b)
    except Exception:
        p = np.nan
    return float(diff.mean()), float(p)
