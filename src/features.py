"""Feature assembly + a fold-aware transformer (anti-leakage).

Hard rule: NOTHING from config.LEAKAGE_BLOCKLIST may enter the feature matrix.
All fitted statistics (imputation medians, scaler, category sets, target/frequency
encodings) come from the TRAIN rows only; high-cardinality target-encoding uses an
inner K-fold so train rows get out-of-fold codes (eval: encoding out-of-fold).

Usage:
    pipe = FeaturePipeline(cols, encode="target")     # "target" (binary y) | "frequency"
    Xtr, names = pipe.fit_transform(df_tr, y_tr)
    Xte, _     = pipe.transform(df_te)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config as C


# --------------------------------------------------------------------- temporal
def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Derive year + cyclical month from collection_date (profiling §7-I)."""
    df = df.copy()
    dt = pd.to_datetime(df[C.DATE_COL], errors="coerce")
    df["year"] = dt.dt.year.astype("float64")
    m = dt.dt.month.astype("float64")
    df["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * m / 12.0)
    return df


# ------------------------------------------------------- k-fold target encoder
class _KFoldTargetEncoder:
    """Smoothed target mean per category, with out-of-fold codes for train rows."""

    def __init__(self, smoothing: float = 20.0, n_splits: int = 5, seed: int = C.SEED):
        self.smoothing, self.n_splits, self.seed = smoothing, n_splits, seed
        self.mapping_, self.global_ = {}, 0.0

    def _smoothed(self, s: pd.Series, y: pd.Series):
        stats = y.groupby(s).agg(["mean", "count"])
        g = y.mean()
        return (stats["count"] * stats["mean"] + self.smoothing * g) / (stats["count"] + self.smoothing)

    def fit_transform(self, s: pd.Series, y: pd.Series) -> np.ndarray:
        s = s.astype("object").fillna("__nan__").reset_index(drop=True)
        y = pd.Series(np.asarray(y), dtype="float64")
        self.global_ = float(y.mean())
        oof = np.full(len(s), self.global_)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for tr, va in kf.split(s):
            m = self._smoothed(s.iloc[tr], y.iloc[tr])
            oof[va] = s.iloc[va].map(m).fillna(y.iloc[tr].mean()).to_numpy()
        self.mapping_ = self._smoothed(s, y).to_dict()      # full-train map for transform
        return oof

    def transform(self, s: pd.Series) -> np.ndarray:
        s = s.astype("object").fillna("__nan__")
        return s.map(self.mapping_).fillna(self.global_).to_numpy()


class _FrequencyEncoder:
    """Relative frequency per category (no target needed -> trivially leak-free)."""

    def __init__(self): self.mapping_, self.default_ = {}, 0.0

    def fit_transform(self, s: pd.Series, y=None) -> np.ndarray:
        s = s.astype("object").fillna("__nan__")
        self.mapping_ = (s.value_counts(normalize=True)).to_dict()
        self.default_ = 0.0
        return s.map(self.mapping_).fillna(self.default_).to_numpy()

    def transform(self, s: pd.Series) -> np.ndarray:
        s = s.astype("object").fillna("__nan__")
        return s.map(self.mapping_).fillna(self.default_).to_numpy()


# ------------------------------------------------------------------- pipeline
class FeaturePipeline:
    def __init__(self, feature_cols, encode: str = "target",
                 missing_indicator_thresh: float = 0.20, seed: int = C.SEED):
        leak = set(feature_cols) & set(C.LEAKAGE_BLOCKLIST)
        if leak:
            raise ValueError(f"blocklisted (target-leaking) columns in features: {sorted(leak)}")
        self.feature_cols = list(feature_cols)
        self.encode, self.thresh, self.seed = encode, missing_indicator_thresh, seed
        self.cat_low = [c for c in self.feature_cols if c in C.CATEGORICAL_LOW_CARD]
        self.cat_high = [c for c in self.feature_cols if c in C.CATEGORICAL_HIGH_CARD]
        self.num_cols = [c for c in self.feature_cols
                         if c not in self.cat_low + self.cat_high]
        self.log_cols = [c for c in self.num_cols if c in C.LOG1P_FEATS]

    # ---- helpers
    def _num_block(self, df, fit):
        X = df[self.num_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
        for c in self.log_cols:                      # log1p on non-negative magnitudes
            X[c] = np.log1p(X[c].clip(lower=0))
        if fit:
            self.medians_ = X.median()
            self.miss_cols_ = [c for c in self.num_cols if X[c].isna().mean() > self.thresh]
            self.mu_ = X.fillna(self.medians_).mean()
            self.sd_ = X.fillna(self.medians_).std(ddof=0).replace(0, 1.0)
        miss = pd.DataFrame({f"{c}__missing": df[c].isna().astype("float64").to_numpy()
                             for c in self.miss_cols_}, index=df.index)
        Xs = (X.fillna(self.medians_) - self.mu_) / self.sd_
        return pd.concat([Xs, miss], axis=1)

    def _onehot(self, df, fit):
        if fit:
            self.cat_levels_ = {c: sorted(df[c].astype("object").fillna("__nan__").unique())
                                for c in self.cat_low}
        blocks = []
        for c in self.cat_low:
            s = df[c].astype("object").fillna("__nan__")
            for lv in self.cat_levels_[c]:
                blocks.append(pd.Series((s == lv).astype("float64").to_numpy(),
                                        index=df.index, name=f"{c}={lv}"))
        return pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=df.index)

    def _highcard(self, df, fit, y):
        if not self.cat_high:
            return pd.DataFrame(index=df.index)
        out = {}
        if fit:
            self.encoders_ = {}
        for c in self.cat_high:
            if fit:
                if self.encode == "target":
                    if y is None:
                        raise ValueError("encode='target' needs y_train (binary)")
                    enc = _KFoldTargetEncoder(seed=self.seed)
                    code = enc.fit_transform(df[c], y)
                else:
                    enc = _FrequencyEncoder()
                    code = enc.fit_transform(df[c])
                self.encoders_[c] = enc
            else:
                code = self.encoders_[c].transform(df[c])
            out[f"{c}__enc"] = np.asarray(code)
        return pd.DataFrame(out, index=df.index)

    # ---- API
    def fit_transform(self, df: pd.DataFrame, y=None):
        df = add_temporal(df)
        parts = [self._num_block(df, True), self._onehot(df, True),
                 self._highcard(df, True, y)]
        X = pd.concat(parts, axis=1)
        self.out_names_ = list(X.columns)
        return X.to_numpy(dtype="float64"), self.out_names_

    def transform(self, df: pd.DataFrame):
        df = add_temporal(df)
        parts = [self._num_block(df, False), self._onehot(df, False),
                 self._highcard(df, False, None)]
        X = pd.concat(parts, axis=1)[self.out_names_]
        return X.to_numpy(dtype="float64"), self.out_names_
