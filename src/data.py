"""Load and clean the raw PFAS table. No feature engineering, no target building here
(those live in features.py / targets.py) — this only yields a tidy DataFrame."""
from __future__ import annotations

import pandas as pd

from . import config as C


def load_raw(path=None, smoke: bool = False, smoke_n: int = 600, seed: int = C.SEED):
    """Load the parquet. If `smoke`, return a small reproducible subsample of WELLS
    (whole wells, so group/spatial logic stays meaningful)."""
    path = path or C.DATA_PARQUET
    df = pd.read_parquet(path)
    if smoke:
        rng = pd.Series(df[C.WELL_ID].unique())
        keep = rng.sample(n=min(smoke_n, len(rng)), random_state=seed)
        df = df[df[C.WELL_ID].isin(set(keep))].reset_index(drop=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicated / quasi-empty / constant context columns flagged in profiling.
    Idempotent: only drops columns that are present."""
    df = df.copy()
    drop = [c for c in C.DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)
    # parse date if needed
    if C.DATE_COL in df.columns and not pd.api.types.is_datetime64_any_dtype(df[C.DATE_COL]):
        df[C.DATE_COL] = pd.to_datetime(df[C.DATE_COL], errors="coerce")
    return df


def load(path=None, smoke: bool = False, **kw) -> pd.DataFrame:
    """Convenience: load_raw + clean."""
    return clean(load_raw(path=path, smoke=smoke, **kw))
