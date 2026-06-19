"""Cross-validation splits that respect the two leakage axes (eval C2/C3):

  * group by `gm_well_id` so repeated samplings of one well never straddle a split
    (otherwise pseudo-replicates inflate scores);
  * spatial block CV (KMeans on per-well coordinates) so neighbouring autocorrelated
    wells fall in the SAME fold — the honest reference metric.

Two schemes are exposed:
  - `spatial_block_folds`  : LeaveOneBlockOut over k spatial KMeans blocks (reference).
  - `group_random_folds`   : GroupKFold over wells, spatially shuffled (for the
                              random-vs-spatial Δ artifact test).
Both return an array `fold[i]` per ROW; test fold f = rows with fold == f.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

from . import config as C


def well_coordinates(df: pd.DataFrame):
    """One (lat, lon) per well = mean over its samplings. Returns (well_ids, coords)."""
    g = df.groupby(C.WELL_ID)[[C.LAT, C.LON]].mean()
    return g.index.to_numpy(), g.to_numpy()


def spatial_block_labels(df: pd.DataFrame, k: int = C.N_SPATIAL_BLOCKS,
                         seed: int = C.SEED) -> pd.Series:
    """KMeans block id per ROW, assigned at WELL level (so a well = one block)."""
    well_ids, coords = well_coordinates(df)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    well_block = dict(zip(well_ids, km.fit_predict(coords)))
    return df[C.WELL_ID].map(well_block).astype(int).rename("spatial_block")


def spatial_block_folds(df: pd.DataFrame, k: int = C.N_SPATIAL_BLOCKS,
                        seed: int = C.SEED) -> np.ndarray:
    """Reference spatial CV: fold == spatial block (LeaveOneBlockOut). 0 well and 0
    block shared between train and test."""
    return spatial_block_labels(df, k=k, seed=seed).to_numpy()


def group_random_folds(df: pd.DataFrame, k: int = C.N_RANDOM_FOLDS,
                       seed: int = C.SEED) -> np.ndarray:
    """Spatially-agnostic GroupKFold over wells (still leak-free re. pseudo-replicates,
    but neighbours may split) — used only to measure the optimistic Δ."""
    wells = df[C.WELL_ID].to_numpy()
    # deterministic shuffle of group order for stability across runs
    uniq = pd.Series(df[C.WELL_ID].unique()).sample(frac=1.0, random_state=seed).to_numpy()
    order = {w: i for i, w in enumerate(uniq)}
    proxy = np.array([order[w] for w in wells])
    fold = np.empty(len(df), dtype=int)
    gkf = GroupKFold(n_splits=k)
    # GroupKFold needs X,y placeholders; group by well via proxy index
    for f, (_, test_idx) in enumerate(gkf.split(proxy.reshape(-1, 1), groups=wells)):
        fold[test_idx] = f
    return fold


def iter_folds(fold: np.ndarray):
    """Yield (f, train_mask, test_mask) for each fold id."""
    for f in np.unique(fold):
        test = fold == f
        yield int(f), ~test, test


def assert_no_group_leak(df: pd.DataFrame, fold: np.ndarray):
    """Raise if any well appears in more than one fold (guards C2)."""
    per_well = pd.DataFrame({"w": df[C.WELL_ID].to_numpy(), "f": fold}).groupby("w")["f"].nunique()
    bad = int((per_well > 1).sum())
    if bad:
        raise AssertionError(f"{bad} wells straddle folds (group leak)")


def block_prevalence(y, fold: np.ndarray) -> pd.DataFrame:
    """Per-fold size and target prevalence (sanity: no degenerate fold)."""
    y = np.asarray(y)
    rows = [{"fold": f, "n": int(m.sum()), "prevalence": float(y[m].mean())}
            for f, _, m in iter_folds(fold)]
    return pd.DataFrame(rows)
