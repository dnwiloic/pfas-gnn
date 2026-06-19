"""Build prediction targets from the PFAS measurements, with the detection guard
(eval condition C1): an exceedance counts only on a DETECTED measurement, never on a
high reporting limit carried by a non-detect.

T1a  (primary, binary)   : EPA 2024 — PFOA>4 OR PFOS>4 OR Hazard Index >= 1.
T1b  (secondary, binary) : sum_pfas_ngL > 70 (legacy advisory on the summed series).
T2   (multilabel)        : hybrid EPA-MCL / analytical-2.0 per analyte (config.T2_LABELS).

All target builders read *_ngL and *_detected (which are leakage columns) and return
plain int/array targets; callers then exclude those columns from features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _guarded_conc(df: pd.DataFrame, analyte: str) -> pd.Series:
    """Concentration if detected, else 0 (detection guard). Missing -> 0."""
    c = df[C.ngl(analyte)].astype(float)
    det = df[C.detected(analyte)].fillna(False).astype(bool)
    return c.where(det, 0.0).fillna(0.0)


def hazard_index(df: pd.DataFrame) -> pd.Series:
    """EPA Hazard Index = sum_i C_i / HBWC_i over {PFHxS, PFNA, HFPO_DA, PFBS},
    using detection-guarded concentrations."""
    hi = pd.Series(0.0, index=df.index)
    for a, hbwc in C.HI_HBWC.items():
        hi = hi + _guarded_conc(df, a) / hbwc
    return hi


def build_T1a(df: pd.DataFrame) -> pd.Series:
    """EPA 2024 regulatory exceedance (binary)."""
    pfoa = _guarded_conc(df, "PFOA") > C.EPA_MCL["PFOA"]
    pfos = _guarded_conc(df, "PFOS") > C.EPA_MCL["PFOS"]
    hi = hazard_index(df) >= 1.0
    return (pfoa | pfos | hi).astype(int).rename("T1a")


def build_T1b(df: pd.DataFrame) -> pd.Series:
    """Secondary: summed-PFAS advisory exceedance."""
    return (df["sum_pfas_ngL"].astype(float) > C.T1B_SUM_THRESHOLD).astype(int).rename("T1b")


def build_T2(df: pd.DataFrame, labels=None) -> pd.DataFrame:
    """Multilabel matrix (hybrid thresholds + detection guard).
    Columns = config.T2_LABELS by default. dtype int (0/1)."""
    labels = labels or C.T2_LABELS
    out = {}
    for a in labels:
        thr = C.t2_threshold(a)
        out[f"label_{a}"] = (_guarded_conc(df, a) > thr).astype(int)
    return pd.DataFrame(out, index=df.index)


def build_all(df: pd.DataFrame, t2_labels=None) -> dict:
    """Convenience bundle of every target plus a few diagnostics."""
    T1a, T1b = build_T1a(df), build_T1b(df)
    T2 = build_T2(df, labels=t2_labels)
    return {
        "T1a": T1a, "T1b": T1b, "T2": T2,
        "prevalence": {
            "T1a": float(T1a.mean()), "T1b": float(T1b.mean()),
            **{c: float(T2[c].mean()) for c in T2.columns},
        },
    }
