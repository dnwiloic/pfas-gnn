"""Lightweight progress bars with ETA (remaining-time estimate).

Uses `tqdm.auto` (nice widget in Colab/Jupyter, text bar in a terminal) when available,
and degrades to a silent pass-through otherwise — so the same `src/` code runs in a
notebook (with bars) and under a headless smoke-test (without).
"""
from __future__ import annotations

import time

try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except Exception:                                    # pragma: no cover
    _HAS_TQDM = False


def track(iterable, desc: str = "", total=None, enable: bool = True):
    """Wrap an iterable with a tqdm progress bar (elapsed + ETA) if available."""
    if _HAS_TQDM and enable:
        return _tqdm(iterable, desc=desc, total=total, leave=True, dynamic_ncols=True)
    return iterable


class Stage:
    """Manual progress bar for known-length stages, exposing an ETA. Use when you want
    to set a custom postfix (e.g. last fold's AUC) alongside the remaining-time estimate.

        st = Stage(total=8, desc="XGB / spatial")
        for f in folds:
            ...
            st.update(1, auc=0.62)
        st.close()
    """
    def __init__(self, total: int, desc: str = "", enable: bool = True):
        self.t0 = time.time()
        self.bar = _tqdm(total=total, desc=desc, leave=True, dynamic_ncols=True) \
            if (_HAS_TQDM and enable) else None
        self.total, self.n = total, 0

    def update(self, step: int = 1, **postfix):
        self.n += step
        if self.bar is not None:
            if postfix:
                self.bar.set_postfix(postfix)
            self.bar.update(step)

    def eta_s(self) -> float:
        """Estimated remaining seconds from the average rate so far."""
        if self.n == 0:
            return float("nan")
        rate = (time.time() - self.t0) / self.n
        return rate * (self.total - self.n)

    def close(self):
        if self.bar is not None:
            self.bar.close()
