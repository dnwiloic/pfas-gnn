#!/usr/bin/env python3
"""Recompute T2 labels under REGULATORY thresholds (per-compound) instead of the
uniform 2.0 ng/L analytical threshold. Deterministic, no model training.

Regulatory references (June 2026):
- US EPA 2024 final NPDWR MCLs: PFOA 4, PFOS 4, PFHxS 10, PFNA 10, HFPO-DA 10 ng/L
  (PFHxS/PFNA/HFPO-DA proposed for rescission May 2026; PFOA/PFOS upheld).
- California State Water Board Notification (NL) / Response (RL) levels, rev. Oct 2025:
  PFOA NL 4 / RL 10 ; PFOS NL 4 / RL 40 ; PFHxS NL 3 / RL 10 ;
  PFHxA NL 1000 / RL 10000 ; PFBS NL 500 / RL 5000 ng/L.
All labels apply a DETECTION GUARD (cf. eval C1): exceedance counts only if the
analyte is also flagged detected (a high reporting limit on a non-detect must not
trigger a positive).
"""
import json
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
DATA = "data/CA-PFAS-ASGWS.parquet"
ANALYTICAL = 2.0  # current uniform T2 threshold

# Per-compound regulatory thresholds (ng/L). None = no individual regulatory level.
# Each entry: epa_mcl, ca_nl, ca_rl.
REG = {
    "PFOA":    {"epa_mcl": 4.0,  "ca_nl": 4.0,   "ca_rl": 10.0},
    "PFOS":    {"epa_mcl": 4.0,  "ca_nl": 4.0,   "ca_rl": 40.0},
    "PFHxS":   {"epa_mcl": 10.0, "ca_nl": 3.0,   "ca_rl": 10.0},
    "PFNA":    {"epa_mcl": 10.0, "ca_nl": None,  "ca_rl": None},
    "HFPO_DA": {"epa_mcl": 10.0, "ca_nl": None,  "ca_rl": None},
    "PFBS":    {"epa_mcl": None, "ca_nl": 500.0, "ca_rl": 5000.0},
    "PFHxA":   {"epa_mcl": None, "ca_nl": 1000.0,"ca_rl": 10000.0},
}


def pick(d, framework):
    """Choose a single threshold per compound for a framework.
    - 'epa'      : EPA MCL only (compounds without an MCL are dropped).
    - 'ca_nl'    : California Notification Level; fall back to EPA MCL if no NL.
    - 'ca_rl'    : California Response Level; fall back to EPA MCL if no RL.
    - 'strictest': lowest available regulatory number across EPA/CA NL.
    """
    epa, nl, rl = d["epa_mcl"], d["ca_nl"], d["ca_rl"]
    if framework == "epa":
        return epa
    if framework == "ca_nl":
        return nl if nl is not None else epa
    if framework == "ca_rl":
        return rl if rl is not None else epa
    if framework == "strictest":
        cands = [x for x in (epa, nl) if x is not None]
        return min(cands) if cands else None
    raise ValueError(framework)


def main():
    df = pd.read_parquet(DATA)
    n = len(df)
    frameworks = ["epa", "ca_nl", "ca_rl", "strictest"]
    out = {"n_rows": n, "seed": SEED, "analytical_threshold": ANALYTICAL,
           "frameworks": {}, "per_compound": {}}

    for comp, d in REG.items():
        ngl, det = f"{comp}_ngL", f"{comp}_detected"
        if ngl not in df.columns:
            out["per_compound"][comp] = {"present": False}
            continue
        x = df[ngl]
        measured = x.notna()
        detected = df[det].fillna(False).astype(bool) if det in df.columns else measured
        rec = {
            "present": True,
            "measured_frac": float(measured.mean()),
            "detected_frac": float(detected.mean()),
            "thresholds": {k: d[k] for k in ("epa_mcl", "ca_nl", "ca_rl")},
            "prevalence_analytical_2.0": float(((x > ANALYTICAL) & detected).mean()),
        }
        for fw in frameworks:
            thr = pick(d, fw)
            rec[f"prev_{fw}"] = (None if thr is None
                                 else float(((x > thr) & detected).mean()))
            rec[f"thr_{fw}"] = thr
        out["per_compound"][comp] = rec

    # Framework-level summary: which compounds yield a usable label (prev >= 5%).
    for fw in frameworks:
        comps = {}
        for comp, d in REG.items():
            rec = out["per_compound"].get(comp, {})
            if not rec.get("present"):
                continue
            p = rec.get(f"prev_{fw}")
            if p is not None:
                comps[comp] = {"threshold": rec[f"thr_{fw}"], "prevalence": p,
                               "measured_frac": rec["measured_frac"]}
        usable = {c: v for c, v in comps.items() if v["prevalence"] >= 0.05}
        out["frameworks"][fw] = {
            "n_labels": len(comps),
            "n_usable_ge5pct": len(usable),
            "usable_labels": sorted(usable, key=lambda c: -usable[c]["prevalence"]),
            "compounds": comps,
        }

    with open("experiments/profilage/t2_regulatory_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- console report ----
    print(f"rows={n}\n")
    hdr = f"{'compound':9s} {'meas%':>6s} {'det%':>6s} {'an2.0':>7s} " \
          f"{'epa':>7s} {'ca_nl':>7s} {'ca_rl':>7s} {'strict':>7s}"
    print(hdr); print("-" * len(hdr))
    for comp in REG:
        r = out["per_compound"][comp]
        if not r.get("present"):
            print(f"{comp:9s}  (absent)"); continue
        def f(p): return "  -  " if p is None else f"{p:6.3f}"
        print(f"{comp:9s} {r['measured_frac']*100:5.1f}% {r['detected_frac']*100:5.1f}% "
              f"{r['prevalence_analytical_2.0']:7.3f} "
              f"{f(r['prev_epa'])} {f(r['prev_ca_nl'])} {f(r['prev_ca_rl'])} {f(r['prev_strictest'])}")
    print()
    for fw in frameworks:
        s = out["frameworks"][fw]
        print(f"[{fw:9s}] labels={s['n_labels']:2d}  usable(>=5%)={s['n_usable_ge5pct']:2d}  "
              f"-> {s['usable_labels']}")
    print("\nwrote experiments/profilage/t2_regulatory_metrics.json")


if __name__ == "__main__":
    main()
