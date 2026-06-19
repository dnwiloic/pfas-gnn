#!/usr/bin/env python3
"""Definitive T2 label set under the chosen scheme:
  HYBRID = EPA 2024 MCL for regulated compounds, analytical 2.0 ng/L fallback
           for all others; DETECTION GUARD (eval C1) applied to every label.

EPA 2024 final NPDWR individual MCLs (ng/L): PFOA 4, PFOS 4, PFHxS 10, PFNA 10,
HFPO-DA 10. (PFBS has no individual MCL -> Hazard-Index only -> falls back to 2.0.)
Curation rule (unchanged from profiling): keep a label if measured_frac >= 0.50;
flag usable if prevalence >= 0.05. Constant/near-constant labels are dropped.
"""
import json
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
DATA = "data/CA-PFAS-ASGWS.parquet"
ANALYTICAL = 2.0

EPA_MCL = {"PFOA": 4.0, "PFOS": 4.0, "PFHxS": 10.0, "PFNA": 10.0, "HFPO_DA": 10.0}

# candidate analytes (the 15 curated in profiling + HFPO_DA for completeness)
CANDIDATES = ["PFOS", "PFHxS", "PFOA", "PFBS", "PFHxA", "FTS_6_2", "PFHpA",
              "PFBA", "FTS_8_2", "PFPeA", "PFNA", "PFPeS", "NEtFOSAA",
              "NMeFOSAA", "PFDA", "HFPO_DA"]


def main():
    df = pd.read_parquet(DATA)
    n = len(df)
    rows, labels = {}, {}
    for comp in CANDIDATES:
        ngl, det = f"{comp}_ngL", f"{comp}_detected"
        if ngl not in df.columns:
            continue
        thr = EPA_MCL.get(comp, ANALYTICAL)
        regulated = comp in EPA_MCL
        x = df[ngl]
        measured = x.notna()
        detected = df[det].fillna(False).astype(bool) if det in df.columns else measured
        y = ((x > thr) & detected).astype(int)
        prev = float(y.mean())
        rows[comp] = {
            "threshold_ngL": thr,
            "threshold_source": "EPA_MCL_2024" if regulated else "analytical_2.0",
            "regulated": regulated,
            "measured_frac": float(measured.mean()),
            "detected_frac": float(detected.mean()),
            "prevalence_hybrid": prev,
            "prevalence_old_2.0": float(((x > ANALYTICAL) & detected).mean()),
            "usable_ge5pct": bool(prev >= 0.05),
            "keep": bool(measured.mean() >= 0.50 and prev > 0.0),
        }
        labels[comp] = y

    retained = [c for c in rows if rows[c]["keep"]]
    usable = [c for c in retained if rows[c]["usable_ge5pct"]]
    rare_regulated = [c for c in retained
                      if rows[c]["regulated"] and not rows[c]["usable_ge5pct"]]

    # co-occurrence (Pearson) among retained labels
    L = pd.DataFrame({c: labels[c] for c in retained})
    corr = L.corr().round(3)
    pos_per_row = float(L.sum(axis=1).mean())
    clean_rows = float((L.sum(axis=1) == 0).mean())

    out = {
        "scheme": "hybrid_EPA_MCL_with_analytical_2.0_fallback",
        "detection_guard": True, "seed": SEED, "n_rows": n,
        "per_label": rows,
        "retained_labels": retained,
        "usable_ge5pct": usable,
        "rare_regulated_flagged": rare_regulated,
        "n_retained": len(retained), "n_usable": len(usable),
        "labels_per_row_mean": pos_per_row,
        "rows_all_negative_frac": clean_rows,
        "cooccurrence_pearson": corr.to_dict(),
    }
    with open("experiments/profilage/t2_hybrid_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- report ----
    print(f"rows={n}   scheme=HYBRID(EPA MCL + 2.0 fallback) + detection guard\n")
    hdr = f"{'label':10s} {'thr':>6s} {'src':>14s} {'meas%':>6s} {'prev_2.0':>8s} {'PREV':>7s} {'use':>4s}"
    print(hdr); print("-" * len(hdr))
    order = sorted(rows, key=lambda c: -rows[c]["prevalence_hybrid"])
    for c in order:
        r = rows[c]
        flag = "yes" if r["usable_ge5pct"] else ("reg*" if r["regulated"] else "drop")
        print(f"{c:10s} {r['threshold_ngL']:6.0f} {r['threshold_source']:>14s} "
              f"{r['measured_frac']*100:5.1f}% {r['prevalence_old_2.0']:8.3f} "
              f"{r['prevalence_hybrid']:7.3f}  {flag:>4s}")
    print(f"\nretained (measured>=50%): {len(retained)} -> {retained}")
    print(f"usable (>=5%):            {len(usable)} -> {usable}")
    print(f"rare regulated (flag):    {rare_regulated}")
    print(f"labels/row mean: {pos_per_row:.2f}   all-negative rows: {clean_rows*100:.1f}%")
    print("\ntop co-occurrences (|r|>=0.6):")
    seen = set()
    for a in retained:
        for b in retained:
            if a < b and abs(corr.loc[a, b]) >= 0.6 and (a, b) not in seen:
                seen.add((a, b)); print(f"  {a:9s} ~ {b:9s} {corr.loc[a,b]:+.2f}")
    print("\nwrote experiments/profilage/t2_hybrid_metrics.json")


if __name__ == "__main__":
    main()
