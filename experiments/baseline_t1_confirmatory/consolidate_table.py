"""Produce the consolidated XGBoost T1 baseline table once both runs are complete.

Usage (from repo root):
    python experiments/baseline_t1_confirmatory/consolidate_table.py

Prerequisites:
    experiments/baseline_t1/metrics_spatial.json       <- predictive run (committed)
    experiments/baseline_t1_confirmatory/metrics.json  <- confirmatory run (needs to be run)

Outputs:
    experiments/baseline_t1_confirmatory/consolidated_table.json
    Prints the Markdown table to stdout.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRED_JSON = ROOT / "experiments" / "baseline_t1" / "metrics_spatial.json"
CONF_JSON = ROOT / "experiments" / "baseline_t1_confirmatory" / "metrics.json"
OUT_JSON = ROOT / "experiments" / "baseline_t1_confirmatory" / "consolidated_table.json"


def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if pct:
        return f"{v:+.3f}"
    return f"{v:.3f}"


def load_model(path, model_name, mode):
    with open(path) as f:
        d = json.load(f)
    m = d["models"][model_name]
    sp = m["spatial"]
    rd = m["random"]
    key = "delta_random_minus_spatial" if "delta_random_minus_spatial" in m else "delta"
    dlt = m[key]
    return {"mode": mode, "model": model_name,
            "spatial": sp, "random": rd, "delta": dlt,
            "smoke": d.get("smoke", False)}


def main():
    if not PRED_JSON.exists():
        raise FileNotFoundError(f"Predictive metrics not found: {PRED_JSON}")
    if not CONF_JSON.exists():
        raise FileNotFoundError(
            f"Confirmatory metrics not found: {CONF_JSON}\n"
            f"Run: python experiments/baseline_t1_confirmatory/run_confirmatory_t1.py"
        )

    rows = []
    for path, mode in [(PRED_JSON, "predictive"), (CONF_JSON, "confirmatory")]:
        with open(path) as f:
            d = json.load(f)
        for model_name in ["XGB", "RF"]:
            if model_name in d.get("models", {}):
                rows.append(load_model(path, model_name, mode))

    # Verify no smoke in the rows
    for r in rows:
        if r["smoke"]:
            print(f"WARNING: {r['mode']}/{r['model']} is a SMOKE run — numbers not comparable to full run!")

    table = []
    for r in rows:
        sp = r["spatial"]
        rd = r["random"]
        dlt = r["delta"]
        table.append({
            "mode": r["mode"],
            "model": r["model"],
            "AUC_spatial": sp["roc_auc"],
            "AUC_random": rd["roc_auc"],
            "delta_AUC": dlt["roc_auc"],
            "F1_spatial": sp["f1"],
            "recall_spatial": sp["recall"],
            "precision_spatial": sp["precision"],
            "pr_auc_spatial": sp.get("pr_auc"),
            "brier_spatial": sp.get("brier"),
            "balanced_acc_spatial": sp.get("balanced_accuracy"),
        })

    # Confirmatory - predictive gap
    conf_xgb_sp = next((r["spatial"]["roc_auc"] for r in rows
                        if r["mode"] == "confirmatory" and r["model"] == "XGB"), None)
    pred_xgb_sp = next((r["spatial"]["roc_auc"] for r in rows
                        if r["mode"] == "predictive" and r["model"] == "XGB"), None)
    gap = (conf_xgb_sp - pred_xgb_sp) if (conf_xgb_sp and pred_xgb_sp) else None

    out = {"table": table,
           "confirmatory_minus_predictive_AUC_spatial_XGB": gap,
           "interpretation": {
               "gap": "Chemistry adds this much AUC in spatial CV (partially tautological for T1a)",
               "spatial": "honest generalisation metric (GNN must beat this)",
               "random": "optimistic/upper-bound only",
               "delta": "spatial inflation (random minus spatial)",
           }}

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Saved: {OUT_JSON}")

    # Print markdown table
    header = ("mode", "model", "AUC_sp", "AUC_rd", "ΔAUC",
              "F1_sp", "recall_sp", "prec_sp", "PR-AUC_sp", "Brier_sp")
    widths = [14, 5, 8, 8, 7, 8, 9, 8, 10, 9]
    print()
    print("## Consolidated XGBoost T1 Baseline Table")
    print()

    def row_str(vals):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, widths)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    print(row_str(header))
    print(sep)
    for r in table:
        vals = (r["mode"], r["model"],
                fmt(r["AUC_spatial"]), fmt(r["AUC_random"]), fmt(r["delta_AUC"], pct=True),
                fmt(r["F1_spatial"]), fmt(r["recall_spatial"]), fmt(r["precision_spatial"]),
                fmt(r["pr_auc_spatial"]), fmt(r["brier_spatial"]))
        print(row_str(vals))

    print()
    if gap is not None:
        print(f"**Confirmatory - predictive gap (XGB, spatial AUC)**: {gap:+.3f}")
        print(f"  => Chemistry adds {gap:.3f} AUC points in spatial CV.")
        print(f"  => CAVEAT: T1a is derived from NGL concentrations -> partially tautological.")
    print()
    print("**GNN wall to beat**: XGB predictive spatial AUC = "
          f"{fmt(pred_xgb_sp)} (RF = "
          f"{next((fmt(r['spatial']['roc_auc']) for r in rows if r['mode']=='predictive' and r['model']=='RF'), '—')})")


if __name__ == "__main__":
    main()
