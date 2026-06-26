"""Figure phare : (A) lift v1->v2 du stack complet ; (B) ordre HGT<fusion<stacking +
calibration sous le mur tabulaire v2. Lit les metrics.json committes, n'invente rien."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
v2 = json.load(open(HERE / "metrics.json"))
v1 = json.load(open(HERE.parent / "hgt_fusion_stacking_t1" / "metrics.json"))

archs = ["hgt_standalone", "embedding_fusion", "stacking"]
labels = ["HGT\nseul", "Fusion\nembeddings", "Stacking\n(HGT+XGB+LGBM)"]
oof = lambda m, a: m["spatial"]["architectures"][a]["metrics"]["roc_auc"]
ece = lambda m, a: m["spatial"]["architectures"][a]["metrics"].get("ece", np.nan)
wall_v2 = v2["comparison"]["in_run_xgb_wall_auc_mean"]
wall_v1 = v1["comparison"]["in_run_xgb_wall_auc_mean"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4))

# ---- Panel A : lift v1 -> v2
x = np.arange(len(archs)); w = 0.36
a1 = [oof(v1, a) for a in archs]; a2 = [oof(v2, a) for a in archs]
axA.bar(x - w/2, a1, w, label="v1 (features de base)", color="#9db4c0")
axA.bar(x + w/2, a2, w, label="v2 (collecte enrichie)", color="#3d6b9c")
for xi, (b1, b2) in enumerate(zip(a1, a2)):
    axA.text(xi - w/2, b1 + .005, f"{b1:.3f}", ha="center", fontsize=8.5)
    axA.text(xi + w/2, b2 + .005, f"{b2:.3f}", ha="center", fontsize=8.5, weight="bold")
    axA.annotate(f"+{b2-b1:.3f}", (xi, max(b1, b2) + .022), ha="center",
                 fontsize=9, color="#2e7d32", weight="bold")
axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=9)
axA.set_ylabel("AUC spatiale (global-OOF)"); axA.set_ylim(0.60, 0.78)
axA.set_title("A. L'enrichissement v2 relève tout le pipeline (+0,05 AUC)", fontsize=10.5)
axA.legend(loc="upper left", fontsize=8.5); axA.grid(axis="y", alpha=0.3)

# ---- Panel B : ordre du stack v2 + mur + calibration
a2v = [oof(v2, a) for a in archs]; eces = [ece(v2, a) for a in archs]
bars = axB.bar(x, a2v, 0.55, color=["#6a8caf", "#4a7aa7", "#2e5d8a"])
axB.axhline(wall_v2, ls="--", color="#b03a2e", lw=2,
            label=f"mur XGB tabulaire v2 = {wall_v2:.3f}")
for xi, (b, e) in enumerate(zip(a2v, eces)):
    axB.text(xi, b + .004, f"AUC {b:.3f}", ha="center", fontsize=9, weight="bold")
    axB.text(xi, b - .028, f"ECE {e:.3f}", ha="center", fontsize=8.5, color="white")
axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=9)
axB.set_ylabel("AUC spatiale (global-OOF)"); axB.set_ylim(0.60, 0.78)
axB.set_title("B. Stacking = meilleur prédicteur graphe + mieux calibré\n"
              "(approche le mur tabulaire, sans le dépasser robustement)", fontsize=10.5)
axB.legend(loc="upper left", fontsize=8.5); axB.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(FIG / "figure_phare_v2.png", dpi=140)
print(f"[fig] {FIG/'figure_phare_v2.png'}")

# ---- triplet scatter (secondaire)
fig2, ax = plt.subplots(figsize=(5.6, 5.4))
for a, lab, c in zip(archs, ["HGT", "Fusion", "Stacking"], ["#6a8caf", "#4a7aa7", "#2e5d8a"]):
    sp = oof(v2, a); rd = v2["random"]["architectures"][a]["metrics"]["roc_auc"]
    ax.scatter(rd, sp, s=120, color=c, label=f"{lab} (Δ={rd-sp:+.3f})", zorder=3)
ax.plot([0.5, 0.95], [0.5, 0.95], ls=":", color="grey", label="y=x (pas d'inflation)")
ax.set_xlabel("AUC random (mirage)"); ax.set_ylabel("AUC spatiale (honnête)")
ax.set_xlim(0.5, 0.95); ax.set_ylim(0.5, 0.95)
ax.set_title("Inflation spatiale v2 — tous au-dessus de la diagonale", fontsize=10)
ax.legend(loc="upper left", fontsize=8.5); ax.grid(alpha=0.3)
fig2.tight_layout(); fig2.savefig(FIG / "spatial_vs_random_v2.png", dpi=140)
print(f"[fig] {FIG/'spatial_vs_random_v2.png'}")
