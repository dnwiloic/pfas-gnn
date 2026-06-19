# REPORT — Notebooks Colab autonomes (baselines T1 & T2)

> Agent : `colab-notebook-engineer`. Date : 2026-06-19. Graine 42.
> Statut : **2 notebooks livrés + smoke-testés VERT sur CPU** (CLAUDE.md §4/§5).

## Livrables

- `notebooks/baseline_t1_colab.ipynb` (27 cellules) — orchestre `src.baselines_t1.run_baselines`.
- `notebooks/baseline_t2_colab.ipynb` (33 cellules) — orchestre le pipeline T2 (5 modèles
  dont **FrequencyClassChain**).
- `tests/test_colab_notebooks.py` — driver de smoke-test CPU des deux notebooks.

Chaque notebook est **autonome** : (1) détection GPU + versions, (2) install deps épinglées
(xgboost/optuna/shap/imbalanced-learn ; pas de PyG, baselines non-graphe), (3) bootstrap
`src/` (git OU drive), (4) montage Drive + dataset paramétré + contrôle d'intégrité
(46338×201), (5) toggle `SMOKE_TEST` en tête, (6) cellules idempotentes + checkpoints Drive.

## Smoke-test CPU (2026-06-19) — `python3 tests/test_colab_notebooks.py` → ALL GREEN

| Notebook | Durée | 5 métriques | Détails |
|---|---|---|---|
| baseline_t1 | **153.9 s** (<200 s) | ✅ finies (LR/RF/XGB, spatial+random) | AUC sp LR 0.574 / RF 0.617 / XGB 0.623 ; SHAP + 4 ablations OK ; artefacts écrits |
| baseline_t2 | **110.6 s** (<200 s) | ✅ micro+macro ∈[0,1] (5 modèles) | par-label f1/precision/recall/accuracy ; checkpoint incrémental ; BR 0.665 > floor 0.437 ; SMOTE PFNA 0.843→0.881 ; pseudo + Wilcoxon OK |

FrequencyClassChain — 4 classes de fréquence (sous-échantillon smoke) :
C1 (moins rare) PFPeA/PFBA/PFOS · C2 PFHxA/PFBS/PFOA · C3 PFHpA/PFPeS · C4 (plus rare) PFHxS/PFNA.

## Paramètres à régler par l'utilisateur (cellule 1 « USER PARAMETERS »)

| Paramètre | Notebooks | Rôle |
|---|---|---|
| `SMOKE_TEST` | les 2 | `True` test CPU rapide / `False` run complet GPU |
| `BOOTSTRAP` | les 2 | `"git"` (clone GitHub) ou `"drive"` (copie depuis Drive) |
| `REPO_URL` | les 2 | `https://github.com/dnwiloic/pfas-gnn.git` (ou fork) |
| `GIT_REF` | les 2 | branche ou SHA de commit (défaut `main`) |
| `DRIVE_PROJECT_DIR` | les 2 | ex. `/content/drive/MyDrive/pfas-gnn` |
| `DRIVE_DATA_PATH` | les 2 | chemin Drive du parquet (Option A) |
| `GDRIVE_FILE_ID` | les 2 | ID gdown (Option B ; vide → Option A) |
| `TARGET` | T1 | `"T1a"` (défaut) ou `"T1b"` |

Colab : Runtime → GPU (ou High-RAM CPU) → Run all → autoriser Drive.

## ⚠️ Pré-requis CRITIQUE : code à jour sur Colab

Le code des baselines (`src/baselines_t1.py`, `baselines_t2.py`, `metrics.py`,
FrequencyClassChain, 5 métriques) **n'est pas encore poussé** sur le remote. Avant
`BOOTSTRAP="git"` :
```bash
git add src/ tests/ experiments/ notebooks/
git commit -m "feat: baselines T1/T2 + FreqClassChain + 5 métriques + notebooks Colab"
git push origin main
```
Alternative `BOOTSTRAP="drive"` : copier le `src/` local à jour dans `DRIVE_PROJECT_DIR/src/`.
Les deux notebooks contiennent un **garde-fou** après bootstrap
(`assert hasattr(src.baselines_t2, "FrequencyClassChain")` + `from src.metrics import REQUIRED`)
qui s'arrête avec un message explicite si le code Colab est obsolète.

## Estimations run complet GPU/CPU

| Tâche | Smoke CPU | Run complet Colab |
|---|---|---|
| T1 (8 plis, 20 trials Optuna, 11 333 puits) | 153.9 s | ~20–45 min |
| T2 (10 labels, 8 plis, 5 modèles, 46 338 lignes) | 110.6 s | ~30–90 min |

**Vigilance** : sklearn/XGBoost tournent sur **CPU** même en runtime GPU → une instance
**High-RAM CPU multi-cœurs** peut être préférable pour ces baselines (surtout T2, chaînes
coûteuses peu parallélisées). Checkpoint T2 incrémental (`metrics_incremental.json`) écrit
après chaque modèle → reprise possible après déconnexion.
