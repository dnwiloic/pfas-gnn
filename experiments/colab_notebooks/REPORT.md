# REPORT — Notebooks Colab autonomes (baselines T1 & T2) — politique ZÉRO DRIVE

> Agent : `colab-notebook-engineer`. Mis à jour 2026-06-19. Graine 42.
> Statut : **2 notebooks régénérés sans Google Drive + smoke-testés VERT sur CPU**.

## Politique de stockage : aucun Google Drive

- **Code + dataset via `git clone` UNIQUEMENT** : le dataset est versionné dans le dépôt
  (`data/CA-PFAS-ASGWS.parquet`, tracké dans git) ; un seul `git clone` ramène `src/` ET
  `data/`. Aucun `drive.mount`, aucun `gdown`, aucune autorisation Drive.
- **Sorties dans l'espace de travail** : `experiments/<id>/` du dépôt cloné (smoke dans un
  sous-dossier distinct). Checkpoint T2 incrémental (`metrics_incremental.json`) après
  chaque modèle.
- Vérifié : `grep -riE "drive|gdown"` sur les cellules de code = **0 appel** (seules
  subsistent des mentions en texte « No Google Drive needed »).

## Livrables

- `notebooks/baseline_t1_colab.ipynb` (29 cellules) — `src.baselines_t1.run_baselines`.
- `notebooks/baseline_t2_colab.ipynb` (33 cellules) — 5 modèles dont **FrequencyClassChain**.
- `tests/test_colab_notebooks.py` — smoke-test CPU des deux notebooks + test anti-Drive.

Autonomie : (1) détection GPU + versions, (2) deps épinglées, (3) `git clone` (code+data),
(4) chargement `DATA_PATH` + contrôle d'intégrité 46338×201 + colonnes clés, garde-fou
anti-code-obsolète (`assert FrequencyClassChain` + `import src.metrics`), (5) toggle
`SMOKE_TEST`, cellules idempotentes.

## Smoke-test CPU — `python3 tests/test_colab_notebooks.py` → ALL GREEN (270 s total)

**T1 — 145.8 s (<200 s)**, 5 métriques finies (CV spatiale) :

| modèle | AUC-ROC | F1 | accuracy | recall | precision |
|---|---|---|---|---|---|
| LR | 0.574 | 0.328 | 0.525 | 0.445 | 0.328 |
| RF | 0.617 | 0.496 | 0.469 | 0.791 | 0.394 |
| XGB | 0.623 | 0.426 | 0.481 | 0.627 | 0.398 |

SHAP non vide ; 4 ablations ; `config.yaml`+`metrics.json` écrits dans le workspace.

**T2 — 118.2 s (<200 s)**, 5 métriques micro+macro ∈[0,1], 5 modèles :

| modèle | macro AUROC | micro F1 | micro acc | micro rec | micro prec |
|---|---|---|---|---|---|
| Prevalence | 0.437 | 0.430 | 0.354 | 0.988 | 0.274 |
| BinaryRelevance | 0.665 | 0.508 | 0.666 | 0.700 | 0.399 |
| Chain | 0.654 | 0.500 | 0.660 | 0.691 | 0.392 |
| Ensemble | 0.669 | 0.506 | 0.644 | 0.740 | 0.384 |
| FreqClassChain | 0.659 | 0.499 | 0.647 | 0.715 | 0.384 |

BR/Chain > plancher (0.437). SMOTE PFNA 0.843→0.881. Checkpoint incrémental OK.
FreqClassChain 4 classes : C1 PFPeA/PFBA/PFOS · C2 PFHxA/PFBS/PFOA · C3 PFHpA/PFPeS ·
C4 (plus rare) PFHxS/PFNA.

## Paramètres à régler (cellule « USER PARAMETERS », aucun champ Drive)

| Paramètre | Notebooks | Rôle |
|---|---|---|
| `SMOKE_TEST` | les 2 | `True` test CPU / `False` run complet |
| `REPO_URL` | les 2 | `https://github.com/dnwiloic/pfas-gnn.git` |
| `GIT_REF` | les 2 | branche ou SHA (défaut `main`) |
| `DATA_PATH` | les 2 | défaut `data/CA-PFAS-ASGWS.parquet` (relatif au dépôt cloné) |
| `TARGET` | T1 | `"T1a"` (défaut) ou `"T1b"` |

Colab : Runtime → GPU **ou High-RAM CPU** → Run all. Aucune autorisation Drive.
Pré-requis : pousser `src/`+`data/`+`notebooks/` à `GIT_REF` avant de lancer (le garde-fou
stoppe si le code cloné est obsolète).

## ⚠️ Workspace Colab éphémère → persistance explicite

`/content/` est effacé à la déconnexion. Chaque notebook finit par une **cellule de
persistance** : (A, recommandé) zip de `experiments/<id>/` + `files.download()` ; (B,
optionnel) `git add/commit/push`. Sans cette étape, les sorties sont perdues. Le checkpoint
T2 incrémental limite la perte à un seul modèle en cas de coupure.

## Estimations run complet

| Tâche | Smoke CPU | Run complet Colab |
|---|---|---|
| T1 (8 plis, 20 trials, ~11 333 puits) | 145.8 s | ~20–45 min |
| T2 (10 labels, 8 plis, 5 modèles, 46 338 lignes) | 118.2 s | ~30–90 min |

## Utilisation effective du GPU (ajout 2026-06-19)

**Réalité technique** : parmi les baselines, **seul XGBoost peut utiliser le GPU**
(`device="cuda"`, xgboost ≥ 2). RandomForest, HistGradientBoosting et LogisticRegression
de scikit-learn sont **CPU-only**.

- `src/config.gpu_available()` (détection cachée) + `xgb_device_params()` → `device="cuda"`
  auto sur Colab GPU, repli CPU `hist` au smoke.
- **T1** : XGBoost passe sur GPU quand un GPU est présent (LR/RF restent CPU).
- **T2** : moteur de base **XGBoost-GPU** (`baselines_t2.default_base_kind()` → `"xgb"` sur
  GPU), branché dans BinaryRelevance / chaînes / FreqClassChain / SMOTE → **tout le run
  lourd T2 exploite le GPU** (imbalance via `scale_pos_weight`).
- **Cellule de vérification GPU** ajoutée aux deux notebooks (après le garde-fou) : petit
  fit XGBoost `device="cuda"` chronométré qui **prouve** l'usage GPU, et rappelle que
  RF/HGB/LR restent CPU.

⚠️ Pour ces baselines d'arbres, une instance **CPU High-RAM** multi-cœurs reste souvent
aussi performante (XGBoost-GPU n'accélère vraiment qu'à grande échelle) et évite que Colab
réclame un runtime GPU inactif. Le GPU sera surtout déterminant pour la **phase GNN**.

## Visualisation des entraînements (ETA)

`src/progress.py` (tqdm `auto`) affiche des **barres de progression avec temps écoulé et
temps restant estimé**, sur les boucles de plis (T1 et T2) et de modèles, étiquetées
`[GPU]`/`[CPU]`. Dégradation silencieuse hors notebook (smoke headless).

Re-smoke après ces ajouts : **ALL GREEN** (T1 131.4 s, T2 105.6 s, zéro Drive résiduel).
