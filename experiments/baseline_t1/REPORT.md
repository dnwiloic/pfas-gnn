# REPORT — Baseline non-graphe T1 (dépassement réglementaire binaire, PFAS CA)

> Agent : `tabular-ml-engineer`. Date : 2026-06-19. Graine : 42 partout.
> **Statut : module SMOKE-TESTÉ VERT sur CPU (160 s). Run complet NON exécuté →
> à lancer sur Colab GPU** (discipline CLAUDE.md §4/§5 respectée : pas de run long local).
> Reproductible : `python3 tests/test_baselines_t1.py` (smoke) ; module : `src/baselines_t1.py`.
> Les chiffres ci-dessous sont SMOKE (500 puits, k=3, 3 trials Optuna) — indicatifs,
> à remplacer par le run complet k=8 sur Colab.

## 0. Résumé exécutif

Trois baselines non-graphe (régression logistique plancher, Random Forest, XGBoost)
évaluées en **double CV** : spatiale par blocs (k=8, référence) + aléatoire groupée par
puits (k=8, diagnostic), sur **T1a** (EPA 2024, garde-fou détection C1). Conditions
C1–C6 respectées par réutilisation du socle `src/`.

Chiffres **smoke** (500 puits, k=3) :

| Modèle | AUC spatial | AUC random | Δ (rd−sp) | Recall sp | Brier sp | Gain top-20% sp |
|--------|:-----------:|:----------:|:---------:|:---------:|:--------:|:---------------:|
| LR (plancher) | 0.574 ± 0.069 | 0.652 | +0.079 | 0.445 | 0.307 | 0.233 |
| RF | 0.617 ± 0.108 | 0.674 | +0.058 | 0.791 | 0.214 | 0.263 |
| **XGBoost** | **0.623** ± 0.069 | 0.654 | **+0.031** | 0.627 | 0.272 | **0.317** |

Comparaisons appariées (Nadeau-Bengio, k=3) : RF vs LR Δ=+0.043 (p=0.43) ; XGB vs LR
Δ=+0.049 (p=0.32) ; RF vs XGB Δ=−0.006 (p=0.93) — **tous non significatifs** (k=3
insuffisant ; k=8 du run complet requis pour la résolution statistique).

**Mur GNN provisoire** : AUC spatiale > 0.62 (à figer après le run complet).

> **Métriques (ajout 2026-06-19).** Les **5 métriques imposées — AUC-ROC, F1, accuracy,
> rappel, précision — sont calculées pour T1** (au seuil OOF) via le module partagé
> `src/metrics.py`, en CV spatiale ET aléatoire, en plus de PR-AUC / balanced-accuracy /
> Brier / gain cumulé. Elles figurent dans `metrics.json` (`*_mean`/`*_std` par modèle).

## 1. Protocole (conditions éval)

- **C1** garde-fou détection : via `targets.build_T1a`.
- **C2** groupage `gm_well_id` : partout, vérifié par `assert_no_group_leak`.
- **C3** CV double : outer k=8 spatial + k=8 aléatoire ; inner k=4 spatial sur le train.
- **C5** target-encoding OOF (`_KFoldTargetEncoder`) pour `dwr_basin`/`sgma_subbasin`.
- **C6** `gm_dataset_name` exclu des features (`config.feature_columns`).
- Déséquilibre : `class_weight`/`scale_pos_weight` dans l'espace Optuna (auto-sélection ;
  T1a quasi-équilibré ~0.445 → pas de pondération forcée).
- Seuil τ* : max-F1 sur probabilités OOF du train interne, **jamais sur le test**.
- Optuna TPE (20 trials en complet, 3 en smoke), graine 42.

## 2. Scores du run complet (à remplir sur Colab)

[PLACEHOLDER — exécuter `run_baselines(smoke=False)` (`python -m src.baselines_t1` ou
notebook) sur Colab GPU : XGBoost `tree_method="hist"` GPU + RF `n_jobs=-1`. Estimation :
45–90 min GPU. Le tableau final rapportera, par modèle et par schéma de CV
(spatial / aléatoire + Δ), les **5 métriques** : AUC-ROC, F1, accuracy, rappel, précision.]

## 3. Importance des features (SHAP, XGB, smoke pli 0) + contrôle de plausibilité

| Rang | Feature | SHAP | Famille | Vigilance |
|------|---------|:----:|---------|-----------|
| 1 | `dwr_basin__enc` | 0.979 | Admin/hydrogéo | ⚠️ proxy géo fort (η≈0.50 vs T1a) — confondeur design |
| 2 | `dist_geotracker_km` | 0.265 | Geotracker | OK mécaniste (distance source PFAS) |
| 3 | `n_geotracker_within_50km` | 0.193 | Geotracker | OK (densité sources) |
| 4 | `sgma_subbasin_name__enc` | 0.180 | Admin/hydrogéo | ⚠️ proxy géo (η≈0.50, C5) |
| 5 | `soil_texture_class__enc` | 0.145 | Sol SSURGO | OK (perméabilité) |
| 6 | `gldas_dist_km` | 0.118 | Climat | artefact possible (proxy région) |
| 7 | `regional_board=CENTRAL VALLEY` | 0.113 | Admin | ⚠️ proxy régional |
| 8 | `soil_sand_pct` | 0.109 | Sol SSURGO | OK (drainage) |
| 9 | `cocontam_tce` | 0.099 | Cocontaminant | proxy panel labo possible — audit SHAP spatial |
| 10 | `aqs_no2_ppb` | 0.098 | Air AQS | proxy urbanité/industrie |

**Alerte clé** : `dwr_basin__enc` domine (SHAP 0.979). Le target-encoding est anti-fuite
de cible (C5 OK) mais reste un **confondeur géographique** ; vérifier sur le run complet
si son importance **s'effondre en CV spatiale vs aléatoire** (→ artefact de design) ou
**tient** (→ valeur mécaniste de géologie de bassin).

## 4. Ablations (RF, smoke, sans Optuna)

| Configuration | AUC sp | AUC rd | Δ |
|---|:---:|:---:|:---:|
| no_loc / cocontam=all (réf.) | 0.700 | 0.669 | −0.031 |
| with_loc / all | 0.658 | 0.671 | +0.013 |
| no_loc / cocontam=core | 0.561 | 0.649 | +0.088 |
| no_loc / cocontam=none | 0.648 | 0.673 | +0.025 |

- **lat/lon dégrade** l'AUC spatiale (−4 pts) → confirme : **pas de lat/lon en features
  de nœuds GNN**, les passer par les arêtes k-NN distanciées (C4).
- cocontam=core seul réduit fortement l'AUC sp ET amplifie l'artefact (Δ +0.088) →
  à arbitrer avec l'hydro sur le périmètre cocontaminant.

## 5. Positionnement littérature (Dong et al. 2024)

Dong et al. rapportent AUC ~0.78–0.85 en **split aléatoire par ligne**. Nos AUC random
(0.65–0.67, smoke, split **par puits**) sont plus basses, et nos AUC **spatiales**
(0.57–0.62) mesurent la généralisation **géographique** réelle — non rapportée dans un
protocole non spatial. À reconfirmer au run complet.

## 6. Recommandations pour les GNN

1. Battre le **mur spatial** (AUC sp à figer au run complet, ~0.62), pas l'AUC aléatoire.
2. Rapporter systématiquement le **triplet (random, spatial, Δ)** ; un GNN n'aide que
   s'il monte le spatial **sans gonfler Δ**.
3. lat/lon **uniquement** via arêtes k-NN plafonnées ~1–2 km, coupées aux frontières de
   bloc (C4) ; ne pas mettre lat/lon en features de nœud.
4. Auditer SHAP `dwr_basin`/cocontaminants en spatial vs aléatoire (artefact vs signal).

## 7. Artefacts

- `src/baselines_t1.py` — module importable (RF, XGBoost, LR ; Optuna ; seuil OOF ; SHAP ; ablations).
- `tests/test_baselines_t1.py` — smoke test (VERT, 160 s CPU).
- `experiments/baseline_t1_smoke/` — artefacts smoke (`config.yaml`, `metrics.json`, `feature_importance.csv`).
- **Run complet Colab** → écrira `experiments/baseline_t1/{config.yaml,metrics.json}` + complétera ce REPORT (§2).
