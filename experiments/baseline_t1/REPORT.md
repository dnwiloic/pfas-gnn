# REPORT — Baseline T1 (dépassement réglementaire binaire, PFAS CA)

> Graine 42. Cible T1a = EPA 2024 NPDWR (PFOA>4 ∨ PFOS>4 ∨ PFNA>10 ∨ PFHxS>10 ∨
> GenX>10 ∨ Hazard Index>1), prévalence ~45,7 %.
>
> **Statut : DEUX régimes documentés.** (1) Point **aléatoire/optimiste** repris d'un
> notebook antérieur (`ca-pfas-ml/.../01_binary_classification_epa2024`, 2026-06-20) ≈
> protocole Dong et al. (§1) ; (2) Point **spatial** = notre référence rigoureuse,
> calculée en local (§1bis, `run_spatial_t1.py`). ⚠️ Le 0,97 du §1 est le **plafond
> littérature** (split aléatoire, `gm_dataset_name`, pas de garde-fou) ; **la vraie
> référence est l'AUC spatiale ~0,60** (§1bis, §5, §6).

## 0. Résumé exécutif

- **Protocole standard (split aléatoire 80/20, 86 features, Optuna)** : RF **AUC 0,974**,
  XGBoost **AUC 0,971** — niveau de la littérature (Dong et al. ~0,97-0,99).
- Ces scores sont **gonflés** par trois facteurs vs notre protocole strict (§5) : split
  **aléatoire** (pas de groupage puits, pas de blocs spatiaux), inclusion du confondeur de
  design **`gm_dataset_name`** (rang 1 en importance XGB), **absence de garde-fou de
  détection** (C1).
- **Référence spatiale honnête** (notre pipeline, run complet k=8, sans Optuna —
  `run_spatial_t1.py`) : **RF AUC spatial 0,601 ± 0,056 · XGB 0,588 ± 0,068**. En CV
  **aléatoire** groupée (même blocklist stricte) : ~0,90. **L'écart 0,90 → 0,60
  (Δ ≈ +0,30) est l'inflation spatiale**, cœur de la contribution. **Le mur pour les
  GNN = AUC SPATIALE ~0,60**, pas le 0,97 (ni même le 0,90 aléatoire).

## 1. Modèles × les 5 métriques (split aléatoire 80/20, après Optuna)

Jeu de test = 9 268 puits (20 %, stratifié aléatoire). prévalence test 45,7 %.

| modèle | AUC‑ROC | F1 | accuracy | rappel | précision | AP | bal.acc |
|---|---|---|---|---|---|---|---|
| **Random Forest (Optuna)** | **0,974** | 0,920 | 0,927 | 0,915 | 0,925 | 0,969 | 0,926 |
| **XGBoost (Optuna)** | 0,971 | 0,912 | 0,919 | 0,916 | 0,909 | 0,967 | 0,919 |
| Random Forest (défaut) | 0,966 | 0,900 | 0,909 | 0,893 | 0,906 | 0,961 | 0,908 |
| XGBoost (défaut) | 0,967 | 0,902 | 0,910 | 0,906 | 0,899 | 0,962 | 0,910 |

**CV 5-fold (aléatoire, params défaut)** : RF AUC 0,962 ± 0,001 · F1 0,891 ± 0,004 ·
rappel 0,887 ± 0,002 ; XGB AUC 0,965 ± 0,001 · F1 0,898 ± 0,002 · rappel 0,900 ± 0,004.

Seuil optimal OOF ≈ 0,49 (RF) / 0,48 (XGB) — proche du défaut 0,50 (cible quasi-équilibrée).

## 1bis. Référence SPATIALE — notre pipeline strict (le vrai mur, run complet k=8)

`experiments/baseline_t1/run_spatial_t1.py` — données complètes (46 338, prév. 44,5 %
avec garde-fou C1), CV **spatiale par blocs** (référence) + **aléatoire groupée** (Δ),
groupé `gm_well_id`, blocklist stricte (**`gm_dataset_name` exclu**, C6), seuil OOF, sans
Optuna. Les 5 métriques (seuil OOF F1-optimal → rappel élevé / accuracy basse, normal).

| modèle | AUC **sp** | F1 sp | acc sp | rappel sp | préc sp | PR-AUC sp | AUC **rd** | Δ AUC |
|---|---|---|---|---|---|---|---|---|
| Random Forest | **0,601** ± 0,056 | 0,554 | 0,446 | 0,968 | 0,397 | 0,490 | 0,898 | **+0,297** |
| XGBoost | **0,588** ± 0,068 | 0,520 | 0,510 | 0,750 | 0,411 | 0,474 | 0,900 | **+0,313** |

- **AUC spatiale ~0,59-0,60 = à peine au-dessus du hasard** : en généralisation
  géographique stricte, prédire le dépassement réglementaire à partir du contexte seul
  est **difficile** (et plus dur que les meilleurs labels de T2 spatial, ~0,68).
- Notre **AUC aléatoire ~0,90 < le 0,97 externe** : l'écart (~0,07) vient surtout de
  `gm_dataset_name` (gardé dans le travail externe, exclu ici en C6) + Optuna + jeu de
  features. ⇒ même en régime aléatoire, notre protocole est moins « gonflé ».

## 2. Lecture opérationnelle

- **Gain cumulé** : en testant les **25 % puits les plus à risque**, on détecte **54 %**
  des puits réellement > MCL (RF et XGB).
- **Calibration** : probabilités bien calibrées (courbe proche de la diagonale).

## 3. Importance des features (et alerte de fuite)

Top features communes RF∩XGB : `n_geotracker_within_50km`, **`gm_dataset_name`** (rang 1
XGB), `n_geotracker_within_10km`, `gm_well_category`, `cocontam_pce`…

⚠️ **`gm_dataset_name` en tête = fuite de design** : le programme d'échantillonnage (p. ex.
WB_CLEANUP cible des sites déjà pollués) est fortement corrélé à la cible. Notre protocole
l'**exclut** (C6). Sa présence ici explique une part importante du 0,97.

## 4. Dataset & protocole de ce baseline (externe)

- 46 338 échantillons × **86 features** (concentrations/_detected/label_ exclus ; **lat/lon
  + county/regional_board/dwr_region exclus** « localisation pure » ; `gm_dataset_name`
  **conservé**). Imputation médiane, OrdinalEncoder pour les catégorielles.
- **Split aléatoire stratifié 80/20** (train 37 070 / test 9 268). Optuna : RF 40 trials,
  XGB 60 trials. Modèles sauvés (`models/*.pkl`).

## 5. ⚠️ Caveats méthodologiques (pourquoi 0,97 ≠ notre référence)

| Facteur | Ce baseline (externe) | Notre protocole strict |
|---|---|---|
| Découpage | **aléatoire** 80/20 + 5-fold aléatoire | **CV spatiale par blocs** + groupé `gm_well_id` (C2/C3) |
| Pseudo-réplicats | un puits peut être dans train ET test | interdits (groupage puits) |
| `gm_dataset_name` | **conservé** (rang 1) | **exclu** (confondeur de design, C6) |
| Garde-fou détection | **absent** (prév. 45,7 %) | **appliqué** (C1, prév. 44,5 %) |

⇒ ces 0,97 sont le **régime aléatoire/optimiste** (≈ Dong et al.). Notre référence
**spatiale** (smoke ~0,62) mesure la généralisation géographique réelle. **Toujours
comparer un GNN à la référence SPATIALE**, jamais à ce 0,97.

## 6. Triplet (aléatoire, spatial, Δ) — synthèse T1

| régime | AUC | source |
|---|---|---|
| Aléatoire externe (Dong-like, +`gm_dataset_name`, Optuna) | **~0,97** | notebook externe (§1) |
| Aléatoire — notre protocole strict | **~0,90** | `run_spatial_t1.py` (§1bis) |
| **Spatial — notre référence** | **~0,60** | `run_spatial_t1.py` (§1bis) |

⇒ **Le mur que les GNN doivent battre = AUC spatiale ~0,60** (RF 0,601 / XGB 0,588), avec
le triplet rapporté à chaque fois. Cohérent avec T2 (spatial ~0,68 vs aléatoire ~0,90).
Point spatial T1 **obtenu en local** (~20 min CPU) — plus besoin du run Colab long.

### Provenance
Notebook externe `ca-pfas-ml/notebooks/01_binary_classification_epa2024` (PDF 2026-06-20).
Chiffres recopiés fidèlement ; figures/artefacts originaux dans ce dépôt-là
(`models/{rf,xgb}_binary_epa2024.pkl`, `reports/figures/`).
