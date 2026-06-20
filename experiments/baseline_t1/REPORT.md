# REPORT — Baseline T1 (dépassement réglementaire binaire, PFAS CA)

> Graine 42. Cible T1a = EPA 2024 NPDWR (PFOA>4 ∨ PFOS>4 ∨ PFNA>10 ∨ PFHxS>10 ∨
> GenX>10 ∨ Hazard Index>1), prévalence ~45,7 %.
>
> **Statut : métriques de baseline adoptées d'un notebook ANTÉRIEUR**
> (`ca-pfas-ml/notebooks/01_binary_classification_epa2024`, 2026-06-20), car le run
> rigoureux (CV spatiale) sur Colab est trop long. **⚠️ Ces chiffres relèvent du
> RÉGIME ALÉATOIRE/OPTIMISTE (≈ protocole Dong et al.), PAS de notre référence
> spatiale** — voir §5 (3 sources d'inflation). À traiter comme le **plafond
> littérature**, à contraster avec la CV spatiale (notre pipeline).

## 0. Résumé exécutif

- **Protocole standard (split aléatoire 80/20, 86 features, Optuna)** : RF **AUC 0,974**,
  XGBoost **AUC 0,971** — niveau de la littérature (Dong et al. ~0,97-0,99).
- Ces scores sont **gonflés** par trois facteurs vs notre protocole strict (§5) : split
  **aléatoire** (pas de groupage puits, pas de blocs spatiaux), inclusion du confondeur de
  design **`gm_dataset_name`** (rang 1 en importance XGB), **absence de garde-fou de
  détection** (C1).
- **Référence spatiale honnête** (notre pipeline `src/baselines_t1.py`, CV par blocs) :
  smoke ≈ **0,62 AUC**, run complet à obtenir. **L'écart 0,97 → ~0,62 = l'inflation
  spatiale + design**, cœur de notre contribution. **Le mur pour les GNN reste l'AUC
  spatiale (~0,62), pas ce 0,97.**

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

## 6. Réconciliation & étape suivante

- Pour un positionnement cohérent, rapporter le **triplet (aléatoire, spatial, Δ)** comme
  pour T2. Ce baseline fournit le point **aléatoire** de T1 (~0,97, ≈ littérature) ; il
  manque le point **spatial** (notre `run_baselines(smoke=False)` — long sur Colab).
- Options pour le point spatial T1 : (a) laisser finir le run Colab ; (b) lancer chez nous
  un run **spatial réduit** (sans Optuna, k=8) — faisable en CPU multi-cœurs ; (c) extrapoler
  depuis le smoke (~0,62) en attendant.

### Provenance
Notebook externe `ca-pfas-ml/notebooks/01_binary_classification_epa2024` (PDF 2026-06-20).
Chiffres recopiés fidèlement ; figures/artefacts originaux dans ce dépôt-là
(`models/{rf,xgb}_binary_epa2024.pkl`, `reports/figures/`).
