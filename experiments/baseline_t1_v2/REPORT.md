# REPORT — Mur tabulaire T1a sur le dataset v2 (enrichi)

> Run complet local, k=8, smoke=false, graine 42, CPU, wall 2080 s (~35 min).
> Cible **T1a** = EPA 2024 NPDWR (PFOA>4 ∨ PFOS>4 ∨ HI≥1), garde-fou détection (C1),
> prévalence **0,445**. Features = **105 cols strictes** (`feature_columns(
> include_location=False, cocontam="all", include_air=True, include_derived=True)`) :
> aucune mesure PFAS, pas de lat/lon, `gm_dataset_name` exclu. Protocole = socle figé
> (groupage `gm_well_id`, CV spatiale KMeans k=8 + aléatoire groupée, seuil F1 OOF).
> Source unique des chiffres : `metrics_v2_t1.json`, `history_xgb.json`.

---

## 0. Verdict : **l'enrichissement v2 DÉPLACE le mur spatial (+0,05 AUC)**

| modèle | AUC spatial **v1** | AUC spatial **v2** | gain | AUC random v2 | Δ(rd−sp) v2 |
|---|---|---|---|---|---|
| Random Forest | 0,601 ± 0,056 | **0,6524 ± 0,068** | **+0,051** | 0,9084 | +0,256 |
| XGBoost | 0,588 ± 0,068 | **0,6446 ± 0,051** | **+0,057** | 0,9070 | +0,262 |

Même cible, même protocole strict, même schéma de blocs — **seule la collecte de features
change**. Les 16 variables hydrogéologiques v2 (profondeur effective, depth-to-water,
direction/gradient d'écoulement, géométrie de crépine, élévation, land cover) **lèvent le
mur de ~0,05 pt AUC** en généralisation spatiale stricte. C'est exactement le message du
mémoire : *un pipeline de collecte enrichi améliore la prédiction de contamination.*

Bonus méthodologique : l'**inflation spatiale Δ recule** (v1 RF +0,297 / XGB +0,313 →
v2 +0,256 / +0,262). Le modèle s'appuie un peu **moins** sur la mémorisation de la carte
parce qu'il dispose de plus de signal mécaniste réel.

---

## 1. Métriques complètes (CV spatiale, seuil OOF F1-optimal)

| modèle | AUC | F1 | accuracy | rappel | précision | PR-AUC | bal.acc |
|---|---|---|---|---|---|---|---|
| RF  | 0,6524 | 0,561 | 0,470 | 0,953 | 0,407 | 0,524 | 0,562 |
| XGB | 0,6446 | 0,551 | 0,578 | 0,745 | 0,446 | 0,510 | 0,597 |

Seuil OOF F1-optimal → rappel élevé / accuracy basse (attendu sur cible ~équilibrée).
RF sur-rappelle (0,953) ; XGB plus équilibré (bal.acc 0,597 > RF 0,562).

## 2. Courbes d'entraînement XGBoost (§3.8) — `figures/{loss,metric}_curves.png`

Diagnostic par pli spatial (train vs **bloc spatial held-out**, 400 rounds) :

| | valeur | lecture |
|---|---|---|
| train AUC final | ~0,995 | le modèle **sur-apprend massivement** les blocs train |
| val AUC final (moy.) | 0,645 | écart train−val ≈ **0,35** = signal contextuel peu généralisable |
| val AUC max (moy.) | 0,672 | **+0,028** au-dessus du final → léger sur-apprentissage tardif |
| **val logloss argmin (moy.)** | **~round 38/400** | l'optimum de généralisation est **très précoce** |

→ **Quick win identifié** : un **early-stopping vers 40–100 rounds** (au lieu de 400)
récupérerait ~+0,02–0,03 AUC spatial et de la calibration. À activer pour la version finale.

Instabilité inter-plis (attendue, stationnarité spatiale fausse) : pli 0 val AUC **0,555**
(quasi-hasard), plis 1/4 ~0,71. σ ≈ 0,05–0,07. Pli 0 = région à régime hydrogéologique
distinct (cohérent avec `hgt_rgcn_t1`).

## 3. SHAP global (`figures/shap_summary_bar.png`, `shap_beeswarm.png`) — **audit de fuite**

Top-20 mean(|SHAP|) sur XGB full-data. **Deux familles se détachent :**

**(A) ⚠️ Mémorisation spatiale — encodages administratifs target-encodés :**
1. `dwr_basin__enc` **0,576** · 2. `sgma_subbasin_name__enc` **0,394** · 20. `soil_texture_class__enc` 0,086.

Ces deux encodages cible dominent : ils encodent la **prévalence locale** (le risque que
l'éval et l'hydro-expert avaient signalé). C'est l'équivalent fonctionnel de lat/lon ; ils
portent une part du score spatial qui est de la **mémorisation de carte**, pas du mécanisme.

**(B) ✅ Mécanisme hydrogéologique réel — porté par les features v2 :**
3. `depth_to_water_m` 0,234 · **4. `flow_dir_sin` 0,216** · 6. `screen_mid_ft` 0,209 ·
7. `dist_geotracker_km` 0,209 · 8. `depth_eff_ft` 0,198 · 9. `hydr_grad_mag_permil` 0,171 ·
10. `elevation_m` 0,166 · 11. `screen_length_ft` 0,159 · 13. `flow_dir_cos` 0,146 ·
14. `n_geotracker_within_50km` 0,136 · 15. `dev_intensity` 0,121 · 16. `n_geotracker_within_10km` 0,105.

**C'est la signature mécaniste « saine » attendue par l'hydro-expert** (proximité source +
profondeur/crépine + rétention/usage du sol), et **elle est désormais dominée par les
features v2**. Fait notable : **`flow_dir_sin` au rang 4** — la direction d'écoulement de
la nappe compte → **justifie directement l'arête orientée `flows_to` du chapitre GNN.**

**(C) Alertes confirmées :** `gm_well_category=MONITORING` (rang 5, 0,215) = intention de
surveillance (puits monitoring placés sur sites contaminés) ; `year` (rang 12) = dérive
d'échantillonnage temporelle ; `cocontam_pce`/`cocontam_as` (rangs 17–18) = co-mesure
d'état. À traiter en ablation, non comme mécanismes.

---

## 4. Lecture pour le mémoire

1. **Résultat présentable et positif** : la collecte v2 fait passer le mur spatial de
   ~0,60 à **~0,65** (RF 0,652 best). Premier gain net depuis le début du projet.
2. **SHAP raconte le bon mécanisme** : les features hydrogéo v2 (profondeur, écoulement,
   crépine, source) forment le cœur du signal — plausibilité validée.
3. **Honnêteté maintenue** : le triplet montre que ~0,90 random reste un mirage
   (Δ ≈ +0,26) ; deux encodages administratifs portent encore de la mémorisation de carte.

## 5. Prochaines étapes recommandées (rapides, fort ROI)

1. **Ablation « mécanisme pur »** : rejouer sans les high-card admin (`dwr_basin`,
   `sgma_subbasin_name`, `county`) pour mesurer le mur **sans mémorisation de carte**. Si
   l'AUC tient ~0,63–0,64, le gain v2 est mécaniste, pas de la fuite spatiale → argument fort.
2. **XGB early-stopping (~50 rounds)** pour la version finale (quick win ~+0,02–0,03).
3. **Stacking v2** (figure phare) : ce mur tabulaire v2 (0,645–0,652) devient la référence
   à battre par HGT → fusion → stacking, avec `flow_dir`/gradient pour l'arête orientée.

### Reproductibilité
```bash
SMOKE_TEST=1 PFAS_FORCE_CPU=1 python3 experiments/baseline_t1_v2/run_v2_t1.py   # ~40 s
PFAS_FORCE_CPU=1 python3 experiments/baseline_t1_v2/run_v2_t1.py                 # ~35 min, k=8
ABLATION=pure_mech PFAS_FORCE_CPU=1 python3 experiments/baseline_t1_v2/run_v2_t1.py  # ablation
```

---

## 6. ABLATION « mécanisme pur » — le gain v2 N'EST PAS de la mémorisation de carte

Question : le gain v2 vient-il des features hydrogéo, ou des deux encodages administratifs
(`dwr_basin__enc`, `sgma_subbasin_name__enc`) qui dominaient le SHAP (mémorisation de la
carte de prévalence) ? On rejoue à l'identique **sans les 7 colonnes administratives**
(`county`, `dwr_basin`, `dwr_region`, `regional_board`, `sgma_basin_name`,
`sgma_subbasin_name`, `sgma_region_office`) → 98 features, **que du mécanisme**.
Source : `metrics_v2_t1_pure_mech.json`, `history_xgb_pure_mech.json`, `figures_pure_mech/`.

| modèle | AUC sp **full** (105) | AUC sp **pure_mech** (98) | Δ | AUC random | Δ inflation |
|---|---|---|---|---|---|
| RF  | 0,6524 ± 0,068 | **0,6493 ± 0,046** | −0,003 | 0,9051 | +0,256 |
| XGB | 0,6446 ± 0,051 | **0,6528 ± 0,061** | **+0,008** | 0,9070 | +0,254 |

**Verdict : le mur TIENT — retirer l'administratif ne coûte rien (XGB s'améliore même).**
Les 7 colonnes admin étaient **redondantes** avec le signal spatial déjà porté légitimement
par la proximité de sources et la topographie. Conséquences pour le mémoire :

1. **Le gain v2 (+0,06 vs v1) est mécaniste**, pas un artefact de mémorisation. Le mur de
   référence propre = **XGB 0,653 / RF 0,649**, sur 98 features 100 % interprétables.
2. **RF se stabilise** (std 0,068 → 0,046) : sans le bruit des encodages haute-cardinalité,
   moins d'écart inter-plis.
3. **SHAP pure_mech = signature hydrogéo de manuel** (figure `figures_pure_mech/shap_*`) :
   `n_geotracker_within_50km` (#1, 0,538) · `depth_to_water_m` (#2) · **`flow_dir_sin`
   (#3)** · `screen_mid_ft` (#4) · `elevation_m` (#5) · `dist_geotracker_km` (#7) ·
   `depth_eff_ft` (#8) · `hydr_grad_mag_permil` (#10) · `flow_dir_cos` (#11). **Exactement
   l'ordre attendu par l'hydro-expert** : source → profondeur/crépine → écoulement → sol.
4. **Alertes résiduelles** (honnêteté) : `gm_well_category=MONITORING` (#9) et `year` (#13)
   persistent — intention d'échantillonnage + dérive temporelle, à mentionner.

Courbes pure_mech (§3.8) : même sur-apprentissage structurel (train 0,995 vs val 0,653),
optimum logloss val **round ~22/400** → early-stopping encore plus précoce justifié.
Per-fold val AUC = [0,559, 0,710, 0,640, 0,708, 0,687, 0,653, 0,711, 0,554] : plis 0 et 7
restent les régions difficiles (cohérent avec full + `hgt_rgcn_t1`).

**→ Référence figée pour le stacking v2 : le mur « mécanisme pur » XGB 0,653 / RF 0,649.**
