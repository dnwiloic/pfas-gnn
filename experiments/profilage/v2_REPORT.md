# Profilage — CA-PFAS-ASGWS_v2.parquet

> Rapport de profilage en mode prédictif strict. Aucune connaissance préalable du jeu
> de données : tout fait ci-dessous est établi par analyse directe et reproductible.
> Graine fixée : `np.random.seed(42)`. Source : `data/CA-PFAS-ASGWS_v2.parquet`.
> pandas 2.1.4 · numpy 1.26.4 · sklearn 1.8.0 · python 3 (`/usr/bin/python3`).

---

## 1. Structure générale

- **Dimensions** : **46 338 lignes × 217 colonnes**.
- **Granularité** : ce n'est PAS un jeu « un puits = une ligne ». C'est un **panel
  longitudinal** : `gm_well_id` compte **11 333 puits uniques** mesurés à **2 069 dates**
  distinctes (`collection_date`, du **2016-04-01 au 2026-01-26**).
  - Lignes par puits : moyenne 4.09, médiane 1, max **139**. 50 % des puits n'ont
    qu'une seule mesure ; quelques puits sont sur-échantillonnés.
  - 0 doublon `(gm_well_id, collection_date)`, 0 ligne entièrement dupliquée.
- **Identifiants** : `gm_well_id` (object, clé puits, 11 333 modalités),
  `collection_date` (datetime, clé temporelle). Ensemble = clé primaire de la ligne.
- **Types** : 1 datetime, ~9 object (catégorielles), 31 bool (`*_detected`), ~30 int64
  (labels + compteurs + flags), le reste float64.
- **Sources** (`gm_dataset_name`, 6 modalités) :
  DDW 36 993 · WB_CLEANUP 5 860 · WRD 1 042 · GAMA_USGS 896 · LOCALGW 871 · USGS_NWIS 676.
- **Découpage administratif** : `county` (58), `regional_board` (9), `dwr_region` (10),
  `dwr_basin` (239), `sgma_basin_name` (162), `sgma_subbasin_name` (237),
  `sgma_region_office` (4). `gm_well_category` (5) : MUNICIPAL 38 067, MONITORING 6 921,
  DOMESTIC 721, WATER SUPPLY OTHER 559, IRRIGATION/INDUSTRIAL 70.

---

## 2. Valeurs manquantes

### Colonnes inutilisables (constantes ou > 85 % manquantes)
| colonne | %manquant | nuniq | verdict |
|---|---|---|---|
| `cocontam_xylenes` | 99.8 % | 1 | **drop** (constante quasi vide) |
| `cocontam_tmb124` / `cocontam_dce12c` / `cocontam_btbzt` | 98.7 % | 1 | **drop** |
| `soil_gradation_uniformity` / `_curvature` | 98.2 % | 36 | **drop** |
| `soil_silt_coarse_pct` / `soil_silt_fine_pct` | 95.1 % | 60-64 | **drop** |
| `cocontam_no3n` | 94.9 % | 135 | **drop** (couverture 5 %) |
| `well_depth_ft` | 94.5 % | 858 | drop (remplacé par `depth_eff_ft`/`depth_to_water_m`) |
| `pfas_class_assignment` | 0 % | **1** | **drop** (constante, JSON statique) |
| `label_PFEESA` / `label_PFMBA` / `label_PFMPA` | 0 % | **1** | cibles dégénérées (jamais 1) |

### Patterns de manquant (mécanisme)
Le manquant est **MAR**, principalement piloté par la **source** et la **catégorie de puits** :
- Complétude du panel PFAS étendu (ex. `FTS_4_2_ngL`) par source :
  GAMA_USGS 0 %, USGS_NWIS 0 %, WB_CLEANUP 14 %, DDW 49 %, LOCALGW 57 %, **WRD 100 %**.
  → certaines sources ne mesurent que le panel « cœur » (PFOA/PFOS/PFBS…), d'autres le
  panel étendu. **Ce n'est pas MCAR.**
- `well_depth_ft` manquant par catégorie : DOMESTIC 6 % vs **MUNICIPAL 98 %**,
  MONITORING 88 %. → la profondeur n'est renseignée que pour certains types de puits.
- `cocontam_tds` manquant : 0 % (USGS) à 68 % (LOCALGW).

**Conséquence méthodo** : la source (`gm_dataset_name`) et la catégorie portent de
l'information sur le mécanisme de manquant → conserver comme features + ajouter des
**indicateurs de manquant**. Deux indicateurs sont déjà fournis et exacts :
`depth_missing == depth_eff_ft.isna()` (match 100 %), `topo_missing == elevation_m.isna()`
(match 100 %).

---

## 3. Distributions (faits saillants)

- **`sum_pfas_ngL`** (somme PFAS, voir §4) : min 0, médiane 32.3, moyenne 174,
  q90 142, q99 4 217, max 5 723 ng/L. Très **fortement asymétrique à droite**
  → utiliser `log1p` pour toute feature dérivée (mais cette colonne est une CIBLE/fuite).
- **Profondeurs** : `depth_eff_ft` médiane 255 ft (4.6–2 500), 49 % manquant ;
  `depth_to_water_m` médiane 25.7 m (0–199), **0 % manquant** (imputé/modélisé).
- **`hydr_grad_mag_permil`** : médiane 3.5, max 43 900 ‰ → **valeurs extrêmes**,
  capper / log avant usage.
- **Direction d'écoulement** encodée en sin/cos (`flow_dir_sin`, `flow_dir_cos` ∈ [-1,1])
  — bon encodage circulaire, prêt à l'emploi.
- **Catégorielles à forte cardinalité** : `soil_texture_class` (102), `dwr_basin` (239),
  `sgma_subbasin_name` (237) → encodage cible-libre (fréquence / regroupement), pas de
  one-hot brut.

---

## 4. Identification des cibles

### Colonnes PFAS repérées (sémantique + vérif chiffrée)
- **31 concentrations** `*_ngL` (ex. `PFOA_ngL`, `PFOS_ngL`, `PFBS_ngL`…) : mesures
  individuelles en ng/L.
- **31 flags** `*_detected` (bool) : détection analytique.
- **31 labels** `label_*` (int 0/1).
- **`sum_pfas_ngL`** + **`target_sum_gt70`** + **`pfas_class_assignment`**.

### T1 — binaire (déjà matérialisée et vérifiée)
- `target_sum_gt70 == (sum_pfas_ngL > 70)` : **match 100.0 %**.
- `sum_pfas_ngL == Σ des 31 `*_ngL`` (NaN→0) : écart max **1.8e-12** (égalité numérique),
  corr 1.000. → la somme est exactement la reconstruction des 31 analytes.
- **Seuil = 70 ng/L** : cohérent avec le **Response Level cumulé PFOA+PFOS** historique
  de la California State Water Board / orientation US EPA Health Advisory (ordre 70 ng/L).
  Définition retenue pour T1 : **`y = 1 si somme PFAS > 70 ng/L`**.
- **Prévalence T1 = 0.2480** (11 490 / 46 338 positifs) → déséquilibre modéré (~1:3).

> Remarque : 70 ng/L est un seuil **cumulé**. Une variante T1 réglementaire plus récente
> (MCL fédéraux 2024 : PFOA 4 ng/L, PFOS 4 ng/L individuels) pourrait être dérivée des
> `*_ngL`, mais cela reste hors mode strict (les `*_ngL` sont des fuites). On garde la
> cible fournie `target_sum_gt70` comme T1 de référence.

### T2 — multilabel (vérifiée)
- `label_<analyte> == (<analyte>_ngL > 2.0)` (NaN→0) : **match 100 % pour les 31 labels**.
  Le seuil est **uniforme à 2.0 ng/L** (≈ niveau de notification/quantification), PAS un
  seuil réglementaire propre à chaque PFAS. Preuve : pour chaque analyte, max(conc | label=0)
  = 2.0 exactement et min(conc | label=1) > 2.0.
- **3 labels dégénérés** (toujours 0) : `label_PFEESA`, `label_PFMBA`, `label_PFMPA`
  → **à exclure de T2** (aucun signal).
- Prévalences T2 (sur les 28 labels exploitables), du plus fréquent au plus rare :
  PFOS 0.495 · PFOA 0.452 · PFHxS 0.470 · PFBS 0.391 · PFHxA 0.383 · PFHpA 0.279 ·
  FTS_6_2 0.279 · PFBA 0.274 · FTS_8_2 0.244 · PFPeA 0.235 · PFNA 0.147 · PFPeS 0.102 …
  jusqu'aux très rares : NFDHA 0.017, F53B_major 0.025, PFDS 0.023.
  → **forte hétérogénéité de prévalence** → macro-AUROC + métriques par label requises.
- Cohérence T1/T2 : `target_sum_gt70=1` ⊂ `(≥1 label=1)` (crosstab : aucune ligne sum>70
  sans label positif). 74.8 % des lignes ont ≥1 dépassement individuel à 2 ng/L, mais
  seulement 24.8 % dépassent la somme 70.

### Stabilité intra-puits (alerte fuite panel)
- **94.2 % des puits ont une T1 constante** sur toutes leurs dates (std intra-puits
  moyenne 0.026). → si un split aléatoire met des lignes d'un même puits des deux côtés,
  le modèle « reconnaît le puits » : **fuite par mémorisation**. **Splitter par puits.**

---

## 5. Détection de fuite (data leakage)

Méthode : écran univarié |corr| et AUC de chaque colonne vs `target_sum_gt70`
(`roc_auc_score`, graine 42). Les colonnes PFAS dominent le classement, les colonnes de
contexte plafonnent bien plus bas.

### À EXCLURE ABSOLUMENT des features (fuite directe de la cible)
| famille | colonnes | preuve |
|---|---|---|
| Cible elle-même | `target_sum_gt70` | corr 1.0, AUC 1.0 — c'est y |
| Somme PFAS | `sum_pfas_ngL` | AUC **1.0** (T1 = somme>70 par définition) |
| 31 concentrations | tous les `*_ngL` | mesures de la cible ; ex. `PFOA_ngL` AUC 0.945, `PFBS_ngL` AUC 0.936, `PFBA_ngL` AUC 0.964 ; et leur somme = la cible |
| 31 labels individuels | tous les `label_*` | dérivés de `*_ngL` (>2.0) ; ex. `label_PFHpA` corr **0.730**, `label_PFHxA` 0.623, `label_PFOA` 0.588 |
| 31 flags détection | tous les `*_detected` | post-mesure de l'analyte ; ex. `PFHpA_detected` corr 0.696 ; crosstab PFOS detected×label quasi diagonal |
| Méta cible | `pfas_class_assignment` | JSON statique de classes PFAS (constante, sans info mais cible-liée) |

→ **96 colonnes PFAS-dérivées exclues** (1 + 1 + 31 + 31 + 31 + 1). Toute feature pour
T1/T2 doit provenir des 121 colonnes restantes.

### À SURVEILLER (légitimes mais à justifier)
- **Co-contaminants `cocontam_*`** (≈42 colonnes) : ce sont des mesures d'AUTRES polluants
  (TCE, PCE, nitrates, TDS, As, Mn, BTEX…) co-échantillonnés. Ils ne dérivent PAS des PFAS
  et leurs corrélations avec T1 sont faibles (max `cocontam_dbcp` 0.112, `cocontam_pce`
  0.089). **Légitimes** comme proxys de contamination industrielle/urbaine, MAIS : (a) ils
  sont co-mesurés à la même date → risque de fuite temporelle subtile si on prédit « ce
  jour-là » ; (b) leur disponibilité dépend de la source (MAR). À conserver avec prudence
  + indicateur de manquant, et à monitorer en importance SHAP.
- **`depth_eff_ft` / `screen_mid_ft` / `well_depth_ft`** : |corr| 0.27–0.38 (les plus
  fortes features non-PFAS). C'est **plausible mécaniquement** (puits profonds = moins de
  PFAS de surface), AUC < 0.5 (relation inverse), pas une fuite. À garder.
- **Coordonnées `latitude`/`longitude`** : voir §7 — à NE PAS mettre en feature brute
  (mémorisation spatiale), mais utilisables pour construire blocs/graphe.

### NON fuite (confirmé)
Aucune feature de contexte n'a de corrélation suspecte (>0.4) avec T1. Le plus haut hors
PFAS est `depth_eff_log1p` à 0.382, interprétable.

---

## 6. Espace de features candidat (121 colonnes non-PFAS)

Groupé par famille, avec transformation/imputation proposée :

1. **Hydrogéologie / puits** : `depth_eff_ft`(+`_log1p`), `screen_mid_ft`,
   `screen_length_ft`, `depth_to_water_m`(+`_log1p`), `dtw_far`,
   `hydr_grad_mag_permil` (log + cap), `flow_dir_sin`, `flow_dir_cos`,
   `dist_nearest_gwl_km`. → impute médiane + indicateurs `depth_missing`/`topo_missing`.
2. **Topographie** : `elevation_m`. → impute médiane + `topo_missing`.
3. **Sol (SSURGO-like)** : `soil_sand/clay/silt_pct`, `soil_om_pct`, `soil_ph`,
   `soil_ksat_um_s`, `soil_awc_cm_cm`, `soil_bulk_density`, granulométrie sable,
   `soil_water_1bar/15bar_pct`, `soil_texture_class` (cat), `soil_ratio_water_clay`.
   → drop les variantes >95 % manquantes (silt fin/grossier, gradation) ; impute médiane.
4. **Proximité sources de contamination (GeoTracker)** : `dist_geotracker_km`,
   `nearest_geotracker_type` (cat : Chrome Plater/Bulk Terminal/Airport/Refinery),
   `n_geotracker_within_{1,3,10,50}km`. → features clés (proxy source PFAS), prêtes.
5. **Co-contaminants** : `cocontam_*` (garder ceux <85 % manquants, ~30) — voir §5 surveil.
6. **Qualité de l'air (AQS)** : `aqs_pm25/pm10/no2/so2/co/ozone`, `aqs_wind_ms`,
   `aqs_humidity_pct`. → impute médiane.
7. **Climat / hydrologie (GLDAS)** : `rainfall_mm_month`, `et_mm_month`, `runoff_mm`,
   `soil_moi_*`, `root_zone_moist`, `temp_c`, `snowpack_mm`, `soil_moisture_total_mm`,
   `gldas_dist_km`. → impute médiane.
8. **Usage du sol** : `lc_developed`, `dev_intensity` (0–4 ordinal).
9. **Administratif / contexte** : `county`, `regional_board`, `dwr_region`,
   `dwr_basin`, `sgma_*`, `gm_dataset_name`, `gm_well_category`.
   → encodage fréquence ou regroupement (haute cardinalité) ; `gm_dataset_name`
   conservée car porteuse du mécanisme de manquant.
10. **Temporel** : dériver de `collection_date` → année, mois, saison (sin/cos),
    tendance. (À utiliser avec prudence : la couverture analytique évolue dans le temps.)
11. **Indicateurs de manquant** : `depth_missing`, `topo_missing` (fournis) +
    en créer pour les co-contaminants à manquant MAR.

---

## 7. Structure spatiale

- **Coordonnées** : `latitude` (10 678 valeurs uniques), `longitude` (10 699), 0 % manquant.
- **Distances inter-puits** (BallTree haversine, niveau puits, n=11 333) :
  plus proche voisin médiane **0.107 km**, q25 0.005 km, q75 0.546 km, max 40.6 km.
  **49.2 % des puits ont un voisin < 100 m, 88.6 % < 1 km** → puits très **agrégés**.
- **Autocorrélation spatiale de T1** (Moran's I, k=8 NN binaire, niveau puits) :
  **I = 0.558** (attendu sous H0 = -0.0001 ; permutation 99×: moyenne 0.006, std 0.004,
  **p ≈ 0.01**). → autocorrélation **forte et significative**.
- **Prévalence T1 par comté** : de **0 %** (Siskiyou, Inyo) à **82 %** (Contra Costa),
  Tehama 71 %, Orange 42 %, Riverside 39 %. Hétérogénéité spatiale massive.

### Conséquences / stratégie de validation
- **Splitter par puits ET par bloc spatial.** Un split aléatoire au niveau ligne gonfle
  les scores par double fuite (panel intra-puits 94 % + autocorrélation I=0.558).
- **Blocs spatiaux recommandés** : KMeans sur `(lat,lon)` **au niveau puits**, puis chaque
  ligne hérite du bloc de son puits. Tests :
  - k=5 : tailles 2 609–24 647, prévalence 0.116–0.313.
  - **k=8 (recommandé)** : tailles 1 747–18 768, prévalence 0.034–0.415 — bon compromis
    contraste/équilibre.
  - k=10 : un bloc descend à 327 lignes (trop petit).
  Alternative interprétable : blocs = `county` ou `dwr_basin` (GroupKFold), plus lisibles
  mais tailles déséquilibrées.
- **Localisation pure** : **NE PAS** mettre `latitude`/`longitude` brutes en feature
  (le modèle mémoriserait la carte de prévalence). Les coordonnées servent uniquement à
  (a) construire les blocs de CV, (b) construire le graphe (arêtes kNN) pour les GNN, et
  (c) dériver des features de contexte agrégées sans identité (ex. densité GeoTracker,
  déjà présentes). Rapporter systématiquement **CV aléatoire (par puits) vs CV spatiale
  (blocs)** et l'écart.

---

## 8. Comparaison v1 vs v2

- v1 `CA-PFAS-ASGWS.parquet` : **46 338 × 201**. v2 : **46 338 × 217**.
- **Mêmes lignes exactement** : 46 338 clés `(gm_well_id, collection_date)` partagées,
  0 ligne v1-only, 0 v2-only. **Aucune colonne supprimée.**
- **16 colonnes AJOUTÉES en v2** (toutes contexte hydrogéo/topo, aucune PFAS) :
  `depth_eff_ft`, `depth_eff_log1p`, `depth_missing`, `depth_to_water_m`,
  `depth_to_water_log1p`, `dtw_far`, `dist_nearest_gwl_km`, `elevation_m`,
  `flow_dir_sin`, `flow_dir_cos`, `hydr_grad_mag_permil`, `lc_developed`,
  `dev_intensity`, `screen_length_ft`, `screen_mid_ft`, `topo_missing`.
- Interprétation : v2 = **enrichissement features** (profondeur effective, profondeur de
  nappe, gradient/direction d'écoulement, élévation, géométrie de crépine, usage du sol),
  sans changer cibles ni population. Les nouvelles features de profondeur sont parmi les
  plus corrélées à T1 (legitimes, |corr| 0.27–0.38).

---

## 9. Recommandations

**Cibles**
- **T1** : `target_sum_gt70` (somme PFAS > 70 ng/L), prévalence 0.248. Prête.
- **T2** : 28 labels `label_*` SAUF `label_PFEESA/PFMBA/PFMPA` (toujours 0). Seuil
  uniforme 2 ng/L. Forte hétérogénéité de prévalence → macro-AUROC + par-label + EMR/Hamming.

**Fuite — liste d'exclusion reconstruite (96 colonnes)**
- Toutes `*_ngL` (31), `label_*` (31), `*_detected` (31), `sum_pfas_ngL`,
  `target_sum_gt70`, `pfas_class_assignment`. + ne pas mettre `latitude`/`longitude`
  brutes en feature.

**Features**
- Partir des **121 colonnes non-PFAS**, retirer les inutilisables (§2 : ~12 colonnes
  >85 % manquantes/constantes), imputer médiane + indicateurs de manquant, log1p sur les
  variables très asymétriques (profondeurs, gradient), encodage circulaire déjà fourni
  pour la direction d'écoulement, encodage fréquence pour les catégorielles à forte
  cardinalité. Conserver `gm_dataset_name`/`gm_well_category` (mécanisme de manquant).
- Surveiller les `cocontam_*` (co-mesure même date → risque de fuite temporelle subtile)
  via SHAP.

**Validation**
- **GroupKFold par `gm_well_id`** (anti-fuite panel : 94 % T1 constante intra-puits)
  **ET** CV spatiale par blocs **KMeans k=8 sur (lat,lon) au niveau puits**. Rapporter
  les deux et l'écart (l'écart mesure la fuite spatiale, I=0.558).
- Optimiser le seuil de décision sur probabilités out-of-fold uniquement.

**Alertes fortes**
1. **Fuite panel** : ne jamais splitter au niveau ligne — splitter par puits.
2. **Fuite spatiale** : Moran's I = 0.558 (p≈0.01) → CV spatiale obligatoire ; coordonnées
   hors features.
3. **Fuite cible** : 96 colonnes PFAS-dérivées (dont les `_detected`, souvent oubliés)
   doivent être exclues ; `sum_pfas_ngL` et la somme reconstruite ont AUC 1.0.
4. **MAR par source** : la disponibilité du panel PFAS étendu et des co-contaminants
   dépend de `gm_dataset_name`/catégorie → conserver ces variables + indicateurs, ne pas
   imputer naïvement.
5. **3 labels T2 dégénérés** à retirer.

---

## Critique hydrogéochimique (hydro-domain-expert)

> Revue mécaniste de l'espace de features du §6, depuis la chimie/transport des PFAS en
> eaux souterraines. Référence uniquement les colonnes listées dans ce rapport. Convention :
> « driver de 1er ordre » = mécanisme direct et dominant ; « modulateur » = effet réel mais
> de 2nd ordre / conditionnel ; « proxy » = pas de causalité directe mais corrélat de source.

### A. Pertinence mécaniste des features de contexte (Q1)

**Drivers / proxys mécanistiquement FORTS (à conserver, ce sont les colonnes porteuses) :**

- **Proximité de sources (GeoTracker)** — `dist_geotracker_km`, `nearest_geotracker_type`,
  `n_geotracker_within_{1,3,10,50}km`. C'est le **socle mécaniste** du problème. Les PFAS
  n'ont pas d'origine géogénique : toute détection remonte à une source anthropique
  (AFFF d'aéroports/sites militaires, raffineries, chrome platers, terminaux de
  stockage). `nearest_geotracker_type` est particulièrement précieux car la **typologie de
  source prédit la signature** : un aéroport/site AFFF → PFOS, PFHxS, 6:2 FTS dominants ;
  un site industriel/fluoropolymère → PFOA, PFNA, GenX. Les compteurs multi-rayons
  encodent une **dose-distance** (densité de sources amont) cohérente avec un panache
  advectif. Réserve : la distance euclidienne n'est pas la distance hydraulique (cf. §C et
  critique du graphe) — une source à 500 m en aval n'a pas le même sens qu'à 500 m en amont.

- **Géométrie de prélèvement vertical** — `depth_eff_ft`, `screen_mid_ft`,
  `screen_length_ft`, `depth_to_water_m`, `dtw_far`. Mécanisme solide : les PFAS sont
  émis en surface et migrent vers le bas ; la concentration capturée dépend de **quelle
  tranche d'aquifère la crépine intègre**. Voir §C pour le détail.

- **Usage du sol développé** — `lc_developed`, `dev_intensity`. Proxy de pression
  anthropique diffuse (réseaux d'assainissement, STEP, ruissellement urbain, dépôts
  atmosphériques industriels). Mécanistiquement défendable comme source diffuse
  complémentaire aux sources ponctuelles GeoTracker.

- **Matière organique et texture du sol** — `soil_om_pct`, `soil_clay_pct`,
  `soil_sand_pct`, `soil_ksat_um_s`, `soil_texture_class`, `soil_bulk_density`. Mécanisme
  de **rétention** réel : les PFAS à longue chaîne (PFOS, PFOA, PFHxS) sont retenus par
  sorption à l'interface eau-solide et sur la matière organique ; `soil_ksat_um_s` (conduc-
  tivité hydraulique du sol) gouverne la vitesse d'infiltration vers la nappe. Ce sont des
  **modulateurs légitimes** du transport. Nuance importante : ces variables décrivent
  l'horizon de SOL de surface (SSURGO), pas le matériau aquifère saturé — leur pouvoir
  prédictif sur une concentration en nappe profonde est de 2nd ordre.

**Modulateurs FAIBLES / mécanistiquement DOUTEUX (à dégrader en priorité d'inclusion,
candidats à l'ablation) :**

- **Qualité de l'air (AQS)** — `aqs_pm25`, `aqs_pm10`, `aqs_no2`, `aqs_so2`, `aqs_co`,
  `aqs_ozone`, `aqs_wind_ms`, `aqs_humidity_pct`. **Mécanistiquement très faibles pour
  l'eau souterraine.** Le dépôt atmosphérique de PFAS existe mais n'est pas mesuré par ces
  polluants critères (NO2/SO2/CO/ozone/PM sont des proxys de combustion/trafic, non des
  PFAS aéroportés). Au mieux co-proxys d'urbanisation, déjà mieux capturés par
  `lc_developed`/`dev_intensity`. Risque : qu'un modèle leur attribue de l'importance par
  **confusion spatiale avec l'urbanisation** (artefact, cf. §F). Recommandation : tester en
  ablation, attendre une justification SHAP avant de les conserver.

- **Climat / GLDAS instantané** — `temp_c`, `aqs_humidity_pct`, `snowpack_mm`. La
  recharge (`rainfall_mm_month`, `runoff_mm`, `et_mm_month`, `root_zone_moist`,
  `soil_moi_*`) a un sens mécaniste (la recharge dilue OU mobilise/lessive les PFAS vers
  la nappe — effet **ambigu de signe**), mais la **température et le snowpack instantanés**
  n'ont pas de lien causal direct avec une concentration PFAS en nappe. Ce sont des proxys
  d'altitude/saison/région → risque de réencoder la géographie (cf. §F). `gldas_dist_km`
  est une distance à une grille de données : purement méthodologique, **aucun sens
  mécaniste**, à exclure des features (le garder éventuellement comme indicateur de
  qualité d'imputation, pas comme prédicteur).

- **Découpage administratif** — `county`, `regional_board`, `dwr_region`, `dwr_basin`,
  `sgma_*`. Aucun mécanisme hydrogéochimique : une frontière de comté n'arrête pas un
  panache. Leur seul pouvoir prédictif viendrait de la **mémorisation de la carte de
  prévalence** (Contra Costa 82 % vs Siskiyou 0 %, §7) — c'est exactement le risque
  d'artefact spatial que le §7 cherche à éviter en retirant lat/lon. Inclure `county` en
  feature contredit cette précaution : à **traiter comme lat/lon** (réservé aux blocs de
  CV, pas en feature), ou au mieux n'autoriser que `dwr_basin`/`sgma_subbasin_name` si on
  veut un proxy d'**unité hydrogéologique** (un bassin SGMA approxime un système aquifère
  partagé — justification mécaniste faible mais non nulle, contrairement au comté). À
  monitorer en SHAP : une importance forte du comté = signal d'alerte.

- **`gm_dataset_name` / `gm_well_category`** : aucun sens mécaniste sur la concentration,
  mais le §2 montre qu'ils encodent le **mécanisme de manquant (MAR)**. Légitimes à ce
  titre uniquement ; toute importance SHAP au-delà de la gestion du manquant signalerait
  que le modèle apprend un **biais d'échantillonnage par programme** (ex. WB_CLEANUP =
  sites de dépollution, donc déjà contaminés par construction) plutôt qu'un mécanisme —
  fuite d'intention d'échantillonnage. Voir §F.

### B. Features mécanistiquement importantes mais ABSENTES (Q2)

Aucune de ces variables n'apparaît dans les 121 colonnes du §6 ; leur absence plafonne le
réalisme mécaniste atteignable :

- **Géochimie de l'eau in situ** : pH de l'eau souterraine, potentiel redox (Eh/O2
  dissous), conductivité spécifique. Mécaniste de 1er ordre pour la **spéciation et la
  sorption** des PFAS (sorption des sulfonates dépend du pH et de la force ionique ;
  conditions redox gouvernent la (bio)transformation des précurseurs en PFAA terminaux).
  Le rapport ne liste que `soil_ph` (pH du SOL de surface, pas de l'eau) et
  `cocontam_tds` (proxy de minéralisation, mais c'est un co-contaminant à manquant MAR).
  Lacune importante.

- **Carbone organique dissous (COD) de la nappe** : contrôle direct la sorption/mobilité
  des PFAS dans l'aquifère saturé. `soil_om_pct` (matière organique du sol de surface)
  n'en est qu'un substitut éloigné.

- **Type/lithologie de l'aquifère et confinement** (alluvial vs fracturé vs confiné,
  conductivité hydraulique de l'aquifère, transmissivité). Le §6 ne fournit que des
  propriétés de SOL de surface (SSURGO) et `hydr_grad_mag_permil` ; il manque le **milieu
  saturé** où circule réellement le panache. `dwr_basin`/`sgma_*` n'en sont qu'un proxy
  administratif grossier.

- **Âge / temps de résidence de l'eau** (proxys : tritium, CFC, nitrates comme traceur).
  Les PFAS étant récents (post-~1950) et persistants, l'âge de l'eau discrimine fortement
  eau ancienne profonde (PFAS-libre) vs eau moderne. `depth_to_water_m`/`depth_eff_ft` en
  sont un proxy indirect, mais un traceur d'âge serait de 1er ordre.

- **Inventaire de sources spécifiques PFAS** : présence avérée de sites AFFF, STEP avec
  rejet, épandage de boues, décharges. GeoTracker (`nearest_geotracker_type`) en capture
  une partie via la typologie, mais sans distinguer les sites à **usage PFAS confirmé** des
  sites de dépollution génériques (solvants chlorés sans PFAS). C'est la lacune la plus
  pénalisante pour la spécificité mécaniste.

### C. Profondeur, géométrie de crépine et écoulement (Q3)

Famille globalement **bien justifiée mécanistiquement** — c'est l'apport le plus solide de
v2 :

- **`depth_eff_ft`, `screen_mid_ft`** (profondeur de la zone captée) : mécanisme de
  **dilution/atténuation verticale**. Source en surface → décroissance de concentration
  avec la profondeur d'intégration de la crépine. Cohérent avec le |corr| 0.27–0.38 négatif
  observé (§5). Driver légitime.

- **`screen_length_ft`** : une crépine longue **intègre/mélange** plusieurs niveaux →
  dilution d'un panache localisé ; effet mécaniste réel sur la concentration mesurée
  (artefact de prélèvement physique, pas chimique, mais légitime). À garder.

- **`depth_to_water_m`, `dtw_far`** : profondeur de la zone non saturée que les PFAS
  doivent traverser. Mécanisme double et **ambigu** : nappe profonde = plus long temps de
  transit/atténuation (protège), MAIS aussi accumulation possible à l'interface
  capillaire. Réserve méthodologique forte : `depth_to_water_m` est à **0 % de manquant
  alors qu'il est « imputé/modélisé » (§3)** — une variable modélisée peut réintroduire de
  la structure spatiale du modèle source (risque d'artefact, cf. §F). À traiter avec la
  même prudence qu'un proxy géographique.

- **`elevation_m`** : pas de mécanisme PFAS direct. Proxy de position topographique
  (zones de recharge en altitude vs zones de décharge en vallée où l'eau — et les
  contaminants — convergent). Effet de 2nd ordre, fortement corrélé à la géographie →
  risque de réencodage spatial. Modulateur faible, à surveiller en SHAP.

- **`hydr_grad_mag_permil`, `flow_dir_sin/cos`** : mécanistiquement **les bonnes
  variables** (l'advection le long du gradient est LE mode de transport PFAS dominant).
  MAIS deux réserves : (1) le gradient et la direction n'ont de sens prédictif que
  **relativement à la position des sources** — isolés, ils n'indiquent pas si le puits est
  en amont ou en aval d'un panache. Leur valeur se révélera surtout dans un **GNN où les
  arêtes suivent le sens d'écoulement** (orienter les arêtes well→well selon
  `flow_dir_*`), pas comme features tabulaires nodales. (2) `hydr_grad_mag_permil` a un max
  de 43 900 ‰ (§3) — un gradient de 4 390 % est **physiquement aberrant** (cela
  dépasserait une pente verticale) ; ce sont des artefacts d'interpolation à capper
  agressivement, sinon le modèle apprendra du bruit numérique.

- **`dist_nearest_gwl_km`** (distance au point de mesure de niveau le plus proche) :
  c'est un indicateur de **qualité d'interpolation** des variables d'écoulement, pas un
  mécanisme. Loin d'un piézomètre → gradient/direction peu fiables. À garder comme
  pondération de confiance, pas comme driver mécaniste.

### D. Co-contaminants comme indicateurs PFAS (Q4)

Mécanistiquement, les `cocontam_*` (TCE/`cocontam_pce`, nitrates `cocontam_no3n`, TDS
`cocontam_tds`, As/Mn, BTEX, `cocontam_dbcp`, `cocontam_dce12c`…) ne sont **pas des
indicateurs causaux** de PFAS : aucune chimie ne lie une molécule de PCE à une molécule de
PFOS. Leur seul lien est une **co-localisation de sources** (un site industriel/militaire
émet souvent solvants chlorés ET AFFF). Conditions :

- **Légitime** si et seulement si le co-contaminant agit comme **proxy de source/activité
  industrielle stable dans le temps** (le site a historiquement utilisé les deux). C'est
  plausible pour les solvants chlorés (`cocontam_pce`, `cocontam_dce12c`, `cocontam_tce`)
  sur sites AFFF/militaires, et c'est cohérent avec la corrélation faible mais non nulle
  observée (`cocontam_dbcp` 0.112, `cocontam_pce` 0.089, §5). Cette faiblesse est en fait
  **rassurante** : un proxy de co-source doit être faiblement prédictif, pas fortement.

- **Problématique** sur deux fronts. (1) **Fuite temporelle** (déjà signalée §5) : le
  co-contaminant est mesuré le MÊME jour que le PFAS. Si l'intention d'échantillonnage est
  « ce site est pollué, on analyse tout », alors `cocontam_*` encode le fait qu'une
  contamination est *déjà connue ici* — ce n'est plus de la prédiction de contexte mais de
  la **co-mesure d'état**. (2) `cocontam_no3n` est un **mauvais proxy spécifique** : les
  nitrates tracent l'agriculture/assainissement, pas les sources PFAS industrielles —
  mécanisme orthogonal, de toute façon écarté (94.9 % manquant, §2). **Recommandation
  mécaniste** : conserver uniquement les solvants chlorés à couverture suffisante comme
  proxy de co-source AFFF/industrielle ; les traiter en ablation (avec/sans) pour mesurer
  leur part de fuite temporelle ; une importance SHAP élevée d'un `cocontam_*` doit être
  lue comme **suspecte par défaut** (co-mesure d'état) et non comme mécanisme.

### E. Seuils de cible (Q5)

- **T1 = 70 ng/L (somme)** : défendable comme cible réglementaire de référence (Response
  Level cumulé historique). Réserve mécaniste : une **somme non pondérée** de 31 analytes
  mélange des composés de toxicité et de persistance très différentes (PFOS/PFOA à longue
  chaîne, bioaccumulatifs, vs PFBA/PFPeA à chaîne courte, peu toxiques mais très mobiles).
  Deux puits à « somme = 70 » peuvent avoir des profils de risque opposés. Acceptable pour
  une tâche de dépassement réglementaire, mais le §9 a raison de noter que les MCL 2024
  (PFOA/PFOS à 4 ng/L) seraient plus protecteurs — hors mode strict ici.

- **T2 = 2 ng/L uniforme** : **mécanistiquement et toxicologiquement non défendable comme
  seuil de risque.** 2 ng/L est un seuil **analytique** (quantification/notification),
  pas un seuil sanitaire. Conséquences mécanistes concrètes visibles dans les données :
  (1) il **gomme la hiérarchie de toxicité** — PFOS/PFOA (MCL fédéral 4 ng/L,
  bioaccumulatifs) sont mis sur le même pied que PFBA/PFHxA/PFPeA (chaîne courte, seuils de
  référence de l'ordre de centaines à milliers de ng/L). (2) Il **confond détectabilité et
  dépassement** : les prévalences T2 (PFOS 0.495, PFOA 0.452 vs NFDHA 0.017, §4) reflètent
  autant la **fréquence d'occurrence/limites analytiques** que le risque réel. Un modèle
  T2 à 2 ng/L prédit donc « est-ce mesurable » plus que « est-ce dangereux ».
  **Recommandation** : pour la pertinence sanitaire, des **seuils différenciés par
  analyte** seraient préférables — PFOS/PFOA/PFHxS/PFNA à leurs valeurs réglementaires
  (ordre 4–10 ng/L), chaîne courte à leurs seuils de référence propres. À défaut de
  re-dériver (les `*_ngL` sont des fuites côté features mais restent utilisables pour
  **définir la cible**), garder 2 ng/L comme cible « occurrence » en l'**étiquetant
  explicitement comme seuil analytique et non sanitaire** dans tout livrable, pour éviter
  une sur-interprétation du risque.

### F. Alertes de plausibilité — importance SHAP suspecte par défaut (Q6)

Une importance forte de l'une de ces colonnes doit être traitée comme **artefact/fuite
présumé** jusqu'à preuve mécaniste, pas comme découverte :

1. **`gm_dataset_name` / `gm_well_category`** importantes → le modèle apprend
   l'**intention d'échantillonnage** (WB_CLEANUP = sites de dépollution, contaminés par
   construction) et non un mécanisme physique. Fuite de design d'étude.
2. **Découpage administratif** (`county`, `regional_board`, `dwr_*`, `sgma_*`) important →
   **mémorisation de la carte de prévalence** (Contra Costa 82 % vs Siskiyou 0 %),
   équivalent fonctionnel d'utiliser lat/lon brutes que le §7 interdit. Réencodage spatial.
3. **`cocontam_*`** important → présomption de **fuite temporelle / co-mesure d'état**
   (cf. §D), pas de mécanisme de co-source.
4. **Variables AQS / climat instantané** (`aqs_*`, `temp_c`, `snowpack_mm`,
   `elevation_m`) importantes → **proxy géographique déguisé** (altitude/région), pas de
   chimie PFAS. Confusion spatiale avec l'urbanisation/topographie.
5. **`depth_to_water_m`** important alors qu'il est **imputé/modélisé à 0 % manquant** →
   risque de réinjecter la structure du modèle d'imputation (artefact). Comparer à
   `depth_eff_ft` (réel, 49 % manquant) : un écart d'importance en faveur de la variable
   imputée est un signal d'alerte.
6. **`gldas_dist_km` / `dist_nearest_gwl_km`** importantes → le modèle exploite la
   **qualité d'imputation** comme prédicteur (les puits bien instrumentés diffèrent
   systématiquement des autres), pur artefact méthodologique.

**Inversement, signature mécaniste SAINE attendue** : importance dominée par
`dist_geotracker_km` / `n_geotracker_within_*km` / `nearest_geotracker_type` (proximité
de sources), puis profondeur/crépine (`depth_eff_ft`, `screen_mid_ft`), puis rétention sol
(`soil_om_pct`, texture) et usage du sol (`dev_intensity`). Si ce n'est PAS l'ordre
observé, suspecter un des artefacts ci-dessus avant de conclure à un mécanisme nouveau.

**Note transverse pour les GNN** : les variables d'écoulement (`flow_dir_sin/cos`,
`hydr_grad_mag_permil`) ne déploient leur sens mécaniste qu'en **orientant les arêtes du
graphe selon le sens d'écoulement réel** (connectivité de nappe), pas comme features
nodales tabulaires. Une arête « kNN spatial » qui ignore le gradient ne fait que
**réencoder la carte** (même artefact que §F-2) ; à confronter à la connectivité
hydraulique avant de valider la topologie.
