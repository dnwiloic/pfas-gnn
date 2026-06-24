# Recommandations : collecte et agrégation de données pour l'exploration HGT — PFAS CA

**Date :** 2026-06-24.
**Périmètre :** Construire un dataset permettant d'explorer systématiquement les architectures
HGT sur le graphe hétérogène PFAS CA (T1a, mode prédictif strict, CV spatiale k=8).
**Sources :** synthèse des analyses de gnn-researcher, hydro-domain-expert, data-analyst,
et eval-methodologist sur les données disponibles dans `ca-pfas-ml/data/raw/` et `processed/`.

> Ce document ne contient aucune conclusion sur les résultats de modélisation — il fixe
> uniquement ce qu'il faut collecter, comment l'agréger, et les contraintes méthodologiques
> à respecter.

---

## Synthèse exécutive

Le graphe hétérogène à 4 types de nœuds réels est désormais **mécanistiquement et
méthodologiquement justifiable** (évaluation `eval_hetero5_spatial.md`). Les données brutes
nécessaires existent pour l'essentiel dans `ca-pfas-ml/data/raw/`. Deux actions principales :

1. **Construire les 5 tables parquet** du graphe (§4) à partir des sources disponibles.
2. **Collecter 3 sources manquantes prioritaires** (§5) pour enrichir les features de nœuds.

---

## 1. Architecture cible du graphe

### Types de nœuds

| Type | Source | Nb entités | Rôle mécaniste |
|---|---|---:|---|
| **`well`** | `CA-PFAS-ASGWS.parquet` | 11 333 puits | Point d'observation de la nappe (1 nœud/puits, pas 1/ligne) |
| **`facility`** | geotracker (463) + WWTP (350) + **DoD (83)** + **décharges SWIS (1 710)** | **~2 560 sites** | Source de contamination PFAS (AFFF/DoD = n°1, PFOS industriel, biosolides, lixiviat) — *attribut `facility_type` à 7 niveaux* |
| **`aqs_station`** | `aqs_ca_annual.parquet` | **305 stations** | Co-exposition atmosphérique partagée (co-traceurs industriels) |
| **`subbasin`** | `sgma_subbasin_name` / geojson | **237** | Compartiment hydrogéologique partagé |

### Types d'arêtes (priorité décroissante)

| Relation | Cap | Arêtes estimées | Justification mécaniste | Protocole de coupe |
|---|---:|---:|---|---|
| `well↔well near` | 1,5 km | ~65 k | Voisinage hydraulique direct | `cut_cross_block`, assert 0 |
| `well↔well subbasin` | 2 km intra | ~qq k | Connectivité de bassin | `cut_cross_block` séparé |
| `facility→well` | **3 km** (1 km en variante) | **à recalculer** (4 sources) | Source → récepteur de panache | Dédoubler `facility@bloc` |
| `aqs_station→well` | 10 km | ~15 k | Co-exposition atmosphérique | Dédoubler `aqs@bloc` |
| `well→well flows_to` | ≤ 3 km + angle | à calculer | Transport orienté (aval) | `cut_cross_block`, après feature gradient |

**Note sur le cap facility :** 1 km → 21 % de puits couverts, 0 entité multi-blocs (coupe gratuite).
3 km → 44 % de puits couverts, 4 entités multi-blocs (gérées par dédoublement). **Recommandation :
3 km comme relation principale, 1 km comme variante « proximité forte »** ; les deux ablations
étiquetées dans le run de comparaison.

**⚠️ Recompute requis (4 sources) :** les chiffres ci-dessus valent pour 813 sites
(geotracker + WWTP). Avec DoD + décharges (~2 560 sites, dont **1 710 décharges denses**),
la couverture et le nombre d'arêtes augmentent fortement. Recalculer les caps, et envisager
un **kNN plafonné par `facility_type`** (p. ex. ≤ k voisins par type) pour qu'un type dense
(décharges, Chrome Plater n=271) ne sature pas le voisinage et ne redevienne pas un
proxy-densité.

---

## 2. Features par type de nœud

### Nœud `well` (64 features)

**Existantes (61 colonnes `config.feature_columns()`) :** CONSERVER telles quelles.
`lat`/`lon` HORS features (C-LOC.1 — la géographie passe par la topologie).

**À ajouter (gradient hydraulique, déjà calculé) :**
- `flow_dir_sin`, `flow_dir_cos` — direction d'écoulement DWR
- `hydr_grad_mag_permil` — magnitude du gradient
- `flow_dir_missing` — indicateur pour les 92 NA (puits > 44 km de toute station DWR, zone SE)
- Imputation des 92 NA : sin=cos=0, magnitude → médiane sous-bassin. **Ne pas imputer par médiane régionale** (aucune donnée locale).
- **Source :** `ca-pfas-ml/data/processed/well_hydraulic_gradient.csv` — jointure parfaite (`gm_well_id` exact, 11 333/11 333).

**À retirer des features `well` (cohérence C-LOC.1, si présents dans `feature_columns()`) :**
les scalaires de proximité de source — `geotracker_*`, `wwtp_*`, `dod_*`, `landfill_*`
(`*_dist_km`, `*_n_within_*`). Ce signal passe désormais par la **topologie `facility→well`** :
les garder en plus serait un double comptage du proxy positionnel, et masquerait l'apport des
arêtes mécanistes orientées par le gradient. (Le compte de features `well` baisse en conséquence.)

**À collecter (§5.1) :** géochimie redox GAMA (O₂, Eh, NO₃, SO₄, Fe²⁺, Mn, NH₄, profondeur).

### Nœud `facility` (≈9 features, 7 types)

> **CHANGEMENT vs version initiale.** DoD et décharges deviennent des **NŒUDS**, pas des
> features de puits. Les laisser en scalaires `dod_*`/`landfill_*` sur le puits ré-aplatit
> la relation source→puits que le HGT doit exploiter — et c'est incohérent avec la priorité
> mécaniste ci-dessous (DoD = n°1, décharges = source diffuse réelle). On **unifie les quatre
> couches en un seul type `facility`** (~2 560 sites) portant un attribut `facility_type` à
> 7 niveaux. Un type unique (plutôt que 4 types séparés) évite la sparsité des relations
> minuscules (DoD n=83) et laisse le HGT **partager la physique de transport** entre types
> de sources tout en les distinguant par l'attribut.

**Sources brutes disponibles** (les quatre, déjà sur disque dans
`ca-pfas-ml/data/raw/contamination/` — voir `ca-pfas-ml/docs/limite_sources_pfas.md`) :
- `geotracker_pfas_investigation_sites.csv` — 463 sites (4 types industriels)
- `wwtp_epa_frs_ca.csv` — 350 STEP (effluents / biosolides)
- `dod_pfas_federal_sites_ca.csv` — 83 sites DoD/fédéraux (AFFF, **source n°1**)
- `swis_landfills_ca.csv` — 1 710 décharges / sites de disposal (lixiviat)

**Table `nodes_facility.parquet` à construire :**
```
facility_id   : str    # "GEO_<global_id>" | "WWTP_<i>" | "DOD_<i>" | "SWIS_<i>"
facility_type : cat(7) # Chrome Plater | Bulk Terminal | Refinery | Airport | WWTP | DoD | Landfill
                       #   (one-hot à l'encodage)
src_active    : bool   # source active/ouverte : STEP présente ; décharge status≈Active ;
                       #   DoD actif ; geotracker présumé sous ordre ; défaut 1
wwtp_major    : bool   # STEP de classe Major (proxy de débit, ≈53 % des STEP) ; 0 sinon
dod_known     : bool   # site DoD à détection PFAS CONFIRMÉE (pfas_presence='Known Detection',
                       #   33/83) ; 0 sinon
```
**`lat`/`lon` EXCLUS des features** (C-LOC.1 — cohérence avec `well`/`subbasin` : la position
n'entre que par la topologie des arêtes). Les coords servent uniquement au calcul des arêtes.

**Force de source dispo MAINTENANT** (pas besoin d'attendre l'API ECHO §5.2) : `facility_type`,
`src_active`, `wwtp_major`, `dod_known`. Le débit WWTP (§5.2) reste un *raffinement* ultérieur
de `wwtp_major`.

**Qualité / dédup :** dédupliquer chaque source sur (lat,lon) arrondi 4 déc. (7 doublons
geotracker, 29 WWTP). Les sources sont disjointes en coordonnées → concaténation sûre.
Filtrer bbox CA [32.5,42]×[-124,-114].

**Interdit (C-FAC.1) :** aucune concentration PFAS mesurée sur le site (panache, investigation)
en feature.
**À auditer (C-FAC.2 — NOUVEAU) :** `dod_known` est une *présence-au-site*, pas une
concentration — mais une base « known detection » a pu **déclencher** l'échantillonnage des
puits voisins (biais de sélection). Auditer la corrélation OOF de `dod_known` avec T1a (même
garde-fou que la géochimie §5.1) ; retirer si suspect.

**Priorité mécaniste des types (pour interprétation / ablation par type) :**
1. **DoD / fire-training (AFFF)** — source n°1 PFAS, désormais **NŒUD**
2. Airport (AFFF pistes) — panache direct, forte masse
3. WWTP Major (biosolides, effluents) — diffuse
4. **Landfill / Disposal (lixiviat)** — diffuse, désormais **NŒUD** (n=1 710, dense → cap §1)
5. Chrome Plater (PFOS galvanoplastie) — n=271, risque proxy-densité
6. Bulk Terminal / Refinery — contribution PFAS la plus incertaine

**Répercussions à propager (hors §1–§2) :**
- §3 : `edges_well_facility.parquet` recalculé sur les **4** sources.
- §4 étape 2 : concaténer les **4** CSV (et non 2) ; `facility_id` préfixé par source.
- §7 checklist : `nodes_facility.parquet` → **~2 560 sites** (et non ~797).

### Nœud `aqs_station` (7 features)

**Source :** `ca-pfas-ml/data/raw/environment/aqs/aqs_ca_annual.parquet`
(30 381 lignes, 305 stations, 8 polluants, 2015–2025)

**Pivotage à réaliser :** `(monitor_id, year, param_name, annual_mean)` →
`(monitor_id, lat, lon, ozone_mean, pm25_mean, pm10_mean, no2_mean, wind_speed_mean, humidity_mean, co_mean)`
Moyenne pondérée par `completeness_pct`, **années 2015–2023 uniquement** (fenêtre temporelle
alignée sur les prélèvements de puits, 2016–2023).

**Polluants retenus :** ozone (32 % NA), pm25 (50 %), pm10 (46 %), no2 (60 %), wind_speed (41 %),
humidity (56 %), co (73 %). **SO₂ exclu** (89 % NA, trop creux).
Imputation : plus-proche-station + indicateur `*_observed` (bool) par polluant.

**Table `nodes_aqs.parquet` :**
```
monitor_id     : str
lat, lon       : float
ozone, pm25, pm10, no2, wind_speed, humidity, co : float (NA imputés)
ozone_obs, pm25_obs, ... : bool  # indicateur de mesure réelle
```

**Fuite temporelle (C-AQS.3) :** La jointure AQS doit respecter `aqs.year ≤ collection_date.year`
du puits (pas de moyenne englobant le futur). Pour les puits antérieurs à 2019 (~570 lignes, <3 %)
utiliser une fenêtre `[year_puits − k, year_puits]`. Pour les puits 2020–2026 (majorité),
la moyenne 2015–2023 est temporellement saine.

### Nœud `subbasin` (4–5 features)

**Source :** `sgma_subbasin_name` (colonne parquet, 237 valeurs) +
`ca-pfas-ml/data/raw/environment/sgma_basins.geojson` (polygones officiels, 29 Mo)

**Features de nœud recommandées :**
- Surface du polygone (km²)
- Périmètre
- Nombre de puits dans le sous-bassin (calculé)
- Densité de puits (puits/km²)
- Élévation médiane des puits (proxy de contexte hydrogéologique)

Sans geojson : fallback = stats agrégées sur les puits du sous-bassin. Minimum 3–4 features
pour que le hub apporte quelque chose au-delà d'un simple agrégateur.

---

## 3. Tables d'arêtes à pré-calculer

Pré-calcul haversine sur les sources brutes (BallTree), stockage en `.parquet` :

| Fichier | Colonnes | Nb lignes | Cap |
|---|---|---:|---:|
| `edges_well_facility.parquet` | `well_idx, facility_idx, dist_km, cap` | ~16 k (3 km) | 1 km / 3 km |
| `edges_well_aqs.parquet` | `well_idx, aqs_idx, dist_km` | ~15 k | 10 km |
| `edges_well_well_near.parquet` | `src_idx, dst_idx, dist_km` | ~65 k paires | 1,5 km |

**Note :** les arêtes `well↔well` sont déjà calculées dans `src/graph.py`
(`knn_edges_km`, k=8, cap 1,5 km). Ne pas dupliquer — réutiliser.

Les arêtes `well→subbasin` sont construites à la volée depuis `sgma_subbasin_name`
(jointure directe, pas besoin de pré-calcul haversine).

---

## 4. Pipeline d'agrégation — étapes ordonnées

```
Étape 0 — Fixer la graine : SEED=42 partout
Étape 1 — Table puits
    - Charger CA-PFAS-ASGWS.parquet
    - Agréger au niveau puits (1 ligne / gm_well_id) :
        features numériques → médiane ; catégorielles → mode
    - Joindre well_hydraulic_gradient.csv (jointure parfaite sur gm_well_id)
    - Ajouter flow_dir_missing (bool)
    - → nodes_well.parquet

Étape 2 — Table facility (4 sources)
    - Charger geotracker + wwtp + dod_pfas_federal_sites_ca.csv + swis_landfills_ca.csv
    - Dédupliquer sur (lat,lon) arrondi 4 déc. dans chaque source
    - Concaténer ; facility_id préfixé par source (GEO_/WWTP_/DOD_/SWIS_)
    - Ajouter facility_type (7 niveaux), src_active, wwtp_major, dod_known
    - Filtrer bbox CA [32.5,42] × [-124,-114]
    - → nodes_facility.parquet (~2 560 sites)

Étape 3 — Table AQS
    - Charger aqs_ca_annual.parquet
    - Filtrer years 2015–2023
    - Pivot : monitor_id × param_name → annual_mean (moyenne pondérée completeness_pct)
    - Exclure SO₂ (89 % NA)
    - Ajouter indicateurs *_observed, imputer par plus-proche-station (BallTree)
    - → nodes_aqs.parquet

Étape 4 — Blocs spatiaux (CRITIQUE — à persister)
    - KMeans(n_clusters=8, random_state=42).fit(nodes_well[lat,lon] * [111,89])
    - → table well_blocks.parquet : gm_well_id, well_block (int8)
    - NE PAS recalculer à la volée — la graine doit être figée pour la reproductibilité
      et pour la cohérence des assertions 2-sauts entre runs

Étape 5 — Tables d'arêtes
    - edges_well_facility.parquet (BallTree, caps 1 et 3 km, stocker les deux)
    - edges_well_aqs.parquet (BallTree, cap 10 km)
    - NE PAS stocker les blocs dans les arêtes — le dédoublement @bloc
      se calcule à la volée au moment de la construction du HeteroData

Étape 6 — Validation pré-run (assertions obligatoires avant tout run long)
    - 0 well_block NA
    - Pour chaque hub dédoublé facility@b : {B(w) pour w voisin} = singleton → 0 violation
    - Pour chaque hub dédoublé aqs@b : idem
    - Rapporter Δ(AUC aléatoire − AUC spatiale) avec les nouveaux hubs activés (C-DELTA)
      → si Δ > référence (+0,17–0,20), fuite réintroduite → run BLOQUÉ
```

---

## 5. Sources manquantes — priorités de collecte

### 5.1 CRITIQUE — Géochimie redox GAMA (déjà sur disque, ~6 GB)

`ca-pfas-ml/data/raw/gama/*.csv` — contient les mesures chimiques multi-paramètres.

**Paramètres à extraire :** O₂ dissous, Eh (potentiel redox), NO₃, SO₄, Fe²⁺, Mn, NH₄, pH,
conductivité/TDS, **profondeur de puits** (GAMA allwells).

**Pourquoi :** Ces indicateurs contrôlent la mobilité des PFAS dans les aquifères
(conditions redox → transformation précurseurs → PFAS terminaux mobiles ; sorption dépend du pH ;
profondeur discrimine nappes superficielles vulnérables vs aquifères profonds protégés).
C'est la seule information mécaniste directement disponible et non incluse dans le parquet actuel.

**Garde-fou (fuite de sélection) :** vérifier que ces paramètres ne sont disponibles QUE pour
les puits déjà sous investigation PFAS (si mesurés en même campagne, risque de biais de sélection
→ auditer la corrélation avec T1a avant d'inclure).

### 5.2 UTILE — Débits de rejet WWTP (ECHO DMR, en ligne)

API EPA ECHO : `https://echo.epa.gov/` → DMR (Discharge Monitoring Reports) → débit journalier
moyen (MGD) par établissement NPDES. Un WWTP Major à 50 MGD contamine plus qu'un Minor à 0,1 MGD.

**Jointure :** via nom/ville/état (pas d'ID commun avec le fichier WWTP actuel → jointure floue).
Ajoute la feature `wwtp_flow_mgd` sur le nœud `facility`.

### 5.3 UTILE — Lithologie de subsurface (SSURGO/SoilGrids)

**SSURGO** (USDA/NRCS) ou SoilGrids : texture du sol (% sable/limon/argile), conductivité
hydraulique saturée, profondeur de la zone non-saturée. Ces paramètres contrôlent la vitesse de
percolation depuis la surface vers la nappe.

**Jointure :** par coordonnées (lat/lon des puits → extraction raster ou API SSURGO).

### 5.4 NICE-TO-HAVE — DWR piézométrie historique longue durée

La source actuelle (`dwr_periodic_gwl_recent.csv`) est un snapshot statique pré-agrégé sans
colonne date. Une série historique (10+ ans) permettrait un **gradient moyen robuste** au lieu
d'un snapshot bruité. Source : portail DWR CASGEM/SGMA.

---

## 6. Exigences méthodologiques (non négociables)

### Dataset et split

- **1 nœud = 1 puits** : agréger les 4,09 lignes/puits AVANT la construction du graphe.
  Ne jamais construire 1 nœud/ligne (fuite intra-puits).
- **`well_block` persisté** : calculé une fois (KMeans k=8, seed=42), stocké dans
  `well_blocks.parquet`, jamais recalculé à la volée. Le bloc conditionne toutes les assertions.
- **Dédoublement `hub@bloc`** : `copy_idx = base_entity_idx * 8 + block_id` (entier, vectorisable).

### Anti-fuite

- `LEAKAGE_BLOCKLIST` (96 cols actuelles) : à **re-scanner indépendamment** (corrélation + sémantique)
  avant chaque ajout de nouvelles colonnes. Ne pas se fier à la liste figée.
- Gradient hydraulique : features légitimes (DWR, pas PFAS). Fit scaler sur train seul.
- AQS annual_mean : ne jamais utiliser une moyenne englobant le futur du prélèvement (risque
  fuite temporelle pour les ~570 prélèvements antérieurs à 2019).
- Géochimie GAMA : auditer la corrélation OOF à T1a avant inclusion ; vérifier l'absence de
  biais de sélection (puits sous investigation PFAS only).

### Ablation des relations (stratégie additive)

Pour éviter 2⁴ = 16 runs, utiliser un chemin **additif unique** :

```
Run 1 : well↔well near + subbasin (baseline graphe validé)
Run 2 : +facility (cap 3 km)
Run 3 : +aqs_station (cap 10 km)
Run 4 : +flows_to (cap 3 km, après feature gradient validée)
```

Rapporter à chaque cran : AUC spatiale, Δ(rand-spatial), IC bootstrap, test Wilcoxon apparié.
Un Δ qui remonte > référence (+0,17–0,20) signale une fuite réintroduite → **run REFUSÉ**.

### Faisabilité Colab

- **~14–16 k nœuds** après dédoublement (worst case ~22 k), ~60–80 k arêtes dirigées.
- **< 1 GB** activations GPU (hidden=64, 2 couches, full-batch). Faisable sans mini-batching.
- Architectures à tester : HeteroConv(SAGEConv mean) [~33k params] → RGCN [~16k] → HGT [~75k].
  HGT désormais justifié (≥ 3 relations sémantiquement distinctes avec facility + aqs + subbasin).

---

## 7. Checklist de validation avant run long

```
[ ] nodes_well.parquet — 11 333 lignes, 0 NA gm_well_id, gradient joint
[ ] nodes_facility.parquet — ~2 560 sites (4 sources), 0 NA lat/lon, facility_type ∈ 7 niveaux, dod_known/wwtp_major présents
[ ] nodes_aqs.parquet — 302–305 stations, SO₂ exclu, *_observed présent
[ ] well_blocks.parquet — 11 333 lignes, int8 [0–7], seed=42 figée
[ ] edges_well_facility.parquet — caps 1 km ET 3 km stockés
[ ] edges_well_aqs.parquet — cap 10 km
[ ] Dédoublement facility@bloc : assertion 0 violation (tous voisins = même bloc)
[ ] Dédoublement aqs@bloc : assertion 0 violation
[ ] Δ(rand-spatial) recalculé avec les hubs → ≤ référence +0,17–0,20
[ ] Smoke-test CPU vert (< 3 min, courbes perte+AUC non vides, §3.8 CLAUDE.md)
[ ] Scan anti-fuite des nouvelles colonnes (corrélation OOF)
```

---

*Ce fichier est le point d'entrée pour toute nouvelle expérience HGT. Les résultats de modélisation
sont dans `experiments/hgt_rgcn_t1/REPORT.md` et les protocoles validés dans
`experiments/hgt_rgcn_t1/eval_hetero5_spatial.md`.*
