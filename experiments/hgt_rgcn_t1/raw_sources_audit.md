# Audit des sources brutes — Entités physiques partagées pour graphe hétérogène

Auteur : analyse directe (data-analyst). Date : 2026-06-23.
Dépôts inspectés : `pfas-gnn/` et `ca-pfas-ml/`.

---

## Verdict global : les entités physiques partagées EXISTENT

Contrairement à la conclusion initiale du gnn-researcher (basée sur le parquet agrégé seul),
`ca-pfas-ml/data/raw/` contient trois sources brutes qui fournissent de vraies entités
physiques partageables pour le graphe hétérogène.

---

## 1. Geotracker — Sites de contamination PFAS (DONNÉES TROUVÉES)

**Fichier :** `ca-pfas-ml/data/raw/contamination/geotracker_pfas_investigation_sites.csv`

| Attribut | Valeur |
|---|---|
| Nb de sites | **463** |
| Colonnes | `fid`, `global_id` (unique), `name`, `address`, `city`, `site_type`, `latitude`, `longitude` |
| NA coordonnées | **0** |
| Clé unique | `global_id` (463 valeurs distinctes, ex: `T10000012764`) |

**Distribution par type :**
| site_type | Nb |
|---|---:|
| Chrome Plater | 271 |
| Bulk Terminal | 122 |
| Refinery | 40 |
| Airport | 30 |

**Connectivité puits↔sites (haversine, ~37°N) :**
| Cap (km) | Puits connectés | % | Arêtes totales |
|---:|---:|---:|---:|
| 1 | 2 147 | **18.9 %** | 4 153 |
| 3 | 4 029 | 35.6 % | 13 719 |
| 5 | 5 214 | 46.0 % | 27 967 |
| 10 | 6 832 | 60.3 % | 82 635 |

**Conclusion :** À cap=1 km, 4 153 arêtes puits→site sur 11 333 puits (18.9 %).
C'est **sparse** (pas une clique), chaque puits est relié à 0–quelques sites dans son voisinage immédiat.
Ce n'est pas la "clique par type" refusée en pfas-gnn (où TOUS les puits d'un type étaient reliés) —
ici les arêtes sont capées par la distance physique. Mécaniquement défendable : un site à <1 km
peut contaminer la nappe par écoulement direct. Les 81.1 % de puits sans site à <1 km restent
des nœuds isolés sur cette relation (comportement inductif naturel).

---

## 2. AQS — Stations de surveillance de la qualité de l'air (DONNÉES TROUVÉES)

**Fichier :** `ca-pfas-ml/data/raw/environment/aqs/aqs_ca_annual.parquet`

| Attribut | Valeur |
|---|---|
| Nb de stations uniques | **305** (`monitor_id` unique, ex: `060655001`) |
| Colonnes | `monitor_id`, `latitude`, `longitude`, `param_code`, `param_name`, `annual_mean` |
| Polluants | pm25, pm10, no2, so2, wind_speed, humidity, ozone, co |

**Connectivité puits↔stations :**
| Cap (km) | Puits connectés | % | Stations partagées |
|---:|---:|---:|---:|
| 5 | 4 200 | 37.1 % | 219 / 234 |
| 10 | 7 654 | **67.5 %** | 268 / 281 |
| 20 | 9 985 | 88.1 % | 292 / 301 |
| 30 | 10 816 | 95.4 % | 302 / 304 |

À cap=10 km, **268 des 281 stations connectées sont réellement partagées** (plusieurs puits→même station).
C'est une vraie entité partagée : la même mesure AQS s'applique à plusieurs puits dans la zone de la station.

**Conclusion :** Cap recommandé 10–20 km (zone d'influence typique d'une station AQS).
Les arêtes puits→station sont mécanistiquement justifiées : les puits d'une zone partagent
le même niveau d'exposition atmosphérique (dépôts PFAS atmosphériques, co-contaminants).

---

## 3. WWTP — Stations d'épuration (source PFAS biosolides) (DONNÉES TROUVÉES)

**Fichier :** `ca-pfas-ml/data/raw/contamination/wwtp_epa_frs_ca.csv`

| Attribut | Valeur |
|---|---|
| Nb de WWTP | **350** |
| Colonnes | `latitude`, `longitude`, `name`, `city`, `site_type` (WWTP Major / Minor), `status` |
| NA coordonnées | **0** |

**Total sources PFAS combinables :** 463 (geotracker) + 350 (WWTP) = **813 sites**

**Rôle mécaniste :** Les WWTP sont une source PFAS majeure via les biosolides épandus
en agriculture et l'infiltration. Complémentaires aux sites geotracker (AFFF/solvants)
pour couvrir les 3 voies de contamination principales (AFFF, industrie, biosolides).

---

## 4. Gradient hydraulique (DÉJÀ CALCULÉ — prêt à l'emploi)

**Fichier :** `ca-pfas-ml/data/processed/well_hydraulic_gradient.csv`

| Attribut | Valeur |
|---|---|
| Nb de puits | **11 333** (100 % de couverture, match exact avec le parquet) |
| Colonnes | `gm_well_id`, `latitude`, `longitude`, `hydr_head_m`, `hydr_grad_mag`, `hydr_grad_mag_permil`, `flow_dir_sin`, `flow_dir_cos`, `dist_nearest_gwl_km` |
| NA flow_dir | **92 / 11 333 (0.8 %)** |

**Signification :**
- `flow_dir_sin` / `flow_dir_cos` : direction de l'écoulement souterrain (angle en 2D)
- `hydr_grad_mag_permil` : magnitude du gradient (‰)
- `dist_nearest_gwl_km` : distance à la mesure DWR la plus proche (médiane ~1.8 km)

**Usage :** Ces colonnes permettent de construire une arête **`flows_to` orientée** entre
puits voisins : puits A → puits B si B est dans la direction d'écoulement depuis A (angle ≤ θ°)
ET à distance ≤ cap_km. C'est l'arête mécaniste recommandée par l'hydro-expert.

**Source des données DWR :** `ca-pfas-ml/data/raw/hydro/dwr_periodic_gwl_recent.csv` (765 Ko)

---

## 5. Autres sources (contexte)

- **`data/raw/environment/sgma_basins.geojson`** (29 Mo) : polygones SGMA officiels — permet
  de faire la jointure spatiale puits↔sous-bassin plutôt que de se fier à la colonne texte.
- **`data/raw/gama/`** : données GAMA brutes (~5,9 Go) — source originale des mesures PFAS
  et des métadonnées de puits (dwr, ddw, localgw, usgsnwis).
- **`data/raw/environment/gldas_ca_monthly.parquet`** (12 Mo) : grille NASA GLDAS mensuelle
  (sol, humidité) — source des features NLCD/environnementales.
- **`data/cache/noaa/`** : données NOAA (précipitations, température) par station GHCND —
  potentiellement une autre source de nœuds atmosphériques si GHCND_ID disponible.

---

## Synthèse décisionnelle — Architecture graphe hétérogène révisée

| Type de nœud | Source brute | Nb entités | Entité partagée ? | Verdict |
|---|---|---:|---|---|
| `well` | parquet principal | 11 333 | — | **GARDER** (1 nœud/puits) |
| `facility` | geotracker_pfas_investigation_sites.csv + wwtp | 463 + 350 = **813** | OUI (plusieurs puits → même site) | **GARDER** (cap 1–3 km) |
| `aqs_station` | aqs_ca_annual.parquet | **305** | OUI (268/281 partagées à 10 km) | **GARDER** (cap 10–20 km) |
| `subbasin` | sgma_subbasin_name (ou geojson) | 237 | OUI (médiane 10 puits) | **GARDER** (hub hydrogéo, cap 2 km) |
| `water` | — | — | NON (aquifère non disponible) | **SUPPRIMER** |

**Arêtes à implémenter :**
| Relation | Source | Cap | Nb arêtes estimé | Mécanisme |
|---|---|---:|---:|---|
| `well→well` `near` | k-NN haversine | 1.5 km | ~existant pfas-gnn | voisinage spatial |
| `well→well` `flows_to` | gradient hydraulique | 1.5–3 km + angle | à calculer | transport orienté |
| `well→facility` | geotracker + WWTP | 1–3 km | 4 153–13 719 | source de contamination |
| `well→aqs_station` | aqs_ca_annual.parquet | 10–20 km | 7 654–9 985 | co-exposition atmosphérique |
| `well→subbasin` | sgma_subbasin_name | hub, cap 2 km | ~existant pfas-gnn | compartiment hydrogéologique |

**Prochaines étapes :**
1. L'eval-methodologist doit valider le protocole de coupe spatiale pour `facility` et `aqs_station`
   (en particulier : un site geotracker entre deux blocs — ses arêtes doivent-elles être coupées ?)
2. Implémenter dans `src/` les fonctions `build_facility_edges(wells_df, geo_df, cap_km)` et
   `build_aqs_edges(wells_df, aqs_df, cap_km)`.
3. Le gradient hydraulique (`flow_dir_sin/cos`) peut être ajouté comme feature de nœud dans `pfas-gnn`
   immédiatement (99.2 % de couverture, match parfait sur `gm_well_id`).
