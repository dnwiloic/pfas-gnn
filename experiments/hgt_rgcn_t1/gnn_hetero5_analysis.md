# Rendre rigoureux le graphe hétérogène à 5 types de nœuds (notebook 04)

Analyse GNN approfondie. Auteur : agent GNN. Date : 2026-06-23.
Objectif : CONSERVER la structure à 5 types de nœuds (`sample`, `env`, `facility`,
`water`, `geo_cluster`) et la rendre mécaniste + spatialement honnête.

Faits de code confirmés (notebook 04, lignes de la version aplatie du `.ipynb`) :
- `FEATURE_CATEGORIES` l.89-98 : `Environmental` (6), `Facility_Proximity` (6),
  `Water_Quality` (6 cocontam), `Geospatial = ['latitude','longitude']` l.97.
- Split `train_test_split(..., stratify=y)` l.116-119 — **aléatoire**.
- `fit_graph_transformers` l.166-173 : KMeans `n_clusters=min(20, len/50)` sur lat/lon,
  fit train (bon réflexe l.170).
- `create_heterogeneous_graph` l.175-197 : `sample.x` = TOUTES les features (l.177) ;
  arêtes identité `ei = stack([arange(n), arange(n)])` l.179, dupliquées pour
  env/facility/water (l.187-188) ; geo_cluster : `sample→geo_cluster` via `km.predict`
  l.195-196, nœud cluster = centroïde l.193.
- Architecture : 2× `HGTConv(hidden, hidden, metadata, heads=4)` l.218.

Faits dataset confirmés (CA-PFAS-ASGWS.parquet, 46 338 lignes / 11 333 puits) :
- `nearest_geotracker_type` : 4 valeurs (Chrome Plater, Bulk Terminal, Airport, Refinery), 0 % NA.
- `sgma_subbasin_name` : **237** valeurs, 6,1 % NA au niveau ligne ; au niveau PUITS
  9 858/11 333 ont un sous-bassin, médiane 10 puits/sous-bassin, max 636.
- `dwr_basin` : 239, `dwr_region` : 10, `county` : 58.
- `well_depth_ft` : **94,5 % manquant** → INUTILISABLE comme nœud aquifère.
- Aucune colonne `aquifer_type`, `well_depth_category`, ni d'ID/coordonnées
  d'installation geotracker individuelle.

---

## Axe 1 — Reformuler les arêtes identité en arêtes réellement relationnelles

Principe : une arête ne porte de l'information QUE si plusieurs `sample` partagent le
MÊME nœud du type cible. Les arêtes identité (l.179, `arange↔arange`) ne partagent rien :
elles re-projettent un sous-vecteur de `sample.x` puis le recollent — un MLP par blocs.
Toute reformulation doit créer une **vraie entité partagée**, et on doit distinguer
*entité physique partagée* (légitime) de *clique par catégorie* (fuite déguisée, refusée
dans pfas-gnn : la clique sous-bassin complète a été rejetée, Cramér V(block,subbasin)=0,98).

### Nœud `facility` — VERDICT : REFORMULER (avec garde-fou) ou SUPPRIMER

**Pas d'entité réelle disponible.** Le dataset n'a ni ID ni coordonnées d'installation
individuelle ; seules des features agrégées au puits (`dist_geotracker_km`,
`n_geotracker_within_{1,3,10,50}km`, `nearest_geotracker_type`). Un vrai nœud `facility`
(un site = un nœud relié à tous les puits dans son rayon de panache capé) — la seule
construction mécaniste défendable — est IMPOSSIBLE sans la source geotracker brute.

**Option A (proposée dans la mission) : 4 nœuds `facility_type`** partagés par tous les
samples dont c'est le `nearest_geotracker_type`. C'est une **clique par catégorie
déguisée** : relier tous les puits « Airport » entre eux via un nœud central crée une
composante connexe de ~7 520 lignes / des milliers de puits, dispersés sur toute la
Californie. Le message-passing y diffuse la prévalence moyenne de la catégorie « Airport »
sur des puits sans aucune relation hydrogéologique. C'est exactement le motif refusé en
pfas-gnn (clique par type = signal-free / fuite, eval §2.2-2.3). **Pire que l'identité** :
l'identité n'ajoutait rien, la clique de type ajoute un biais de catégorie + une voie de
fuite (sous split aléatoire, un test « Airport » hérite de la cible des train « Airport »).
À REFUSER tel quel. `nearest_geotracker_type` doit rester une **feature catégorielle** de
`sample` (ce qu'elle est déjà via l'embedding HGT), pas un nœud.

**Option B (recommandée) : nœud `facility` = grappe spatiale de puits autour d'un
foyer de sources.** Au lieu d'un type, partager un nœud quand les puits sont *proches ET
exposés au même type de source*. Concrètement : nœud `source_hub` = intersection
(sous-bassin × type de source dominant), relié seulement aux puits du sous-bassin partageant
le `nearest_geotracker_type`. Cela borne la portée (intra-sous-bassin) et capte « même
foyer de contamination plausible ». Reste un proxy faute de coordonnées de site, mais
mécaniquement supérieur à la clique de type. **Verdict pragmatique : si on garde 5 types,
faire `facility` = Option B avec cap sous-bassin ; sinon SUPPRIMER et laisser les features
geotracker sur `sample`.**

### Nœud `water` — VERDICT : SUPPRIMER (pas d'entité d'aquifère disponible)

Le groupe `Water_Quality` (l.95-96 : `cocontam_tds/no3n/so4/tce/pce/mtbe`) décrit des
**propriétés du point de nappe**, pas une entité partagée. Pour un vrai nœud « compartiment
hydrogéologique » il faudrait `aquifer_type`/`well_depth_category` — **absents** (`well_depth_ft`
94,5 % NA, aucune colonne aquifère). On NE PEUT PAS construire un nœud water mécaniste.

Deux issues :
1. **SUPPRIMER le type `water`** : les cocontaminants redeviennent des features de `sample`.
   C'est la solution honnête. Les cocontaminants (NO3, TDS, TCE…) sont des co-traceurs
   hydrogéochimiques utiles comme features, pas comme topologie.
2. **Réaffecter** : si on tient à un nœud hydrogéologique partagé, c'est `geo_cluster`
   reformulé en sous-bassin (Axe 1, geo_cluster) qui joue ce rôle — pas besoin d'un type
   `water` redondant. Ne PAS dupliquer le compartiment hydro sur deux types.

### Nœud `env` — VERDICT : REFORMULER en zone partagée à granularité grossière, sinon SUPPRIMER

Le groupe `Environmental` (l.90-91 : pluie, runoff, humidité sol, sand/silt) est local au
puits. Pour le rendre relationnel il faut une **zone réellement partagée**. Candidats :
- **`dwr_region`** (10 valeurs) : trop grossier → 10 nœuds, ~1 000 puits chacun = quasi-
  cliques régionales = fuite. REFUSER.
- **`county`** (58 valeurs, ~195 puits/comté) : administratif, pas hydro, mais moins
  diffusant que dwr_region. Discutable.
- **Cluster AQS partagé** : plusieurs puits hérités du même enregistrement de station EPA
  AQS forment une vraie entité partagée (même mesure d'air). Si la jointure AQS attache
  un `aqs_station_id` (à vérifier ; non exposé dans `config.py`), ce serait le seul nœud
  `env` *physiquement* partagé légitime. Sans ID station, on retombe sur une grille spatiale
  = même problème que geo_cluster.

**Verdict : SUPPRIMER `env` par défaut** (features → `sample`). Le conserver UNIQUEMENT si
un identifiant de station AQS partagé existe réellement dans les données brutes ; sinon
toute zone « env » est une grille spatiale redondante avec geo_cluster.

### Nœud `geo_cluster` — VERDICT : REFORMULER en sous-bassin SGMA (cœur de la rigueur)

C'est le seul canal relationnel actif (l.193-196) et le plus dangereux : KMeans 20 sur
lat/lon = pavé géographique non hydrogéologique → diffuse la cible par pavé, traverse les
divides, non capé. C'est le réencodage de carte refusé en pfas-gnn.

**Remplacer le centroïde KMeans par le sous-bassin SGMA** (`sgma_subbasin_name`, 237
valeurs, médiane 10 puits/sous-bassin). Avantages : (1) unité hydrogéologique réelle
(respecte les limites de bassin) ; (2) déjà l'objet utilisé dans pfas-gnn pour contraindre
les arêtes `same_subbasin_knn` ; (3) granularité ~50× plus fine que 20 KMeans → diffusion
bornée à un compartiment plausible au lieu d'un quart de la Californie.

**MAIS** : un nœud sous-bassin reliant TOUS ses puits = clique de sous-bassin, REFUSÉE en
pfas-gnn (81,6 % de ses paires > 5 km). Donc deux gardes obligatoires :
- **Cap de distance intra-cluster** : ne relier au nœud sous-bassin que les puits à ≤ 2 km
  les uns des autres (réplique exactement `knn_edges_intra_subbasin`, cap 2 km). Le nœud
  sous-bassin devient un *hub local* (un par poche dense), pas une clique géante. Le max
  636 puits/sous-bassin (Central Valley) impose ce cap, sinon un seul hub diffuse sur 636
  puits dispersés.
- **Coupe inter-blocs du hub** (cf. Axe 3) : un hub ne doit jamais relier train et test.

Granularité raisonnable pour 11 333 puits : **les 237 sous-bassins SGMA** (et non 20).
1 475 puits sans sous-bassin (13 %) restent isolés sur cette relation (comportement
identique à pfas-gnn, documenté). Pour les sous-bassins denses (>~100 puits), splitter par
sous-grappe spatiale (un hub par grappe ≤ 2 km) pour éviter une clique résiduelle.

**Et lat/lon ?** Les SORTIR de `sample.x` (l.97, l.177) — `include_location=False` comme
dans pfas-gnn (C-LOC.1). La géographie n'entre que par la topologie (hub sous-bassin),
jamais comme feature. Sinon double canal géographique (features + topologie) = fuite empilée
(diagnostic hydro §2). Garder une ablation explicite « avec/sans lat-lon » pour mesurer
l'oracle géographique.

**Synthèse Axe 1 — graphe à 5 types rendu rigoureux :**
| Type | Verdict | Forme rigoureuse |
|---|---|---|
| `sample` | GARDER mais = PUITS, pas ligne | 1 nœud/puits (agréger les lignes), cf. graph.py |
| `facility` | REFORMULER ou SUPPRIMER | hub (sous-bassin × type source), cap 2 km — sinon feature |
| `water` | SUPPRIMER | cocontam → features ; pas d'aquifère disponible |
| `env` | SUPPRIMER (sauf ID AQS réel) | features → sample ; grille = redondance geo_cluster |
| `geo_cluster` | REFORMULER | hub `sgma_subbasin_name` cap 2 km + coupe inter-blocs, lat/lon hors features |

Note : `sample` = ligne d'échantillonnage est un défaut structurel transverse. Plusieurs
lignes d'un même puits deviennent des nœuds distincts sans arête (continuité perdue).
**Agréger au puits** (comme `graph.well_table`) est la première correction ; sinon les hubs
relient des lignes du même puits entre elles, gonflant artificiellement le degré.

---

## Axe 2 — HGTConv est-il le bon choix pour CE graphe ?

**Diagnostic d'expressivité.** HGT (Hu 2020) maintient des projections Q/K/V **par
(type_src, type_dst, relation)** + des matrices d'attention par relation + des priors
`μ[rel]`. Il est justifié quand les relations sont **sémantiquement très hétérogènes**
(p. ex. graphe académique auteur/papier/lieu). Ici, après reformulation, il ne reste
réellement qu'**UNE relation informative** (puits↔hub sous-bassin, éventuellement + hub
source). Les autres étaient des identités. HGT est donc **surparamétré** : 5 types × jusqu'à
4 relations → des dizaines de matrices Q/K/V apprises sur un graphe quasi-étoile à une seule
relation utile. Sur 11 333 nœuds c'est un budget de paramètres élevé pour un signal faible
→ sur-apprentissage (cohérent avec HGT seul 0,845 < XGBoost 0,952 en aléatoire, et chute
attendue plus forte en spatial).

**Recommandation chiffrée (ordres de grandeur, hidden=64, 2 couches) :**
- **HGTConv** : ~par couche, 3 projections (Q,K,V) × n_types matrices 64×64 + attention par
  relation ≈ O(n_types × 64² + n_rel × 64²). Avec 5 types/4 rel ≈ ~9·64²·2 ≈ ~75 k params de
  conv (hors heads/skip) — **le plus lourd**, justifié seulement si ≥3 relations distinctes.
- **HeteroConv({rel: SAGEConv})** : une `SAGEConv` (2 matrices 64×64) par relation. Avec
  2 relations ≈ 4·64²·2 ≈ ~33 k. **Plus simple, le bon défaut ici** : peu de types, graphe
  petit, agrégation mean robuste, inductif natif (généralise aux puits non vus). C'est
  exactement `hetero_sage`/`hetero_sage_v1` de `gnn_hetero_t1.py` (l.232-236) — déjà éprouvé.
- **RGCN** : 1 matrice 64×64 par relation (décomposable basis/block). ≈ 2·64²·2 ≈ ~16 k.
  **Le plus parcimonieux**, comparable directement à pfas-gnn (`gnn_hetero_t1.py` l.229-231).
  Risque : pas d'attention → toutes les arêtes d'une relation pèsent pareil (acceptable, le
  cap 2 km borne déjà la portée).

**Pour un graphe étoile (centre = hub, feuilles = puits)** l'attention HGT n'apporte rien :
un puits n'a qu'un voisin (son hub) ; il n'y a rien à pondérer entre voisins. L'attention
est utile quand un nœud agrège PLUSIEURS voisins hétérogènes — pas le cas d'une feuille
d'étoile. **SAGEConv mean ou RGCN suffit et est plus stable.**

**Verdict Axe 2 :**
1. **HeteroConv(SAGEConv, mean), aggr='sum'** comme architecture de référence (parcimonie,
   inductif, déjà implémenté `hetero_sage_v1` avec DropEdge + GraphNorm + neighbor sampling).
2. **RGCN** comme baseline relationnelle parcimonieuse (comparaison directe pfas-gnn).
3. **HGT** seulement comme point de comparaison « expressif », en SACHANT qu'il est
   surparamétré pour ce graphe ; ne pas le présenter comme le modèle principal. Si on le
   garde, réduire à 1 couche / heads=2 / hidden=32 pour limiter le sur-apprentissage et
   surveiller le gap fit−val (le module `train_diag` de `gnn_hetero_t1.py` le journalise déjà).

---

## Axe 3 — CV spatiale par blocs sur graphe hétérogène avec nœuds partagés

Principe : seuls les `sample`/puits sont spatiaux ; le bloc s'applique au PUITS. Les nœuds
partagés (hub sous-bassin, hub source) ne portent pas de coordonnées mais **transportent
des messages entre puits de blocs différents** → ils rouvrent la fuite si on ne coupe pas.

**Le danger précis du chemin à 2 sauts.** Dans une étoile `puits_train → hub → puits_test`,
même avec une coupe naïve « arêtes puits↔puits inter-blocs », l'information passe par le hub
en 2 sauts (et une 2e couche HGT/SAGE suffit). Couper seulement les arêtes directes ne suffit
PAS dès qu'un nœud intermédiaire existe.

**Protocole de coupe, par relation (à implémenter en PyG) :**

1. **Relations puits↔puits directes** (`near` k-NN, `same_subbasin_knn`) : coupe identique à
   `gnn_hetero_t1.py` — `cut_cross_block` par relation AVANT symétrisation, assert 0
   résiduel (C-SPAT.2/5). Déjà en place.

2. **Relations via un nœud-hub partagé** (puits↔hub sous-bassin, puits↔hub source). Deux
   stratégies, par ordre de rigueur croissante :
   - **(a) Dédoubler le hub par bloc.** Pour chaque hub `h` et chaque bloc `b`, créer un
     nœud-hub distinct `h@b` ne reliant QUE les puits du bloc `b`. Un message ne peut alors
     jamais traverser `h` d'un bloc à l'autre — la fuite est *structurellement* impossible,
     sans assertion fragile. C'est la **méthode recommandée** (équivalent hétérogène de la
     coupe d'arêtes ; en PyG, suffit de remapper l'index du hub par `(hub_id, block_id)`).
     Coût : multiplie le nombre de nœuds-hubs par ≤ n_blocs (8) — négligeable (237×8 hubs).
   - **(b) Inductif strict (à la C-SPAT.4).** À l'entraînement, ne garder dans le graphe que
     les arêtes puits↔hub dont le puits est TRAIN ; au scoring, rebrancher les puits TEST sur
     leurs hubs pour qu'ils n'agrègent QUE depuis des voisins TRAIN. Un puits test reçoit le
     message du hub (résumé des puits train du même bloc-test) — donc le hub doit lui-même
     être dédoublé par bloc, sinon il mélange. **(a) et (b) se combinent** : dédoubler par
     bloc + message-passing train-only = la garantie de `gnn_hetero_t1.py` étendue aux hubs.

3. **Assertion de validation** (obligatoire avant run long, à faire valider par
   `eval-methodologist`) : pour CHAQUE relation, après coupe/dédoublement, compter les
   paires (src,dst) dont les blocs des deux puits *atteignables en 2 sauts* diffèrent → doit
   être 0. Concrètement : aucun hub ne doit avoir des voisins puits de blocs différents.
   `assert all(node_block[neigh] constant) for neigh in hub.neighbours`.

4. **Refit du graphe par pli.** KMeans/hubs refités DANS le train de chaque pli (le notebook
   le fait déjà sur train global l.170 ; l'étendre aux 8 plis). Les centroïdes/hubs ne voient
   jamais les coordonnées test.

**Protocole de coupe par relation — tableau :**
| Relation | Coupe |
|---|---|
| puits↔puits `near` (cap 1,5 km) | `cut_cross_block` + assert 0 (existant) |
| puits↔puits `same_subbasin_knn` (cap 2 km) | `cut_cross_block` séparé + assert 0 (existant) |
| puits↔hub sous-bassin | **dédoubler hub par bloc** + inductif train-only + assert hub-voisins même bloc |
| puits↔hub source | idem hub sous-bassin |

---

## Axe 4 — Graphe hétérogène à 5 types vs graphe puits-puits (pfas-gnn)

| Dimension | pfas-gnn (puits-puits multi-rel) | notebook 04 rigoureux (hétéro 5 types) |
|---|---|---|
| Nœud | puits (entité physique) | puits (après correction sample→puits) |
| Relations réellement relationnelles | `near` (1,5 km) + `same_subbasin_knn` (2 km) | hub sous-bassin (≡ subbasin_knn agrégé) + hub source |
| env/facility/water | features du puits | à SUPPRIMER (water/env) ou hub source (facility) |
| Architecture | RGCN / HGT / hetero_sage sur 1 type, 2 rel | HeteroConv/RGCN sur ≥2 types |
| Coupe spatiale | `cut_cross_block` par rel, assert 0 (mûr) | + dédoublement hub par bloc (à implémenter) |
| AUC spatiale mesurée | 0,618–0,647 (réel, mesuré) | non mesuré ; estimé ≤ 0,74 SI le hub fuit, sinon ≈ ordre pfas-gnn |
| Interprétation | voisinage spatial capé | hub sous-bassin (compartiment hydro) |
| Risque fuite résiduelle | faible (cap + coupe éprouvés) | **plus élevé** (hub = 2-sauts, fuite si mal coupé) |

**Quand le graphe hétérogène 5 types est-il STRICTEMENT supérieur ?** Seulement si un nœud
intermédiaire encode une **entité physique réellement partagée que la relation puits-puits
ne peut pas exprimer**. Deux cas concrets ici :
1. **Un vrai nœud `facility`** (site geotracker avec coordonnées) reliant les puits sous le
   MÊME panache : ça exprime « partage d'une source », impossible à dire avec une simple
   arête de proximité puits-puits. → supériorité réelle, mais **données absentes**.
2. **Un nœud AQS station partagé** (même mesure d'air pour N puits) : entité partagée
   légitime, exprime une co-exposition non capturée par la distance. → supériorité réelle
   **si `aqs_station_id` existe** (à vérifier).

**Sinon** (cas actuel : hub = sous-bassin), le nœud-hub sous-bassin est *mathématiquement
équivalent* à l'agrégation `same_subbasin_knn` de pfas-gnn, en moins direct (2 sauts au lieu
de 1) et plus risqué côté fuite. Le graphe puits-puits multi-relationnel de pfas-gnn est
alors **préférable** : même information mécaniste, coupe spatiale plus simple et déjà mûre,
moins de paramètres.

**Tableau décisionnel :**
| Situation | Préférer |
|---|---|
| Coordonnées de sites geotracker disponibles | **Hétéro 5 types** (vrai nœud facility = source partagée) |
| ID station AQS partagé disponible | **Hétéro** (nœud env partagé légitime) |
| Seulement features agrégées au puits (cas actuel) | **Puits-puits multi-rel (pfas-gnn)** — le hub sous-bassin n'apporte rien de neuf |
| Besoin de complétion de la matrice T2 lacunaire | **Bipartite puits×PFAS** (orthogonal, cf. graph.py `build_bipartite_well_analyte`) |

---

## Recommandation opérationnelle (pour conserver les 5 types honnêtement)

1. **Corriger le nœud** : `sample` = puits agrégé (réutiliser `graph.well_table`).
2. **Sortir lat/lon des features** (`include_location=False`).
3. **`geo_cluster` = hub `sgma_subbasin_name`** (237), cap intra-hub 2 km, splitter les
   sous-bassins denses ; **supprimer** `water` et `env` (features → puits) ; `facility` =
   hub (sous-bassin × type source) cap 2 km, ou supprimé.
4. **Architecture = HeteroConv(SAGEConv, mean)** par défaut (parcimonie, inductif),
   RGCN en comparaison, HGT seulement en point « expressif » réduit (1 couche, heads=2).
5. **CV spatiale k=8** (réutiliser `splits.spatial_block_folds`) + **dédoublement des hubs
   par bloc** + inductif train-only + assertion « hub-voisins même bloc » ; rapporter
   AUC aléatoire ET spatiale + Δ.
6. **Smoke-test CPU** (≈400 puits, 1 pli, ~15 époques) vérifiant : hubs construits,
   0 hub inter-blocs, courbes perte/AUC train+val non vides (§3.8), figures écrites.
7. **Faire valider le protocole de coupe par hub par `eval-methodologist`** avant tout run
   long Colab.

**Conclusion honnête.** Avec les données disponibles, rendre le graphe 5 types rigoureux le
fait *converger* vers le graphe puits-puits multi-relationnel de pfas-gnn (le hub sous-bassin
≡ `same_subbasin_knn`). Le 5-types ne devient strictement supérieur que si l'on injecte une
entité physique réellement partagée (coordonnées de sites geotracker, ou ID de station AQS)
absente du parquet actuel. La valeur du travail est donc : (a) prouver que la version
rigoureuse du 5-types s'aligne sur le plancher honnête ~0,62–0,65 (et non 0,95), confirmant
que le score élevé était de l'autocorrélation spatiale ; (b) garder l'architecture hétérogène
prête à exploiter de vraies entités partagées si les sources brutes sont ajoutées.
