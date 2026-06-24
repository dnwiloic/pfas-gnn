# Validation du protocole de coupe spatiale — nœuds `facility` et `aqs_station` (graphe hétérogène 5 types)

Méthodologiste : eval-methodologist. Date : 2026-06-24.
Cible : T1a (EPA 2024), prévalence ~44,5 %. CV spatiale k=8 blocs KMeans sur coords puits.
Inflation spatiale de référence (split aléatoire − split spatial) : **~0,17–0,20 AUC** (mesurée phase 1).
Portée d'autocorrélation spatiale de T1a : **~5 km** (pfas-gnn).

Tous les chiffres ci-dessous sont **recalculés indépendamment** sur les sources brutes
(`geotracker_pfas_investigation_sites.csv`, `wwtp_epa_frs_ca.csv`, `aqs_ca_annual.parquet`,
`well_hydraulic_gradient.csv`), pas récités de l'audit.

---

## VERDICT GLOBAL : VALIDÉ SOUS CONDITIONS

La construction des vrais nœuds `facility` et `aqs_station` (entités physiques avec
coordonnées individuelles) **lève le blocage C-NODE.1/C-NODE.2** de `eval_validation.md` :
ce ne sont plus des cliques par catégorie ni des agrégats fabriqués, mais des entités
réellement partagées et localisées. C'est mécanistiquement légitime ET cela rend le graphe
5 types **strictement supérieur** au puits-puits (cas 1 et 2 du tableau décisionnel de
`gnn_hetero5_analysis.md` Axe 4 : coordonnées geotracker disponibles + station AQS partagée).

**MAIS** : le nœud-hub partagé crée un chemin de fuite à 2 sauts `well_train → entité → well_test`.
La mesure quantitative ci-dessous montre que le risque est **négligeable pour `facility` à cap≤1 km**
(0 entité multi-blocs) mais **réel et croissant pour `aqs_station`** (7,8 % des stations à 10 km,
15,6 % à 20 km touchent ≥2 blocs). La validation est donc conditionnée à la stratégie de coupe par
relation détaillée en Q2 et au respect des caps de Q5.

---

## Données empiriques recalculées (fondement de tout le verdict)

**Blocs KMeans k=8 (coords puits, scale [111,89]) — rayons réels :**

| bloc | n puits | rayon moyen | rayon p90 | rayon max |
|---:|---:|---:|---:|---:|
| 0 | 905 | 59,6 km | 98,2 | 207,4 |
| 1 | 982 | 58,8 | 111,5 | 164,3 |
| 2 | 3335 | 38,7 | 83,9 | 134,6 |
| 3 | 818 | 96,2 | 184,4 | 262,7 |
| 4 | 1562 | 59,8 | 107,2 | 184,7 |
| 5 | 1444 | 71,6 | 123,7 | 234,0 |
| 6 | 1842 | 59,9 | 86,8 | 141,1 |
| 7 | 445 | 57,7 | 88,6 | 141,4 |

Rayon de bloc ~40–100 km ≫ cap d'arête (1–20 km) ≫ portée d'autocorrélation (~5 km).
Donc une arête courte reste presque toujours **intra-bloc** ; le risque ne vient que des
arêtes longues près des frontières de blocs.

**Quantification du chemin de fuite 2-sauts (entité touchant ≥2 blocs) :**

| Relation | cap | wells conn. | arêtes | entités conn. | **entités MULTI-BLOCS** | arêtes dans entités multi-blocs |
|---|---:|---:|---:|---:|---:|---:|
| well→facility | 1 km | 20,7 % | 4 560 | 405 | **0 (0,0 %)** | 0 (0,0 %) |
| well→facility | 3 km | 43,8 % | 15 963 | 656 | 4 (0,6 %) | 40 (0,3 %) |
| well→facility | 5 km | 58,3 % | 33 021 | 741 | 11 (1,5 %) | 267 (0,8 %) |
| well→aqs_station | 5 km | 36,6 % | 5 850 | 234 | 10 (4,3 %) | 311 (5,3 %) |
| well→aqs_station | 10 km | 67,3 % | 15 383 | 281 | **22 (7,8 %)** | 1 243 (8,1 %) |
| well→aqs_station | 20 km | 88,1 % | 9 979→39 260 | 301 | **47 (15,6 %)** | 5 766 (14,7 %) |

Une « entité multi-blocs » est un nœud `facility`/`aqs_station` dont les puits voisins
tombent dans ≥2 blocs spatiaux distincts. **Chaque telle entité est un pont train↔test**
en 2 couches GNN. C'est la métrique décisive.

**Autocorrélation spatiale de la direction d'écoulement (Q4) :**
cosinus-similarité de `(flow_dir_sin, flow_dir_cos)` entre puits voisins (<3 km) = **0,927**
vs paires aléatoires = **0,063** (n=12 736). La direction d'écoulement est donc fortement
autocorrélée spatialement → toute arête `flows_to` orientée ré-encode la carte.

---

## Q1 — Fuite cible via `facility` ? VERDICT : PAS DE FUITE CIBLE (signal mécaniste légitime)

Les corrélations citées (`n_geotracker_within_50km` r=+0,256, `dist_geotracker_km` r=−0,159)
ont déjà été auditées en `eval_validation.md §1` et classées **proxys mécanistes plausibles,
pas dérivés d'une mesure PFAS**. Le critère de fuite cible est : *la feature est-elle calculée
À PARTIR de la concentration PFAS du puits cible ?* Réponse non — la position d'un site de
contamination est connue indépendamment de toute mesure PFAS dans le puits.

Les arêtes `well→facility` individuelles encodent la **même information, plus précise**
(distance exacte au lieu d'un comptage par anneau), mais de même nature : exposition à une
source amont. C'est exactement le sens causal recherché (source → puits). |r| max constaté
0,256 ≪ seuil d'alerte (0,3 / 0,9). **Aucune fuite cible.**

Réserve unique : ne JAMAIS attacher au nœud `facility` une feature dérivée de mesures PFAS
faites SUR le site (p. ex. concentration du panache). Les coordonnées + `site_type` (Chrome
Plater / Bulk Terminal / Airport / Refinery / WWTP) sont admissibles ; toute concentration
mesurée est interdite (mode prédictif strict). **Condition C-FAC.1.**

La vraie menace de `facility` n'est pas la fuite *cible* mais la fuite *spatiale* (Q2/Q3).

---

## Q2 — Protocole de coupe pour `facility` (chemin 2 sauts) — VERDICT : STRATÉGIE B (dédoublement par bloc)

Analyse des 3 options proposées :

- **Option A (couper les arêtes `well→facility` dont le puits est dans un bloc différent de
  celui du facility).** *Mal posée* : un `facility` n'a pas de bloc intrinsèque (pas de cible,
  c'est une entité de contexte). La seule règle bien définie serait « bloc majoritaire des
  puits voisins du facility » — fragile (dépend du fold, change quand on retire les puits test)
  et **insuffisante** : si un facility touche les blocs {3, 5}, lui assigner le bloc 3 et couper
  les arêtes vers le bloc 5 supprime de l'information train légitime tout en laissant passer le
  pont si l'assignation bascule. **Rejetée.**

- **Option C (arêtes unidirectionnelles `well→facility` seulement, pas de `facility→well`).**
  *Insuffisante.* Un message ne peut alors plus refluer du facility vers les puits, donc le
  facility devient un puits-cul-de-sac qui **n'apporte aucune information aux puits** (il agrège
  mais ne rediffuse jamais) → le nœud est inutile. Si au contraire on garde `facility→well`
  pour l'utilité, le pont 2-sauts revient. C n'offre donc pas de compromis viable : soit
  inutile, soit fuyant. **Rejetée comme protection anti-fuite.** (Le sens d'arête correct
  mécaniste est d'ailleurs `facility→well`, source→récepteur.)

- **Option B (dédoubler le nœud `facility_j` par bloc : `facility_j@bloc_b`, ne reliant que
  les puits du bloc b).** *Suffisante et structurellement garantie.* Un message ne peut jamais
  traverser un facility d'un bloc à l'autre car les copies sont des nœuds disjoints. Aucune
  assertion fragile, la fuite 2-sauts est impossible par construction. Coût : ≤ 813 × 8 copies
  (négligeable), et en pratique 0 copie supplémentaire pour 99,4–100 % des facilities à
  cap ≤ 3 km (puisque presque tous sont mono-bloc — cf. tableau). **RECOMMANDÉE.** Identique à
  la stratégie 2(a) de `gnn_hetero5_analysis.md` Axe 3, validée ici.

**À combiner avec l'inductif train-only (C-SPAT.4)** : à l'entraînement, le nœud `facility@b`
n'agrège que des puits TRAIN du bloc b ; au scoring d'un pli test, les puits test se rebranchent
sur `facility@b_test` qui ne contient QUE des puits train du même bloc-test → un puits test
reçoit un résumé de ses voisins train, jamais de puits test ni d'autres blocs.

**Règle de coupe formelle `well→facility` :**
```
Pour chaque pli, soit B(w) le bloc spatial du puits w.
1. Construire les arêtes well→facility : (w, f) ssi haversine(w, f) <= cap_facility_km.
2. Remapper chaque facility f en copies par bloc : id'(f, w) = (f, B(w)).
   => l'arête (w, f) devient (w, f@B(w)).
3. Inductif : ne garder dans le graphe d'entraînement que les arêtes dont w ∈ TRAIN.
   Au scoring, attacher chaque w_test à f@B(w_test) (qui n'agrège que des w_train du même bloc).
4. Sens : facility@b -> well (source -> récepteur), bidirectionnel autorisé APRÈS dédoublement
   (le dédoublement neutralise déjà le pont).
5. ASSERTION (bloquante) : pour toute copie f@b, tous ses puits voisins ont B(w)==b. 0 violation.
```

Comme à cap=1 km **aucun** facility n'est multi-blocs, l'étape 2 est gratuite à ce cap
(elle ne crée aucune copie) mais reste obligatoire pour la robustesse et pour cap=3 km.

---

## Q3 — Fuite spatiale via `aqs_station` (cap 10–20 km) — VERDICT : RISQUE RÉEL, dédoublement OBLIGATOIRE + cap ≤ 10 km

Réponse directe à la question « l'arête traverse-t-elle des blocs différents ? » : **OUI, et
de façon mesurable**. Le rayon des blocs (~40–100 km) est grand, donc la plupart des arêtes
AQS restent intra-bloc, mais aux frontières :

- cap=5 km : 10 stations multi-blocs (4,3 %), 5,3 % des arêtes concernées.
- cap=10 km : **22 stations multi-blocs (7,8 %), 8,1 % des arêtes** = 1 243 arêtes pontant ≥2 blocs.
- cap=20 km : **47 stations multi-blocs (15,6 %), 14,7 % des arêtes** = 5 766 arêtes pontantes.

Sans dédoublement par bloc, **8 % (cap 10) à 15 % (cap 20) des arêtes AQS créent un pont
train↔test direct** en 2 couches. Vu que l'inflation spatiale du dataset est de 0,17–0,20 AUC,
laisser 8–15 % d'arêtes fuyantes peut ré-injecter une fraction substantielle de cette inflation
et invalider la mesure du Δ — l'apport méthodologique central du projet.

De plus, cap=10–20 km ≫ portée d'autocorrélation (~5 km) : au-delà de 5 km, « partager une
station » ne traduit plus une co-exposition locale honnête mais ré-encode la position
géographique grossière. Une station à 20 km relie des puits de régimes hydrogéologiques
distincts.

**Verdict Q3 :** `aqs_station` admis **seulement** avec :
1. **Dédoublement par bloc obligatoire** (`aqs_station_s@bloc_b`), même règle que Q2 — c'est ici
   non-négociable car 7,8 %+ d'entités sont multi-blocs (contre 0 % pour facility@1km).
2. **Cap ≤ 10 km** (voir Q5) ; 20 km rejeté (14,7 % d'arêtes pontantes, portée 4× l'autocorrélation).
3. **Assertion par copie** : aucune copie `aqs@b` n'a de voisin d'un autre bloc.
4. Si après dédoublement un grand nombre de puits voient leur station perdre ses voisins
   cross-bloc, c'est **attendu et correct** : on coupe la fuite, pas le signal local.

---

## Q4 — Gradient hydraulique : feature (a) vs arête `flows_to` (b) — VERDICT : (a) D'ABORD, (b) sous conditions strictes

**(a) Feature de nœud (`flow_dir_sin`, `flow_dir_cos`, `hydr_grad_mag_permil`).**
VALIDÉE sans réserve de fuite topologique. Couverture recalculée : **11 241/11 333 = 99,19 %
non-NA** (92 NA, à imputer/masquer). Aucune topologie → aucun pont 2-sauts. Mécaniquement utile
(régime d'écoulement local). **À implémenter EN PREMIER.** Réserve C-LOC : ce sont des grandeurs
physiques dérivées de têtes hydrauliques DWR, **pas** lat/lon — donc admissibles comme features
de nœud (contrairement à lat/lon qui restent hors-features, C-LOC.1). À standardiser (fit train
seul). Vérifier qu'elles ne sont pas un proxy déguisé de la cible : à auditer par corrélation OOF
au moment de l'ajout (attendu faible).

**(b) Arête orientée `flows_to` (puits_A → puits_B si B dans la direction d'écoulement ±θ° et
à ≤ cap_km).** RISQUE CONFIRMÉ empiriquement. La direction d'écoulement est **fortement
autocorrélée spatialement** : cos-sim 0,927 entre puits <3 km vs 0,063 aléatoire. Donc une
arête `flows_to` est essentiellement une arête de proximité spatiale orientée → même profil de
fuite que `well→well near`, traitable par la coupe inter-blocs existante MAIS seulement si le
cap est court. Avec un cap long, l'orientation cohérente sur tout un bassin crée des chaînes
orientées qui peuvent diffuser le long d'un axe traversant un bloc.

**Verdict Q4 : (a) en premier, systématiquement.** (b) admissible **après** (a) et seulement si :
- cap_flows_to ≤ portée d'autocorrélation = **≤ 3 km** (idéalement aligné sur `near` à 1,5 km) ;
- coupe inter-blocs `cut_cross_block` appliquée à la relation `flows_to` **séparément**, assert
  0 résiduel (c'est une arête puits↔puits directe, donc 1-saut : la coupe d'arête classique
  suffit, pas besoin de dédoublement) ;
- gain mesuré > σ inter-plis (~0,06 AUC) face à `(a)+near+subbasin` sans `flows_to`. Si le gain
  est dans le bruit, ne pas conserver `flows_to` (parcimonie).
- ne pas cumuler `flows_to` ET `near` non orientées sur le même rayon sans ablation : risque de
  double canal géographique.

---

## Q5 — Caps recommandés — VERDICT

**`well→facility` : cap = 1 km.** À 1 km : 0 facility multi-blocs (pont 2-sauts nul), 20,7 %
puits couverts, 4 560 arêtes sparse. À 3 km : 43,8 % couverts mais 4 facilities multi-blocs
(gérés par dédoublement) et surtout cap > portée d'autocorrélation (5 km encore ok, mais le
signal « source directe » s'étiole : un site à 3 km ne contamine plus forcément par écoulement
direct). **Cap=1 km maximise le ratio signal mécaniste / fuite** (source proche = panache
plausible) et rend le dédoublement gratuit. Cap=3 km autorisé en ablation étiquetée (avec
dédoublement), mais 1 km est le réglage principal.

**`well→aqs_station` : cap = 10 km.** Compromis couverture (67,3 %) / fuite (7,8 % entités
multi-blocs, gérées par dédoublement obligatoire). **20 km rejeté** : 15,6 % multi-blocs et
cap = 4× la portée d'autocorrélation (~5 km) → ré-encodage de carte. Idéalement tester aussi
**5 km** (4,3 % multi-blocs, 36,6 % couverts) comme variante plus conservatrice ; le choix
10 vs 5 km se tranche sur le Δ spatial mesuré (celui qui minimise l'inflation à signal égal).
La portée AQS (co-exposition atmosphérique) est physiquement > 5 km, ce qui justifie 10 km
malgré le dépassement de la portée d'autocorrélation de T1a — **à condition** que le
dédoublement neutralise la fuite, ce qu'il fait.

**`well→subbasin` (hub, cap 2 km intra) : CONFIRMÉ.** Déjà validé (`same_subbasin_knn` cap 2 km,
`gnn_hetero5_analysis.md` Axe 1). Le hub sous-bassin = agrégation de `same_subbasin_knn` ; mêmes
conditions : dédoublement par bloc + cap intra-hub 2 km + assertion. Rappel honnêteté : ce hub
n'apporte rien de neuf vs puits-puits (Axe 4) ; sa valeur est la cohérence d'architecture, pas
un gain attendu.

---

## Tableau des conditions bloquantes

| Id | Condition | Relation | Bloquant |
|----|-----------|----------|:--------:|
| C-FAC.1 | Aucune feature dérivée d'une mesure PFAS sur le nœud `facility` (coords + site_type only) | facility | OUI |
| C-FAC.2 | Dédoublement par bloc `facility@b` + inductif train-only | facility | OUI |
| C-FAC.3 | Cap = 1 km (3 km en ablation étiquetée seulement) | facility | OUI |
| C-AQS.1 | Dédoublement par bloc `aqs@b` obligatoire (7,8 %+ entités multi-blocs) | aqs_station | OUI |
| C-AQS.2 | Cap ≤ 10 km (20 km REFUSÉ) ; tester 5 km en variante | aqs_station | OUI |
| C-AQS.3 | Aucune feature AQS dérivée d'une mesure PFAS | aqs_station | OUI |
| C-HYD.1 | Gradient hydraulique en FEATURE de nœud d'abord (a), 99,19 % couverture, fit train seul | well | OUI |
| C-HYD.2 | Arête `flows_to` (b) seulement après (a), cap ≤ 3 km, `cut_cross_block` assert 0, gain > σ | well→well | OUI |
| C-ASSERT | Par relation ET par copie : 0 voisin d'un autre bloc (assertion avant run long) | toutes | OUI |
| C-IND | Inductif strict : aucun label/puits test n'entre dans le message-passing train | toutes | OUI |
| C-DELTA | Rapporter Δ(AUC aléatoire − AUC spatiale) AVEC les nouvelles relations, pour vérifier que les hubs ne rouvrent PAS l'inflation | toutes | OUI |
| C-SMOKE | Smoke-test CPU : hubs/copies construits, 0 inter-blocs, courbes perte/AUC train+val non vides (§3.8), figures écrites | — | OUI |

Reportent telles quelles toutes les conditions de `eval_validation.md` (C-SPAT.*, C-THR, C-CAL,
C-CMP, C-MET, C-LOC) — elles restent en vigueur.

---

## Règles de coupe formelles — récapitulatif par relation

| Relation | Cap | Saut | Coupe |
|---|---:|---|---|
| well↔well `near` (existant) | 1,5 km | 1 | `cut_cross_block` + assert 0 |
| well↔well `same_subbasin_knn` (existant) | 2 km | 1 | `cut_cross_block` séparé + assert 0 |
| well→well `flows_to` (Q4b, optionnel) | ≤3 km | 1 | `cut_cross_block` + assert 0 ; après feature (a) |
| facility→well | 1 km | 2 | **dédoubler `facility@b`** + inductif train-only + assert copie mono-bloc |
| aqs_station→well | ≤10 km | 2 | **dédoubler `aqs@b`** + inductif train-only + assert copie mono-bloc |
| subbasin→well (hub, existant) | 2 km intra | 2 | **dédoubler `subbasin@b`** + cap 2 km + assert copie mono-bloc |

**Assertion 2-sauts unifiée** (à exécuter pour chaque type de nœud-hub, avant tout run long) :
> pour tout nœud-hub `h` du graphe (après dédoublement), l'ensemble {B(w) : w voisin de h}
> est un singleton. Toute violation = fuite 2-sauts → run BLOQUÉ.

---

## Diagnostic d'entraînement (rappel CLAUDE.md §3.8 / C-SMOKE)

Le verdict ci-dessus autorise la **construction** du graphe et son smoke-test, pas la
revendication d'un score. Avant tout résultat rapporté, exiger : courbes perte+AUC train ET val
par pli (8 plis spatiaux), historiques `history.json` non vides de longueur = n_époques, figures
dans `experiments/<id>/figures/`, et un diagnostic explicite (convergence, sur-/sous-apprentissage,
plis sous-entraînés, cohérence early-stop ↔ courbe de validation). Un score AUC final sans courbe
examinée reste **À CORRIGER** — c'est précisément le piège (sous-entraînement masqué) corrigé en P0.

---

## Synthèse

L'ajout des vrais nœuds `facility` (geotracker+WWTP, coords) et `aqs_station` (AQS, coords)
est **méthodologiquement valide et mécaniquement supérieur** au graphe puits-puits, ce qui lève
le refus C-NODE de `eval_validation.md`. La fuite cible est absente (Q1). La fuite spatiale
2-sauts est **nulle pour facility à 1 km** mais **réelle pour AQS (7,8 % à 10 km)** : elle est
neutralisée structurellement par le **dédoublement de hub par bloc** (Option B, Q2/Q3),
préférable aux options A et C. Le gradient hydraulique entre **en feature d'abord** (a),
l'arête `flows_to` (b) restant subordonnée à un cap court car la direction d'écoulement est
fortement autocorrélée (cos-sim 0,93 vs 0,06). Caps retenus : facility 1 km, AQS ≤ 10 km,
subbasin 2 km. **VALIDÉ SOUS CONDITIONS** du tableau bloquant.
