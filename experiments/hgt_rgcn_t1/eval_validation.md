# Validation méthodologique — GNN hétérogène (HGT / R-GCN) sur T1

Méthodologiste : eval-methodologist. Date : 2026-06-22.
Dataset audité : `data/CA-PFAS-ASGWS.parquet` (46 338 lignes × 201 colonnes ;
11 333 puits uniques ; 4,09 échantillons/puits en moyenne, max 139).
Cible auditée : T1a (EPA 2024), prévalence **44,5 %** (jeu quasi équilibré, pas un
régime d'événement rare).

---

## VERDICT : VALIDÉ SOUS CONDITIONS

Le protocole d'évaluation (anti-fuite cible, anti-fuite spatiale, CV par blocs,
seuil OOF, calibration) est **sain et déjà partiellement implémenté** dans `src/`.
Il est donc applicable à HGT / R-GCN.

**MAIS la conception du graphe hétérogène telle que proposée est REFUSÉE en l'état :**
le type de nœud « source/installation » **n'existe pas** dans les données et la
construction proposée des arêtes puits→source réintroduirait une fuite spatiale.
Le graphe doit être **homogène puits–puits** (ou bipartite puits–attribut), comme
précisé en condition C-NODE ci-dessous. HGT et R-GCN restent valides comme
encodeurs **relationnels** (plusieurs types d'arêtes), pas comme encodeurs de
plusieurs types de nœuds réels.

---

## 1. Fuite cible — VALIDÉ (blocklist confirmée + scan indépendant)

La blocklist `config.LEAKAGE_BLOCKLIST` (96 colonnes) est correcte et exhaustive.
Je ne m'y suis pas fié : je l'ai recoupée et complétée par un scan indépendant.

**Colonnes à exclure des features (toutes dérivées d'une mesure PFAS) :**

- 31 × `<analyte>_ngL` — concentrations mesurées (entrée interdite, mode prédictif strict).
- 31 × `<analyte>_detected` — booléen de détection = seuil sur la concentration.
- 31 × `label_<analyte>` — labels T2 = seuil sur la concentration.
- `sum_pfas_ngL` — somme des concentrations (fuite directe de T1b ; corrèle T1a).
- `target_sum_gt70` — la cible T1b elle-même.
- `pfas_class_assignment` — **vérifié : colonne constante** (une seule valeur sur
  46 338 lignes, un dict figé). Inutile ET dérivée de la signature analytique → exclue.

**Scan indépendant de proxy résiduel** (corrélation de Pearson de CHAQUE feature
candidate non-blocklistée avec T1a) — aucun proxy caché :

| feature candidate          | corr. avec T1a |
|----------------------------|---------------:|
| n_geotracker_within_50km   | +0,256 |
| n_geotracker_within_10km   | +0,197 |
| aqs_no2_ppb                | +0,189 |
| latitude                   | −0,188 |
| aqs_co_ppm                 | +0,161 |
| dist_geotracker_km         | −0,159 |
| longitude                  | +0,154 |

Maximum |r| = 0,256. Aucune feature ne dépasse le seuil d'alerte (pas de |r| > 0,9
ni même > 0,3). Les têtes de liste sont des **proxys mécanistes plausibles**
(densité de sites contaminés, marqueurs urbains/trafic, position géographique),
**pas** des dérivés de la mesure PFAS. **Aucune fuite cible résiduelle.**

`gm_dataset_name` (provenance, 6 valeurs) : confounder de design — **audit
uniquement, jamais feature** (déjà marqué C6 dans `config.py`). Confirmé.

---

## 2. Type de nœud « source/installation » — N'EXISTE PAS → graphe hétérogène REFUSÉ tel quel

C'est le point dur de la proposition. Audit du dataset :

- **Aucune table, aucun ID, aucune coordonnée d'installation.** Les colonnes
  contenant un signal « site » sont exclusivement des **agrégats au niveau du puits** :
  `dist_geotracker_km`, `nearest_geotracker_type` (4 valeurs : Chrome Plater, Bulk
  Terminal, Airport, Refinery), `n_geotracker_within_{1,3,10,50}km`.
- Ces colonnes décrivent le *voisinage* d'un puits, pas des entités sources
  individuelles localisables. Il n'y a **pas** de second jeu de nœuds.

**Conséquences :**

- **C-NODE.1 — Pas de nœuds « source » fabriqués à partir d'agrégats.** Créer des
  nœuds source en regroupant des puits par `nearest_geotracker_type` (4 cliques
  géantes) reviendrait à une **clique par type** — explicitement REFUSÉ par le
  protocole existant (`graph.build_well_graph` lève une erreur pour
  `source-type cliques`). Refusé ici aussi : ces arêtes ne portent aucune
  information spatiale honnête et créent des chemins de message-passing arbitraires.
- **C-NODE.2 — Graphe homogène puits–puits acceptable et RECOMMANDÉ.** En l'absence
  de sources identifiables, le graphe correct est puits–puits (k-NN spatial capé,
  voisinage hydrologique intra-sous-bassin), exactement ce que `src/graph.py`
  implémente déjà. C'est la seule topologie spatiale défendable.
- **C-NODE.3 — HGT/R-GCN restent légitimes comme encodeurs RELATIONNELS.** Un graphe
  *multi-relationnel à un seul type de nœud réel* est valide : relations
  `('well','near','well')` (k-NN spatial) et `('well','same_subbasin_knn','well')`
  (mécanistique intra-sous-bassin) sont deux types d'arêtes distincts → R-GCN et HGT
  ont du sens. Alternativement, un graphe **bipartite puits–attribut** (analyte pour
  T2, type de site comme nœud-attribut catégoriel pour T1) est admissible SI les
  arêtes attribut↔puits ne court-circuitent jamais la coupe spatiale (elles ne
  touchent qu'un puits = un bloc, donc OK par construction, cf. P1+).
- **C-NODE.4 — `nearest_geotracker_type` reste une feature de nœud**, pas un nœud.
  C'est de l'information de contexte au puits ; la garder en one-hot (déjà dans
  `CATEGORICAL_LOW_CARD`).

---

## 3. Fuite spatiale via les arêtes — RISQUE MAJEUR, contrôle OBLIGATOIRE

Un découpage **aléatoire** train/val/test sur ce graphe est **INVALIDE**. Preuve
empirique tirée des phases antérieures (`experiments/gnn_phase1/metrics.json`,
même protocole, T1a) :

| modèle    | AUC split aléatoire (GroupKFold) | AUC split spatial (blocs) | **Δ inflation** |
|-----------|---------------------------------:|--------------------------:|----------------:|
| GraphSAGE | 0,815 ± 0,019 | 0,618 ± 0,067 | **+0,196** |
| GCN       | 0,842 ± 0,017 | 0,624 ± 0,074 | **+0,218** |

**L'inflation spatiale est de ~0,20 point d'AUC** — gigantesque, et bien supérieure
au bruit inter-plis (σ ≈ 0,02 en aléatoire). Un score « 0,84 » obtenu en split
aléatoire sur ce dataset est un **artefact d'autocorrélation spatiale**, sans valeur
décisionnelle. C'est l'apport méthodologique central du projet : il doit être
**reproduit et rapporté pour HGT et R-GCN** sur le même format (même k de blocs,
même cap_km).

**Double fuite spatiale par message-passing.** Au-delà du split des lignes, les
arêtes elles-mêmes propagent l'information à travers les plis :

- **C-SPAT.1 — CV par blocs spatiaux au niveau PUITS, obligatoire.** Utiliser
  `splits.spatial_block_folds` (KMeans sur coordonnées par puits, k=8). Train et test
  ne partagent ni puits ni bloc. Le `group_random_folds` ne sert QU'À mesurer le Δ.
- **C-SPAT.2 — Coupe des arêtes inter-blocs, obligatoire et ASSERTÉE.** Toute arête
  puits–puits dont les extrémités tombent dans des blocs différents doit être
  supprimée par fold (`graph.cut_cross_block`), avec `assert removed_cross_block`
  cohérent et **0 arête inter-blocs résiduelle**. Sans cela, un nœud test reçoit des
  messages d'un nœud train voisin = fuite. En phase 1, 152 arêtes coupées au total
  (cap 1,5 km) — le contrôle fonctionne.
- **C-SPAT.3 — Cap de distance sur les arêtes.** Conserver le cap (`cap_km`, défaut
  1,5 km spatial / 2 km intra-sous-bassin). Au-delà de la portée d'autocorrélation
  mesurée, « proximité » ne fait que ré-encoder la carte = fuite spatiale.
- **C-SPAT.4 — Inductif sur les labels.** Le message-passing ne doit utiliser que les
  arêtes côté train ; aucun label test ne doit entrer dans le graphe. Un nœud test
  s'attache à ses voisins TRAIN via des arêtes intra-bloc uniquement.
- **C-SPAT.5 — Pour HGT/R-GCN spécifiquement :** la coupe inter-blocs s'applique à
  **CHAQUE type de relation** (spatial ET intra-sous-bassin). Vérifier explicitement
  par relation, car HeteroConv route les messages par type d'arête — une seule
  relation non coupée suffit à rouvrir la fuite. Assertion par relation requise.

---

## 4. Encodage carte / localisation comme proxy — VALIDÉ sous condition

- **C-LOC.1 — `latitude`/`longitude` PAS en features de nœud par défaut.** lat/lon
  corrèlent ±0,15–0,19 avec T1a (gradient géographique). Les passer en features
  laisse le modèle apprendre « où » plutôt que « pourquoi ». La géographie doit
  entrer UNIQUEMENT par la topologie k-NN capée. C'est déjà le défaut
  (`include_location=False`). Si une ablation `include_location=True` est menée, elle
  doit être **étiquetée comme ablation de diagnostic**, comparée au défaut, et ne
  jamais être la config rapportée comme résultat principal.
- **C-LOC.2 — Pas de réencodage carte déguisé.** Les catégories admin
  (`county`, `dwr_basin`, `sgma_subbasin_name`…) en **target-encoding** sont un
  proxy géographique puissant de la cible. Le code utilise un target-encoder
  **out-of-fold (KFold interne)** — correct. Mais sous CV **spatiale**, une catégorie
  géographique entière peut être absente du train → encodage = moyenne globale (pas
  de fuite, mais info nulle). Acceptable. **Interdiction** : ne jamais target-encoder
  sur le test, ni ajuster l'encodeur sur train+test. Vérifié : `FeaturePipeline`
  ajuste sur train seul.
- **C-LOC.3 — `sgma_subbasin_name` contraint les arêtes mécanistiques mais n'est
  JAMAIS feature de nœud dans le graphe mécanistique** (déjà acté `config.py`).

---

## 5. Optimisation de seuil — VALIDÉ

`gnn.py:_f1_threshold` et `hybrid.py:_optimal_threshold_f1` calculent le seuil F1
sur probabilités **OOF/VAL uniquement**, jamais sur le test (docstring + appels
vérifiés). **Condition C-THR :** pour HGT/R-GCN, le seuil doit venir des probas OOF
des plis internes / nœuds VAL, et le même seuil être appliqué au test du pli externe.
Rapporter aussi les métriques sans seuil (AUC, AP) pour ne pas dépendre du seuil.

---

## 6. Calibration — VALIDÉ (présent), à RAPPORTER systématiquement

`metrics.binary_metrics` produit déjà Brier ; le pipeline hybride rapporte Brier +
ECE. **Condition C-CAL :** pour HGT/R-GCN, rapporter **Brier, ECE et courbe de
fiabilité** sur les probas OOF agrégées. Les GNN sont notoirement mal calibrés
(softmax/sigmoïde sur-confiants) ; une AUC honnête mal calibrée reste inexploitable
en décision. Ajouter une calibration post-hoc (Platt/isotonic ajustée OOF) en option
si ECE élevé — ajustée OOF, jamais sur test.

---

## 7. Comparaisons & intervalles — VALIDÉ, à étendre

Bootstrap CI déjà présent (`bootstrap_ci`, n_boot=1000). **Condition C-CMP :**

- IC bootstrap sur l'AUC OOF globale pour HGT et R-GCN.
- **Test apparié sur les plis** (Wilcoxon signé ou t apparié) HGT vs R-GCN vs le mur
  de baselines non-graphe, sur les **8 mêmes plis spatiaux**.
- **Méfiance gain < bruit inter-plis :** σ spatial ≈ 0,06–0,07 ici. Tout gain GNN
  inférieur à ~0,06 AUC face à la baseline est **dans le bruit** et ne doit PAS être
  revendiqué. Rappel : en phase 1, les GNN spatiaux (0,62) sont SOUS le mur non-graphe
  — la barre est haute, ne pas survendre.
- Positionner vs littérature (Dong et al. 2024) en précisant le régime de split :
  les chiffres élevés de la littérature sont souvent en split aléatoire et donc
  **non comparables** au split spatial honnête.

---

## 8. Métriques — cohérence imposée

T1 binaire équilibré (prévalence 44,5 %) : ROC-AUC (primaire), PR-AUC/AP,
rappel, balanced accuracy, F1@seuil-OOF, gain/lift top-k (décision), Brier+ECE
(calibration). **Condition C-MET :** utiliser EXACTEMENT le même jeu de métriques et
le même k de blocs / cap_km que les phases 1–3 et le mur de baselines, sinon les
comparaisons sont invalides.

---

## Récapitulatif des conditions (à respecter à l'implémentation)

| Id | Condition | Bloquant |
|----|-----------|:--------:|
| C-NODE.1 | Pas de nœuds source fabriqués (clique par type interdite) | OUI |
| C-NODE.2 | Graphe homogène puits–puits (ou bipartite puits–attribut) | OUI |
| C-NODE.3 | HGT/R-GCN = encodeurs multi-relationnels (1 type de nœud réel) | — |
| C-NODE.4 | `nearest_geotracker_type` = feature de nœud, pas un nœud | OUI |
| C-SPAT.1 | CV blocs spatiaux niveau puits (`spatial_block_folds`) | OUI |
| C-SPAT.2 | Coupe arêtes inter-blocs + assertion 0 résiduel | OUI |
| C-SPAT.3 | Cap de distance sur arêtes (1,5 / 2 km) | OUI |
| C-SPAT.4 | Inductif sur labels (aucun label test dans le graphe) | OUI |
| C-SPAT.5 | Coupe inter-blocs appliquée et assertée PAR relation | OUI |
| C-SPAT.6 | Rapporter Δ(aléatoire − spatial) pour HGT et R-GCN | OUI |
| C-LOC.1 | lat/lon hors features par défaut (ablation étiquetée) | OUI |
| C-LOC.2 | Target-encoding OOF, ajusté train seul | OUI |
| C-THR  | Seuil F1 sur OOF/VAL uniquement | OUI |
| C-CAL  | Brier + ECE + courbe de fiabilité rapportés | OUI |
| C-CMP  | IC bootstrap + test apparié sur plis; gain > σ inter-plis | OUI |
| C-MET  | Métriques + k blocs + cap identiques aux phases précédentes | OUI |

**Avant tout run long :** smoke-test CPU vert (cf. CLAUDE.md §5), assertions
cross-block à 0 par relation, et confirmation que lat/lon ne sont pas dans les
features de nœud. Tant que C-NODE.1/2 et C-SPAT.* ne sont pas satisfaites, le run
hétérogène est **bloqué**.
