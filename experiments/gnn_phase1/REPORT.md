# REPORT — GNN phase 1 (T1a, graphe spatial au niveau puits)

> Agent `gnn-researcher`. Graine 42. **Réutilise le socle figé** (`src/{config,data,
> targets,splits,features,metrics}.py`) — aucune cible, blocklist, split ni métrique
> réimplémentés. Évalué sous le **MÊME protocole** que le mur non-graphe
> (`experiments/baseline_t1`). Conditions **C1-C6** de `EVAL_PROTOCOL.md` respectées.
>
> Reproductible : `PFAS_FORCE_CPU=1 python3 experiments/gnn_phase1/run_gnn_t1.py`
> → `experiments/gnn_phase1/metrics.json` (source de tous les chiffres).
> Smoke : `PFAS_FORCE_CPU=1 python3 tests/test_gnn_smoke.py` (< 20 s CPU).

---

## 0. Résumé exécutif

- **Premier GNN T1a (GraphSAGE / GCN) sur graphe spatial au niveau puits**, distance-capée
  à 1,5 km, arêtes inter-bloc coupées par pli (C4). Évaluation au **niveau prélèvement**
  (46 338 lignes) pour comparabilité stricte au mur.
- **Triplet (random, spatial, Δ)** :
  | modèle | spatial | random | Δ = rd−sp | vs mur RF 0,601 |
  |---|---|---|---|---|
  | **GraphSAGE** | **0,618 ± 0,067** | 0,815 | **0,196** | +0,017 |
  | **GCN**       | **0,624 ± 0,074** | 0,842 | 0,218 | +0,023 |
- **Lecture honnête** : les deux GNN **égalent le mur spatial** (RF 0,601 / XGB 0,588) et
  le dépassent légèrement, **mais l'écart (+0,017 / +0,023) reste SOUS le seuil de bruit
  inter-pli (~0,03)** → ce n'est **pas encore un gain net significatif**. Le résultat
  marquant n'est pas l'AUC mais **le Δ** : **0,20-0,22 contre 0,30 pour RF** → le GNN
  exploite **nettement moins la structure de carte** (random plus bas : 0,82-0,84 vs ~0,90
  baseline). C'est exactement le profil recherché : tenir le spatial **sans gonfler Δ**.
- **C4 prouvé** : 152 arêtes inter-bloc retirées au total sur la CV (19/pli), **0 arête
  inter-bloc restante** → le message passing ne rouvre pas la fuite spatiale.
- **Marge identifiée** : 2-3 plis sous-entraînés (early stop précoce, AUC ~0,51-0,54) tirent
  la moyenne ; l'early stopping sur un seul bloc spatial de validation est instable →
  première amélioration à viser (cf. §6).

---

## 1. Design du graphe (et sa justification)

### 1.1 Granularité des nœuds = PUITS (11 333), pas prélèvement

- **Cohérence avec le socle** : coords (`splits.well_coordinates`), blocs spatiaux
  (`splits.spatial_block_*`) et clé de groupe C2 (`gm_well_id`) vivent **tous au niveau
  puits**. Un nœud = un puits = une (lat, lon) = un bloc.
- **Volumétrie** : un graphe au niveau prélèvement ferait exploser les arêtes spatiales
  (produit cartésien des prélèvements de puits voisins ≈ **1,6 M arêtes**, puits jusqu'à
  139 prélèvements) sans information nouvelle (coords identiques). Au niveau puits :
  **~31 k arêtes** (cap 1,5 km), graphe léger.
- **Features de nœud** = agrégat par puits (`graph.aggregate_to_wells`) : moyenne pour le
  numérique, mode pour le catégoriel. **`FeaturePipeline` ajustée sur les puits du TRAIN
  uniquement** (anti-fuite), encodage **fréquentiel** (sans cible, trivialement non
  fuitant — comme la baseline T2).
- **Cible d'entraînement par nœud** = **majorité** des cibles des prélèvements du puits
  (`graph.well_majority_target`). Fait établi : **93,9 % des puits ont une T1a constante**
  dans le temps ; seuls 15,9 % des puits multi-prélèvements varient → l'agrégation perd peu.

### 1.2 Évaluation au niveau PRÉLÈVEMENT (comparabilité stricte au mur)

Le mur RF/XGB 0,601 est calculé au **niveau ligne**. Pour comparer sous le même protocole,
la proba du nœud (puits) est **rediffusée à chacun de ses prélèvements** (`row_to_node`) et
les **5 métriques** sont calculées au niveau ligne via `metrics.binary_metrics`. Le
groupage C2 est automatique (un puits = un nœud = un bloc). La prévalence ligne (0,445)
est donc préservée à l'évaluation — pas la prévalence puits-majoritaire (0,311) utilisée
seulement pour l'entraînement.

**Limite assumée** : ce schéma sous-exploite la variation temporelle intra-puits (15,9 %
des puits multi). L'hydro-critique juge le temporel **confondeur faible**, pas un mécanisme
→ acceptable en phase 1 ; une **vue prélèvement** est mise en ablation (§6).

### 1.3 Arêtes : k-NN spatial distance-capé (C4)

- **k = 8 voisins**, **plafond dur 1,5 km** en distance **haversine** (grand cercle → le
  cap est un vrai km physique). Justifié par la portée d'autocorrélation **2-5 km**
  (`EVAL_PROTOCOL` §2.3) : au-delà, « proximité » ne fait que **réencoder la carte** =
  fuite spatiale. Profil de distances vérifié sur les données (p90 du plus proche voisin =
  1,11 km ; cap 1,5 km conserve ~60 % des arêtes 8-NN, 10,7 k/11,3 k puits gardent ≥1 arête).
- **Pondération** : `exp(-(d/cap)²)` (proche = fort) — le cap devient un prior physique doux.
- **Géographie via les arêtes UNIQUEMENT** (C6) : lat/lon **pas** en features de nœud
  (`include_location=False`). `gm_dataset_name` exclu (socle). `cocontam="core"`
  (hydro-trusted), **pas** le bloc VOC/BTEX/fréons.
- **C5 (collision bassin × blocs)** : aucune arête « même bassin/sous-bassin » construite —
  seulement le k-NN distancié, qui ne franchit pas les frontières de bloc après coupe.

### 1.4 C4 — coupe des arêtes inter-bloc, graphe reconstruit par pli

`graph.cut_cross_block` retire **toute arête dont les deux extrémités tombent dans des
blocs CV différents**, AVANT symétrisation. Le graphe est donc (re)construit par pli avec
`fold_block` = vecteur de blocs (spatial ou random). **Résultat mesuré** : 19 arêtes
inter-bloc coupées par pli spatial (152 au total), **0 restante** (vérifié dans le smoke et
le run). Sans cette coupe, le message passing transporterait l'information train→test et
**invaliderait la CV spatiale** (exactement le risque C4).

---

## 2. Modèle & entraînement

- **Architectures** : `graphsage` (SAGEConv, aggr mean, inductif d'esprit), `gcn`
  (GCNConv avec poids d'arête). 2 couches (2 sauts), hidden 64, LayerNorm + ReLU + dropout
  0,5, tête linéaire. `src/gnn.py:build_model` couvre aussi `graphconv` (extensible).
- **Transductif masqué** : tous les nœuds dans un seul graphe ; **perte BCE sur les nœuds
  TRAIN uniquement** (`pos_weight` équilibré), un **bloc spatial du train tenu de côté**
  pour la validation / early stopping (patience 30, max 300 époques), test scoré **une
  seule fois** par pli. Comme les arêtes inter-bloc sont coupées (C4), la prédiction d'un
  puits test ne dépend que de son voisinage **du même bloc** → comportement **inductif**
  vis-à-vis des autres blocs (généralisation à des régions non vues), ce que mesure le
  score spatial.
- **Seuil de décision** : F1-optimal sur les probas **OOF de validation** (jamais sur le
  test), aligné sur le protocole baseline.

---

## 3. Résultats vs le mur (triplet random / spatial / Δ)

Données complètes (46 338 lignes, 11 333 puits), CV à 8 blocs, run canonique **21,4 min
CPU** (`metrics.json`).

| modèle | AUC **spatial** | AUC random | **Δ** | écart vs mur RF (0,601) | vs bruit 0,03 |
|---|---|---|---|---|---|
| Mur RF (baseline) | 0,601 ± 0,056 | 0,898 | 0,297 | — | — |
| Mur XGB (baseline) | 0,588 ± 0,068 | 0,900 | 0,313 | — | — |
| **GraphSAGE** | **0,618 ± 0,067** | 0,815 | **0,196** | **+0,017** | dans le bruit |
| **GCN** | **0,624 ± 0,074** | 0,842 | 0,218 | **+0,023** | dans le bruit |

**Interprétation rigoureuse** :
1. **Le spatial est au niveau du mur** (légèrement au-dessus), pas un gain net (< 0,03).
   Je ne revendique **pas** de victoire sur l'AUC à ce stade.
2. **Le Δ est le vrai signal** : 0,196-0,218 contre 0,30-0,31 pour les arbres. Le GNN
   **généralise mieux géographiquement** relativement à ce qu'il « apprend en random ».
   Son AUC random plus basse (0,815-0,842 < 0,90) confirme qu'il **mémorise moins la carte**
   — cohérent avec C4 (pas d'arêtes longues, pas de lat/lon en nœud).
3. **Hétérogénéité par pli** : per-fold spatial GraphSAGE 0,518→0,699. Les plis bas (0,518,
   0,554) correspondent à un **early stop précoce** (≤ ep 9) sur un bloc de validation
   bruité → sous-entraînement, pas un plafond du modèle. Marge claire (§6).

---

## 4. Smoke-test CPU (vert)

`tests/test_gnn_smoke.py` (1 500 puits, 6 blocs) — **6 tests, ~12-18 s CPU** :
- PyG importe sur CPU (torch 2.12, pyg 2.7) ;
- graphe : 1 500 nœuds, arêtes ≤ 1,5 km (cap respecté), `row_to_node` couvre toutes les
  lignes ;
- **C4 : 2 arêtes inter-bloc coupées → 0 restante** (et le graphe non coupé en avait bien) ;
- **perte finie et décroissante** (0,694 → 0,220 en 30 époques) ;
- 5 métriques calculées au niveau ligne ;
- **graphe bipartite puits×analyte** construit (14 199 arêtes cellule-mesurée, 10 analytes).

**Durée run complet** : CPU **~21 min** pour 2 modèles × 2 régimes × 8 plis (mesuré). Sur
**Colab GPU** : graphe minuscule (11 k nœuds), **quelques secondes par pli** → run complet
attendu **< 2 min GPU**. Pas de run long requis — mais le notebook autonome est fourni.

---

## 5. Livrables

| fichier | rôle |
|---|---|
| `src/graph.py` | construction de graphe (k-NN distance-capé, coupe inter-bloc C4, features de nœud anti-fuite, **graphe bipartite puits×analyte** pour T2) — torch-free, import CPU léger |
| `src/gnn.py` | modèles GraphSAGE/GCN/GraphConv + boucle transductive masquée + `run_t1_cv` (triplet) |
| `tests/test_gnn_smoke.py` | smoke CPU bout-en-bout (graphe, C4=0, perte↓, métriques, bipartite) |
| `experiments/gnn_phase1/run_gnn_t1.py` | driver (toggle `SMOKE_TEST`) → `metrics.json` |
| `experiments/gnn_phase1/{config.yaml,metrics.json}` | config + run canonique |
| `notebooks/gnn_phase1_t1_colab.ipynb` | notebook autonome (clone repo+data, install PyG GPU, run, persistance zéro-Drive) |

---

## 6. Plan d'exploration systématique (priorisé)

> Sous le MÊME protocole (CV spatiale référence + random pour Δ, seuil OOF, 5 métriques,
> C1-C6). Tout nouveau protocole validé par `eval-methodologist` avant run long.

**Priorité 0 — corriger l'instabilité d'entraînement (gratuit, peut débloquer le gain)** :
early stopping robuste (validation = **plusieurs micro-blocs** assemblés plutôt qu'un seul
bloc, cf. `EVAL_PROTOCOL` §2.4), plus de patience, LR schedule. Les plis à 0,51 doivent
remonter vers 0,65+ ; c'est la voie la plus rapide pour passer le spatial au-dessus du bruit.

**Priorité 1 — T2 complétion de matrice bipartite (piste forte, MNAR)** : `src/graph.py`
fournit déjà `build_bipartite_well_analyte` (puits×analyte, arêtes = cellules mesurées via
`baselines_t2.measurement_mask`, label = dépassement). Modèles : **GAE / VGAE / complétion
inductive type IGMC** ; prédiction de liens pour compléter la matrice lacunaire. À comparer
au mur T2 (macro-AUROC spatial 0,680). Recommandé par `HYDRO_CRITIQUE` au-dessus du graphe
de labels (les chaînes ont échoué côté baseline).

**Priorité 2 — ablation de construction d'arêtes** : cap distance {1,0 ; 1,5 ; 2,0 km} ×
k {5, 8, 12} ; **k-NN dans l'espace des features** (vs spatial) ; arêtes **temporelles**
intra-puits (vue prélèvement) ; arêtes **« même source géotracker »** (panache commun,
mécaniste). Rapporter le triplet à chaque fois — une variante qui monte le random sans le
spatial = arêtes qui réencodent la carte (à rejeter).

**Priorité 3 — catalogue convolutif/attentionnel** sous le même harnais `run_t1_cv` :
GraphConv, **SGC**, **APPNP/PPNP** (propagation découplée, robuste), **TAGCN**, **ARMA** ;
attention : **GAT, GATv2** ; expressifs : **GIN, PNA**. Mesurer si l'attention/propagation
profonde aide le **spatial** (pas le random).

**Priorité 4 — profondeur / agrégation / régularisation** : couches {1,2,3} (sur-lissage
au-delà), agrégateurs SAGE {mean, max, lstm}, normalisation {LayerNorm, BatchNorm, aucune},
**dropout d'arêtes**, **résiduels**. Inductif explicite (neighbor sampling GraphSAGE) vs
transductif — mesurer la généralisation à des puits non vus.

**Priorité 5 — hétérogène / passage à l'échelle** : `HeteroData` (puits + source +
bassin comme types de nœuds, R-GCN/HGT/SAGE hétérogène) ; neighbor sampling /
Cluster-GCN / GraphSAINT si le graphe enrichi grossit.

**Priorité 6 — hybride GNN ⊕ arbres** : embedding GNN (sortie avant-tête) **concaténé** aux
features et passé à RF/XGB (stacking). Teste si le signal relationnel est complémentaire du
mur tabulaire — le candidat le plus probable pour un gain net sur le spatial.

---

## 7. Points de vigilance

1. **Gain pas encore net** : +0,017/+0,023 spatial est **dans le bruit** (< 0,03). Le GNN
   **n'a pas encore battu le mur** au sens fort ; il l'**égale avec un Δ plus sain**. Ne
   pas sur-vendre avant la priorité 0/6.
2. **Instabilité d'entraînement** : early stop sur un seul bloc de validation → plis
   sous-entraînés. À corriger avant toute conclusion (priorité 0).
3. **Δ comme garde-fou permanent** : toute variante future doit être jugée sur le **spatial**
   et son **Δ**, jamais sur le random. Une hausse du random sans hausse du spatial = artefact.
4. **Agrégation au niveau puits** : perd la variation temporelle intra-puits ; documenté
   comme limite, à lever via la vue prélèvement (priorité 2) si le temporel s'avère utile.
5. **C4 dépend de `fold_block` cohérent par puits** : `build_well_graph` lève une assertion
   si un puits chevauche deux blocs — garde-fou en place.
6. **Reproductibilité GPU** : le notebook installe PyG via le wheel index correspondant à
   `torch.__version__`+CUDA du runtime ; vérifier l'import avant le run long (cellule 3).

### Artefacts
- `experiments/gnn_phase1/metrics.json` — run canonique (tous les chiffres ci-dessus).
- `experiments/gnn_phase1/config.yaml` — configuration figée.
