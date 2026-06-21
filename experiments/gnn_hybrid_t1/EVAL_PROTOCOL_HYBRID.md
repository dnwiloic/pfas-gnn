# EVAL_PROTOCOL_HYBRID — Audit méthodologique de l'expérience GNN-hybrid T1

> Agent `eval-methodologist` (gardien). J'AUTORISE ou je REFUSE le protocole AVANT tout
> run lourd. Je ne présume rien : chaque chiffre ci-dessous est recalculé indépendamment
> (graine 42, niveau puits, blocs KMeans k=8) par de petits checks CPU (< 3 min).
> Aucun entraînement n'a été lancé (CLAUDE.md §4, mémoire « heavy-runs-colab-only »).

---

## 0. VERDICT : APPROUVÉ SOUS CONDITIONS (contrat ci-dessous, bloquant)

Le triplet **GNN seul / XGBoost seul / hybride** sous le protocole spatial figé (C1–C6 de
`profilage/EVAL_PROTOCOL.md`) est **méthodologiquement légitime** et constitue une
extension valide de la Priorité 6. **MAIS le design d'arêtes proposé (« même source » /
« connectivité hydrologique ») est en partie REFUSÉ tel quel**, parce que mes mesures
montrent que :

- les arêtes **« même type de source géotracker » sont REFUSÉES** (4 cliques, jusqu'à
  4 755 puits, degré max 4 754, 19,5 M arêtes — ce n'est pas un mécanisme mais une
  partition à 4 classes de la carte ; η avec T1a = **0,081**, aucun signal) ;
- les arêtes **« même sous-bassin SGMA » en clique pleine sont REFUSÉES** : elles sont de
  la **fuite spatiale déguisée que C4 ne rattrape pas** (Cramér V(bloc, sous-bassin) =
  **0,981** ; après coupe C4, **98,7 % des arêtes survivent**, mais leur distance médiane
  est **11,7 km** et **81,6 % dépassent 5 km** — au-delà de la portée d'autocorrélation 2-5 km) ;
- l'**arête défendable** est : **k-NN spatial PLAFONNÉ EN DISTANCE, RESTREINT au même
  sous-bassin** (mécanistique « même aquifère ET co-localisé »), + coupe C4. À cap 2 km,
  k=8 : **32 k arêtes, degré borné, 24 arêtes inter-bloc seulement**.

La partie **hybride (embedding GNN ⊕ XGBoost)** est approuvée **uniquement** avec le schéma
**OOF/nested anti-fuite** spécifié au §3. Sans lui, l'hybride triche trivialement.

**Tant que le contrat §C ci-dessous n'est pas implémenté ET smoke-testé, tout run long est
REFUSÉ.**

---

## 1. CONTRÔLE FUITE — CIBLE (item 1) : OK, vérifié indépendamment

Recalcul (61 colonnes de features `core`, blocklist figée 96 col.) :

| Vérification | Résultat | Verdict |
|---|---|---|
| Intersection features × blocklist | **∅ (vide)** | OK |
| Motifs suspects `_ngL/_detected/label_/sum_pfas/target_/pfas_class` dans les features | **aucun** | OK |
| Colonnes d'arêtes `sgma_subbasin_name`, `nearest_geotracker_type` ∈ blocklist ? | **Non** (contexte) | OK |
| η(`nearest_geotracker_type`, T1a) | **0,081** | non-fuite, et **sans signal** |
| η(`sgma_subbasin_name`, T1a) | **0,505** | non-fuite ; **confondeur géographique** (déjà C5) |
| η(`gm_dataset_name`, T1a) | **0,252** | confondeur de design → **C6, exclu des features** |

**Conclusion.** Ni les features tabulaires (XGB) ni les features de nœud (GNN) ne fuient la
cible. Les colonnes qui DÉFINISSENT les arêtes sont du contexte, pas des dérivés de cible.
**MAIS** : `sgma_subbasin_name` est le **proxy géographique** à η=0,505 — précisément la
collision **C5**. L'utiliser pour construire des arêtes longues réinjecte la carte (voir §2).
**Risque confondeur de design** à auditer en aval (SHAP, non bloquant ici) : `gm_dataset_name`
WB_CLEANUP, et `nearest_geotracker_type` comme proxy « site déjà pollué ». Comme η=0,081, ce
dernier n'apporte de toute façon rien — argument supplémentaire pour ne pas en faire une arête.

---

## 2. CRUX — ARÊTES MÉCANISTIQUES SOUS CV SPATIALE PAR BLOCS (item 2)

### 2.1 Décision : la coupe C4 s'applique OBLIGATOIREMENT aux arêtes mécanistiques

Toute arête (spatiale OU mécanistique) dont les extrémités tombent dans deux blocs CV
différents fait transiter l'information train→test par message passing → invalide la CV
spatiale. **C4 est non négociable pour TOUTES les relations** (déjà la règle phase 1/2).

### 2.2 Comptage empirique d'arêtes (graine 42, k=8 KMeans, niveau puits)

**Sources géotracker (`nearest_geotracker_type`, 4 catégories, 0 % manquant) :**

| | valeur |
|---|---|
| Distribution | Chrome Plater 4 755 · Bulk Terminal 3 175 · Airport 2 261 · Refinery 1 142 |
| Cliques (groupes ≥ 2) | **4** · plus grande clique = **4 755 puits** |
| Arêtes totales (co-membership plein) | **19 547 801** |
| Inter-bloc (coupées par C4) | **15 308 094 (78,3 %)** |
| Intra-bloc (survivent à C4) | **4 239 707 (21,7 %)** |
| Degré max / moyen | **4 754 / 3 450** |

→ **REFUSÉ.** Quatre cliques géantes ≠ mécanisme « même source ». Sans `facility_id`, « même
type de source » ne distingue pas deux aéroports à 600 km. Le degré (jusqu'à 4 754) noierait
le message passing (sur-lissage immédiat). Et η=0,081 : aucun signal cible. **À écarter.**

**Sous-bassins SGMA (`sgma_subbasin_name`, 237, 13,0 % manquant au niveau puits) :**

| | valeur |
|---|---|
| Cliques (groupes ≥ 2) | **220** · plus grande = **636 puits** |
| Arêtes totales (co-membership plein) | **1 135 981** |
| Inter-bloc (coupées par C4) | **15 314 (1,3 %)** |
| Intra-bloc (survivent à C4) | **1 120 667 (98,7 %)** |
| Degré max / moyen | **635 / 200** |

→ **Co-membership PLEIN REFUSÉ — c'est de la fuite spatiale que C4 NE RATTRAPE PAS.** Preuve :

### 2.3 La co-appartenance sous-bassin APRÈS coupe C4 ≈ structure de la carte

- **Cramér V(bloc KMeans, sous-bassin) = 0,981.** Le bloc spatial détermine quasiment le
  sous-bassin → « même sous-bassin » et « même bloc » sont presque la même variable
  (cohérent : KMeans sur (lat,lon) recoupe les aquifères). C'est la collision **C5**.
- C4 ne coupe que **1,3 %** des arêtes sous-bassin, parce que la frontière de bloc tombe
  rarement DANS un sous-bassin. Les **98,7 % qui survivent ne sont donc PAS locales** :
  - distance des paires intra-bloc même-sous-bassin : **médiane 11,7 km**, p75 = 18,4 km,
    p95 = 37,9 km, **max 86 km** ;
  - **81,6 % dépassent 5 km** (au-delà de la portée d'autocorrélation mesurée 2-5 km),
    **21,1 % dépassent 20 km** ;
  - seulement **2,5 %** sont déjà dans le graphe k-NN spatial 1,5 km.

**Lecture.** La clique sous-bassin ajoute massivement des **arêtes longues (5-86 km)** que le
graphe spatial 1,5 km exclut justement comme « ré-encodage de la carte ». Comme elles ne
franchissent presque jamais une frontière de bloc, **C4 les laisse passer** : on rouvre la
fuite spatiale par une porte que le garde-fou C4 ne surveille pas. Inacceptable.

### 2.4 Arête mécanistique DÉFENDABLE : k-NN spatial RESTREINT au sous-bassin + cap distance

On garde l'**intention mécaniste** (« même aquifère sous-bassin ») mais on impose le **même
plafond physique que le graphe spatial** (portée 2-5 km, C4 §2.3). Concrètement : pour chaque
puits, relier ses **k plus proches voisins QUI PARTAGENT le sous-bassin**, avec un **cap de
distance dur**, puis coupe C4. Mesuré :

| Construction | arêtes | inter-bloc (coupées) | survivent |
|---|---|---|---|
| k-NN intra-sous-bassin, k=8, cap **2 km** | **32 194** | **24 (0,1 %)** | 32 170 |
| k-NN intra-sous-bassin, k=8, cap **5 km** | 41 925 | 91 (0,2 %) | 41 834 |
| k-NN intra-sous-bassin, k=8, cap **10 km** | 46 361 | 242 (0,5 %) | 46 119 |

→ Degré borné (k=8), arêtes locales, quasi rien à couper. **C'est le seul design d'arête
mécanistique admissible.** Il se distingue du k-NN spatial pur par la **contrainte de
co-appartenance au sous-bassin** (deux puits à 1 km de part et d'autre d'une crête
hydrogéologique ne sont PAS reliés) — un vrai prior d'aquifère, pas un re-encodage de la carte.

> ⚠️ **Honnêteté requise.** Avec cap 1,5–2 km, ce graphe partage ~la plupart de ses arêtes
> avec le k-NN spatial nu : l'apport mécanistique sera **faible**. C'est attendu sans
> `facility_id` ni gradient hydraulique. L'implémenteur doit le **dire**, pas le maquiller :
> rapporter le **Δ(spatial, random)** de chaque graphe et ne **revendiquer un gain que s'il
> dépasse le bruit inter-pli (~0,03)** ET reste sur le score SPATIAL.

### 2.5 Pas de connectivité hydrologique dirigée (fait acté)

Aucune colonne de **direction d'écoulement / gradient**. Un arc amont→aval n'est **pas
constructible**. La seule « connectivité » est la **co-appartenance non orientée au
sous-bassin** — traitée en §2.4. **Ne pas prétendre** à une connectivité hydrologique
dirigée dans le rapport.

---

## 3. CRUX — EMBEDDING INDUCTIF SANS FUITE → XGBoost (item 3)

C'est là qu'un hybride triche le plus facilement. Contrat strict, par étape.

### 3.1 (a) Comment le GNN est entraîné
GraphSAGE **inductif**, **supervisé** sur la cible **T1a niveau puits = majorité** des
prélèvements du puits (`graph.well_majority_target`, déjà figé). Pas d'auto-supervision
imposée (option ultérieure), mais **la cible vue par le GNN et celle ajustée par XGBoost
sont la MÊME** → l'embedding ne doit JAMAIS être produit sur des nœuds dont le label a servi
à l'entraîner ET sera ensuite ré-appris par XGB sur les mêmes lignes (double usage = fuite
d'optimisme). D'où le schéma OOF imbriqué obligatoire ci-dessous.

### 3.2 (b) Embeddings train pour XGBoost : OOF NESTED obligatoire
Pour produire les **features-embedding des lignes d'entraînement de XGBoost sans que
l'embedding ait vu le label de ces mêmes lignes** :

```
Pour chaque pli EXTERNE spatial f (test = bloc f) :
    train_f = puits des blocs ≠ f
    # --- embeddings OOF pour les lignes de train_f (anti-fuite) ---
    découper train_f en J plis INTERNES spatiaux-groupés (micro-blocs assemblés, C3)
    Pour chaque pli interne j :
        fit_j  = train_f \ val_j           (puits)
        graphe_j = build_well_graph(fit_j ∪ val_j, fold_block=bloc interne,
                                    cut_blocks=True)         # C4 : arêtes val_j↔fit_j coupées
        GNN_j = entraîner SUR fit_j SEULEMENT (perte masquée sur fit_j ; early-stop sur
                 un sous-bloc de fit_j, JAMAIS sur val_j)
        emb_OOF[val_j] = embedding(GNN_j, val_j)   # val_j n'a jamais été vu en label NI en arête
    # -> chaque ligne de train_f a un embedding produit par un GNN qui n'a pas vu son label
    X_train_f = [features tabulaires(train_f) ⊕ emb_OOF(train_f)]
    XGB_f = entraîner sur X_train_f (cible T1a niveau ligne, class_weight équilibré)
```

**Règle dure.** L'embedding d'une ligne d'entraînement de XGB provient **toujours** d'un GNN
qui **n'a pas inclus cette ligne (ce puits) dans sa perte ni dans son voisinage d'arêtes**.
Sinon le GNN encode le label que XGB ré-apprend → AUC gonflée, non reproductible hors échantillon.

### 3.3 (c) Embeddings test : aucun label ni arête du bloc test
```
    # --- embedding des lignes de TEST (bloc f) ---
    graphe_ext = build_well_graph(train_f ∪ test_f, fold_block=bloc externe, cut_blocks=True)
                 # C4 : TOUTE arête bloc_f ↔ autre bloc coupée (spatiale ET sous-bassin)
    GNN_ext = entraîner sur train_f SEULEMENT (perte masquée train_f)
    emb_test = embedding(GNN_ext, test_f)      # test_f : 0 label vu, 0 arête cross-block
    proba_test = XGB_f.predict([features(test_f) ⊕ emb_test])
    ASSERT n_removed_cross_block(graphe_ext) couvre 100 % des arêtes bloc_f↔reste
           (spatiale ET sous-bassin) ; ASSERT 0 arête cross-block restante.
```
Le puits test n'agrège QUE depuis ses voisins **train du même bloc** (arêtes intra-bloc) →
inductif vis-à-vis du bloc test, exactement ce que mesure le score spatial. **Aucun label de
test, aucune arête de test→train.**

### 3.4 Garde-fous d'implémentation (à asserter dans le smoke-test)
- `assert_no_group_leak` sur les plis externes ET internes (C2).
- Bloc spatial test ∉ blocs d'entraînement (externe et interne).
- `n_removed_cross_block` reporté par pli pour **chaque type d'arête** ; **0 arête
  cross-block restante** sur tous les plis (spatiale + sous-bassin).
- `FeaturePipeline` (features de nœud) et tout encodage/imputation **fit sur les puits
  d'entraînement du pli SEULEMENT** ; encodage fréquentiel (sans cible) ou target-encoding
  **OOF interne** — jamais sur le test.
- Embedding extrait **avant la tête** (couche cachée), dimension figée et journalisée.
- Comparabilité : **mêmes plis, même graine (42), même `fold_block`** pour GNN / XGB / hybride.

---

## 4. TEST INDÉPENDANT & PROTOCOLE DE COMPARAISON (item 4)

### 4.1 Découpages
- **Référence** : **CV spatiale par blocs** (`splits.spatial_block_folds`, KMeans k=8 niveau
  puits) en leave-one-block-out. Pour la **comparaison appariée** des 3 modèles, schéma à
  **≥ 5 plis spatiaux comparables** (micro-blocs assemblés, C3) — sinon 8 points LOBO sont
  sous-puissants. Le classement des modèles se fait **sur le score SPATIAL uniquement**.
- **Random (pour le Δ seulement)** : `splits.group_random_folds` (GroupKFold par puits,
  voisins potentiellement séparés). Jamais pour classer un modèle ; **uniquement** pour
  calculer Δ = score_random − score_spatial = test d'artefact spatial.

### 4.2 Scoring NIVEAU LIGNE pour comparabilité stricte au mur
Les 3 modèles (GNN seul, XGB seul, hybride) sont scorés **au niveau prélèvement** (46 338
lignes), proba puits **rediffusée** à chaque prélèvement (`row_to_node`) — exactement comme
le mur non-graphe (RF spatial 0,601 / XGB 0,588 ; random ~0,90) et la phase 1. Pour XGB seul
et l'hybride, qui peuvent être nativement au niveau ligne, **agréger leur proba au niveau
puits puis rediffuser**, OU scorer ligne directement, **mais le même choix pour les trois**.
Documenter le choix une fois pour toutes.

### 4.3 Métriques (identiques pour les 3 modèles, cohérentes inter-expériences)
ROC-AUC · **rappel @ seuil** + précision · **balanced accuracy** · **Brier + ECE + courbe de
fiabilité** · **gain cumulé / lift @ k%**. PR-AUC en complément (T1a quasi-équilibrée 44,5 %).
**Toutes rapportées en SPATIAL** ; ROC-AUC et PR-AUC aussi en random pour le Δ.

### 4.4 Seuil & calibration — OOF UNIQUEMENT
Seuil de décision **F1/coût-optimal sur les probas OOF de validation interne**, jamais sur le
test. Recalibration (Platt/Isotonic) **fittée OOF**, jamais sur le test. Brier/ECE rapportés
**en spatial** (c'est là que la calibration se dégrade hors distribution).

### 4.5 Significativité
- IC95 % **bootstrap PAR GROUPE (`gm_well_id`)** sur les OOF concaténés du test spatial
  (jamais par ligne → IC faussement étroits par pseudo-réplicats).
- Test apparié sur les plis : **Nadeau-Bengio (corrected resampled t-test)** et/ou
  **Wilcoxon signé** sur les paires de scores par pli.
- **Seuil de réalité** : un gain hybride vs mur n'est **réel** que s'il est (a) significatif
  au test apparié ET (b) **> bruit inter-pli (~0,03 d'AUC spatiale)**. Rappel phase 1 : les
  GNN seuls étaient à +0,017/+0,023 sur le mur spatial = **dans le bruit**. L'hybride doit
  **franchir** ce seuil pour être revendiqué.
- Positionnement Dong et al. 2024 : comparer le **spatial** au chiffre publié en **signalant
  l'écart de protocole** (Dong en split aléatoire ≈ non comparable directement).

---

## C. CONTRAT D'IMPLÉMENTATION (bloquant — gnn-researcher + tabular-ml-engineer)

1. **Cible** : `targets.build_T1a` (garde-fou détection C1, prévalence ligne 44,5 %). Niveau
   ligne à l'éval ; niveau puits-majorité à l'entraînement GNN.
2. **Features** : `config.feature_columns(include_location=False, cocontam="core")`, blocklist
   96 figée. **lat/lon JAMAIS en feature de nœud** (C6, géographie via arêtes). `gm_dataset_name`
   exclu (C6). Mêmes features tabulaires pour XGB seul et le bloc tabulaire de l'hybride.
3. **Arêtes — REFUS et OBLIGATIONS** :
   - **INTERDIT** : arêtes « même type de source géotracker » (4-cliques, η=0,081, REFUSÉ).
   - **INTERDIT** : arêtes « même sous-bassin » en clique pleine (fuite spatiale longue que
     C4 ne coupe pas ; Cramér V=0,981 ; 81,6 % des paires > 5 km).
   - **AUTORISÉ** : (i) k-NN spatial nu cap **1,5 km** (baseline graphe phase 1) ;
     (ii) **k-NN intra-sous-bassin, k=8, cap ≤ 2 km** (mécanistique défendable). Comparer les
     deux ; rapporter le Δ. Tout autre graphe repasse par `eval-methodologist`.
   - **C4 sur TOUTES les relations** : `graph.cut_cross_block` par pli ; assert 0 arête
     cross-block restante (spatiale ET sous-bassin).
4. **Hybride** : schéma OOF/nested du §3 — embedding train **OOF interne**, embedding test
   d'un GNN n'ayant vu **ni le label ni les arêtes** du bloc test. Embedding pré-tête.
5. **Splits** : externe = spatial-block (≥ 5 plis comparables pour la comparaison ; LOBO k=8
   et LeaveOneRegionOut en robustesse). Random = Δ seulement. **Tout groupé `gm_well_id`**,
   graine 42, partout (externe, interne, seuil, calibration).
6. **Métriques** : §4.3, identiques pour GNN / XGB / hybride, niveau ligne, comparables au mur.
   Seuil + calibration **OOF**. Triplet **(random, spatial, Δ)** par modèle et par métrique.
7. **Significativité** : §4.5 (bootstrap par groupe, Nadeau-Bengio/Wilcoxon, seuil 0,03).
8. **Smoke-test CPU < 3 min** AVANT Colab (CLAUDE.md §5) : charge, formes, graphe (n nœuds,
   arêtes ≤ cap, **0 arête cross-block** par relation), perte finie, embedding extrait,
   XGB fitte sur features⊕embedding, métriques calculées, asserts anti-fuite (§3.4) verts.
9. **Run lourd = Colab GPU uniquement** (CLAUDE.md §4, mémoire). Aucun entraînement local.

---

## Annexe — chiffres reproductibles (graine 42, k=8 KMeans niveau puits)

- Blocs k=8 (puits) : 897 / 1 676 / 3 163 / 891 / 1 621 / 1 023 / 1 586 / 476.
- `sgma_subbasin_name` : 237 valeurs (niveau puits), 13,0 % manquant.
- `nearest_geotracker_type` : Chrome Plater 4 755 · Bulk Terminal 3 175 · Airport 2 261 ·
  Refinery 1 142 (0 % manquant).
- Clique source-type : 4 cliques, 19,55 M arêtes, max degré 4 754, C4 coupe 78,3 %.
- Clique sous-bassin pleine : 1,14 M arêtes, max degré 635, C4 coupe **1,3 %** seulement.
- Cramér V(bloc, sous-bassin) = **0,981**.
- Paires intra-bloc même-sous-bassin : médiane 11,7 km, **81,6 % > 5 km**, 2,5 % dans le
  k-NN 1,5 km, 21,1 % > 20 km.
- k-NN intra-sous-bassin cap 2 km, k=8 : 32 194 arêtes, **24 inter-bloc (0,1 %)**.
- k-NN spatial nu cap 1,5 km, k=8 (post-C4) : 30 477 arêtes.
- η(`nearest_geotracker_type`, T1a)=0,081 · η(`sgma_subbasin_name`, T1a)=0,505 ·
  η(`gm_dataset_name`, T1a)=0,252.
- Features × blocklist = ∅ ; aucun motif suspect dans les 61 features `core`.

Tous obtenus par checks CPU < 3 min (graine 42), sans entraînement. Réutilisent `src/{config,
splits,graph,targets}.py` figés.
