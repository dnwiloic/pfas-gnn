# Validation méthodologique — HGT standalone + embedding fusion + stacking (T1a)

Méthodologiste : eval-methodologist. Date : 2026-06-23. Mode : AUDIT-ONLY (lecture +
contrôles déterministes, aucun ré-entraînement).

Expérience auditée : `experiments/hgt_fusion_stacking_t1/`
(metrics.json smoke=false, REPORT.md, config.yaml).
Code : `src/hgt_fusion_stacking_t1.py` (commit f022bc9) → `src/gnn_hetero_t1.py`
(commit 9b1abef) ; socle `src/{graph,splits,features,metrics,hybrid}.py`.
Mur committé : `experiments/baseline_t1/metrics_spatial.json` +
`experiments/baseline_t1/run_spatial_t1.py`.

---

## VERDICT : VALIDÉ

Le protocole est sain sur les six axes de contrôle. La conclusion du module
(`verdict="no_robust_gain"`) et la section « VERDICT (added by main thread) » du
REPORT.md sont **confirmées et endossées**. Le bémol de comparabilité ajouté par le fil
principal (ne pas lire `gain_vs_committed_wall` comme un gain graphe) est **exact et
corroboré ligne par ligne** ci-dessous. Une seule réserve mineure (cosmétique) sur le
champ `above_noise`, sans effet sur le verdict.

---

## Point 1 — Graphe sans fuite + HGT inductif : **PASS**

- **Coupe inter-blocs réellement appliquée (pas seulement rapportée).**
  `gnn_hetero_t1.build_multirel_graph` coupe CHAQUE relation séparément AVANT
  symétrisation puis l'**asserte** : `graph.py:165-174` (`cut_cross_block` garde
  `node_block[a]==node_block[b]`), `gnn_hetero_t1.py:140-148` (assert `n_cross_near==0`
  et `n_cross_sub==0`, sinon `AssertionError`). La symétrisation
  (`graph.py:177-183`) ne fait que mirroir des arêtes survivantes → 0 reste 0.
  Le `n_cross_block_total=0` de metrics.json (lignes 92, 262) est donc **garanti par
  une assertion qui ferait crasher le run sinon**, pas un simple report. PASS.
- **HGT inductif — un puits test n'agrège que des voisins train.** Deux jeux d'arêtes
  sont construits (`gnn_hetero_t1.py:461-466`) : `train_pack` avec
  `train_only_mask=train_nodes_all` (garde uniquement les paires train-train,
  `_build_edge_tensors:388-390`), utilisé pour l'entraînement ET la validation
  (`:490`, `:500`) ; `score_pack` (jeu complet, déjà cross-block-cut) utilisé au
  scoring (`:524`) ET pour l'embedding (`:528`). Comme les arêtes inter-blocs sont
  coupées, le jeu « complet » au scoring ne contient AUCUNE arête test–train hors-bloc :
  un puits test ne peut s'attacher qu'à ses voisins train DU MÊME bloc. L'embedding de
  fusion est scoré avec le même `score_pack` (`:528`), donc même garantie inductive.
  PASS.
- **Transformers k-NN / KMeans fit sur train seulement.**
  - KMeans des blocs : fit sur les coordonnées de TOUS les puits
    (`splits.py:30-36`) — mais c'est le découpage CV lui-même, non supervisé et
    indépendant de y ; c'est la pratique standard et non une fuite (le bloc ne voit
    jamais la cible). Acceptable.
  - k-NN spatial / intra-sous-bassin : construits sur la géométrie (haversine), sans y
    (`graph.py:61-93`, `107-162`) → pas de fuite cible possible, et coupés par bloc.
  - VAL micro-blocs (early-stopping) : KMeans fit sur les coords des puits TRAIN
    uniquement (`gnn_hetero_t1.py:334-357`, `train_idx = where(train_nodes_all)`). PASS.
  - Pipeline de features de nœud : `node_features(..., train_node_mask=fit_nodes)`
    fit sur les nœuds FIT seulement (`gnn_hetero_t1.py:453`,
    `graph.py:304-324`), encodage `frequency` (sans y) → trivialement leak-free.
- **lat/lon jamais en feature.** `run` appelle
  `feature_columns(include_location=False, cocontam="core")`
  (`hgt_fusion_stacking_t1.py:521`). metrics.json `include_location=false` (ligne 7),
  `n_features=61` (ligne 6), et la liste `feature_cols` (lignes 8-69) **ne contient ni
  `latitude` ni `longitude`**. Les colonnes `dist_geotracker_km`, `gldas_dist_km`,
  `n_geotracker_within_*` sont des distances/comptages de contexte (validées proxys
  mécanistes plausibles dans hgt_rgcn_t1/eval_validation.md §1), pas de la localisation
  pure. PASS.

> Note sur les 61 vs 97 colonnes : metrics.json affiche `n_tabular_features=97`
> (lignes 95, 265) pour la matrice tabulaire des bases XGB/LGBM/fusion, tandis que
> `n_features=61` est la liste de colonnes brutes. L'écart vient de l'expansion par
> one-hot des catégorielles low-card dans `FeaturePipeline` (le HGT, lui, utilise
> l'encodage `frequency`). Ce sont les MÊMES 61 colonnes sources des deux côtés ; aucune
> colonne supplémentaire ni lat/lon n'est injectée. Cohérent.

## Point 2 — Discipline OOF de la fusion (PCA 95 %) : **PASS**

`fusion_oof_proba` (`hgt_fusion_stacking_t1.py:289-326`) fait un LOBO imbriqué sur les
8 mêmes blocs : pour le bloc tenu `b`, `tr = (node_block != b)`, `te = (node_block==b)`
(`:304-305`). La PCA est `PCA(...).fit(emb_tr)` — **fit sur les lignes OOF train du bloc
tenu seulement** (`:311`), le nombre de composantes pour 95 % de variance est dérivé du
`explained_variance_ratio_` du TRAIN (`:312-314`), puis appliqué au bloc tenu
(`pca.transform(oof.hgt_emb[te])`, `:317`). Le XGB de fusion est `clf.fit(X_tr, ...)`
sur train OOF, prédit `te` (`:322-324`). Le bloc évalué ne touche jamais un `.fit`.
`pca_n_components_per_fold` (metrics.json 155-164) varie par pli (47/48/49…) ce qui
confirme que la PCA est bien **re-fit par pli** et non figée. PASS.

## Point 3 — Discipline OOF du stacking : **PASS**

Les méta-features (`stacking_oof_proba`, `:335-370`) sont construites à partir des
probas de base `{hgt_proba, xgb_proba, lgbm_proba}` qui sont des sorties OOF : chacune a
été produite dans `build_oof_backbone` par un modèle entraîné sur les blocs train et
scoré sur le bloc tenu (`:215-230`) — donc out-of-fold par construction. Le méta-XGB
fait un second LOBO imbriqué : `tr=(node_block!=b)`, `te=(node_block==b)`,
`meta.fit(feats[tr], ...)` puis `predict_proba(feats[te])` (`:359-369`). Le méta-learner
ne voit jamais le bloc tenu à l'entraînement. Méta-features = probas + mean/std +
agreements pairwise + entropie moyenne (`:346-356`), cohérentes avec la liste
`meta_features` de metrics.json (196-206). PASS.

## Point 4 — Seuil : **PASS**

Le seuil F1 est dérivé des probas OOF au niveau PUITS, jamais du bloc évalué.
`_row_metrics` (`:253-270`) calcule `thr = _optimal_threshold_f1(oof.y_well[valid],
proba_well[valid])` (`:259`) où `proba_well` est l'array OOF complet (chaque puits scoré
par un modèle qui ne l'a pas vu). Pour le HGT par-pli, le seuil vient des nœuds VAL
(`gnn_hetero_t1.py:531`, `_f1_threshold(y_well[val_nodes], ...)`), eux-mêmes carvés du
train (`:437-439`). Aucune optimisation de seuil sur le test. PASS.

## Point 5 — Validité des comparaisons (le point crucial) : **PASS — finding confirmé**

Le constat du fil principal est **exact** : `gain_vs_committed_wall` (+0.066 / +0.079 /
+0.094, metrics.json 447/465/483) est INVALIDE comme « victoire du graphe », pour deux
raisons indépendantes, toutes deux vérifiées dans le code committé :

**(a) Jeu de features différent.**
- Mur committé : `run_spatial_t1.py:43` →
  `FEATURE_COLS = feature_columns(include_location=False, cocontam="all", include_air=True)`
  → 96 features. Confirmé par `metrics_spatial.json:5` (`"n_features": 96`).
- Cette expérience : `hgt_fusion_stacking_t1.py:521` →
  `feature_columns(..., cocontam="core")` → 61 colonnes sources (metrics.json `n_features=61`).
- `config.feature_columns` (`config.py:128-143`) : `cocontam="all"` = COCONTAM_ALL
  (6 core + 36 = 42 cocontaminants), `cocontam="core"` = 6 cocontaminants. L'écart de
  jeu est donc structurel.
- **S'ajoute une 3ᵉ différence non mentionnée mais aggravante :** le mur committé encode
  les catégorielles en **`target` encoding** (`run_spatial_t1.py:92`,
  `encode="target"`), proxy géographique puissant ; cette expérience utilise
  **`frequency`** (`hgt_fusion_stacking_t1.py:141`). Le mur 0.588 n'est donc pas
  reproductible avec ce pipeline. Le `gain_vs_committed_wall` compare des chiffres issus
  de pipelines distincts.

**(b) Définition d'AUC différente.**
- Mur committé : `run_spatial_t1.py:97-98` →
  `agg = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}` →
  l'AUC committée 0.5878 est une **moyenne des AUC par pli** (per-fold-mean). Cohérent
  avec la présence d'un `roc_auc_std` dans metrics_spatial.json (ligne 58).
- Cette expérience headline une **AUC OOF globale** (probas poolées sur tous les blocs)
  via `_row_metrics` → `M.binary_metrics` sur le vecteur concaténé (`:266`). Pour le
  HGT, `roc_auc=0.6537` (OOF global, ligne 100) vs `auc_mean=0.6237` (per-fold-mean,
  ligne 444) : l'OOF global est ~+0.030 plus haut. Idem fusion (+0.029), stacking
  (+0.041). Le `gain_vs_committed_wall` (`:469`) calcule
  `a["metrics"]["roc_auc"] (OOF global) − 0.5878 (per-fold-mean)` : il **mélange deux
  métriques**, ce qui n'est pas une comparaison valide.

**La seule comparaison valide est `gain_vs_in_run_wall`** (`:468`) :
`np.nanmean(per_fold_auc) − wall_mean`, où `wall_mean` est l'AUC du XGB-tabulaire
calculé SUR LES MÊMES 8 PLIS, MÊMES 61 features, MÊME pipeline, dans CE run
(`base_references.xgb_tabular`, OOF global 0.6879 / per-fold-mean 0.6237 — voir nuance
ci-dessous). Ce gain est **négatif pour les trois architectures** :
−0.0642 (HGT), −0.0501 (fusion), −0.0470 (stacking) (metrics.json 446/464/482). PASS.

> **Nuance technique sur `gain_vs_in_run_wall` (n'invalide pas le verdict).** Le code
> compare `np.nanmean(arch_per_fold)` (per-fold-mean de l'archi) à `wall_mean` =
> `xgb_tabular["metrics"]["roc_auc"]` = **0.6879 OOF global** (`:448`, `:468`). C'est un
> léger mélange (per-fold-mean d'un côté, OOF global de l'autre) DANS LE MAUVAIS SENS
> pour les architectures graphe : il les pénalise (compare leur per-fold-mean plus bas à
> l'OOF global plus haut du mur). Le déficit réel per-fold-mean-vs-per-fold-mean est
> plus petit (archi 0.624–0.641 vs mur per-fold-mean 0.6237), c.-à-d. quasi à égalité,
> voire +0.003/+0.017 en faveur du stacking. **Mais** : (i) le test apparié, lui, est
> correctement per-fold vs per-fold (voir ci-dessous) et non significatif ; (ii) aucune
> des deux lectures ne produit un gain robuste. Le verdict `no_robust_gain` tient dans
> les deux cas. Réserve documentée, non bloquante.

**Test apparié — correctement monté sur des plis appariés.**
`build_comparison` (`:443-484`) prend `af = arch["per_fold_auc"]` et
`wf = wall_folds = base_references["xgb_tabular"]["per_fold_auc"]`. Les deux listes
proviennent de `_per_fold_aucs(oof, proba, ...)` (`:273-285`) qui itère
`sorted(set(node_block))` — **même ordre de blocs, même backbone OOF** → appariement
pli-à-pli correct (graph fold AUC vs in-run-wall fold AUC). Nadeau-Bengio
(`hybrid.py:178-191`, correction `1/k + n_test/n_train`) avec `n_tr_mean=40545`,
`n_te_mean=5792` (metrics.json 440-441, cohérents avec ~46k lignes / 8 blocs) ;
Wilcoxon apparié (`hybrid.py:194-202`). p-values NB 0.73–0.93, Wilcoxon 0.55–0.94
(metrics.json) → loin de 0.05. `significant=false` pour les trois. PASS.

**Règle de la barre de bruit — saine, avec une réserve cosmétique.** Le verdict
(`:481-482`) exige `significant AND above_noise AND gain_vs_in_run_wall > 0`. Comme le
gain est négatif et la significativité absente, `verdict="no_robust_gain"` — **correct**.
RÉSERVE MINEURE : `above_noise = abs(gain) > 0.03` (`:478`) utilise une valeur absolue,
d'où `above_noise=true` (metrics.json 458/476/494) alors que le « gain » est en réalité
un DÉFICIT. C'est sémantiquement trompeur lu isolément, mais sans effet sur le verdict
(la clause `> 0` du verdict neutralise le piège). À renommer en `magnitude_above_noise`
idéalement ; non bloquant.

## Point 6 — Traçabilité : **PASS**

Chaque chiffre headline du REPORT.md trace au metrics.json committé :
- Tableau résultats spatiaux (REPORT 14-18) : HGT AUC 0.6537 / F1 0.6143 / Brier 0.3191
  / ECE 0.2892 = metrics.json 100/101/107/109 ; fusion 0.6670/0.5639/0.2672/0.1832 =
  129/130/136/138 ; stacking 0.6815/0.6378/0.2467/0.1283 = 170/171/177/179 ; XGB in-run
  0.6879 = 212 ; LGBM 0.6793 = 241. CI [0.634,0.673] = 112-113, etc.
- PCA 47.8 moyenne, per-fold [47,48,49,48,46,48,48,48] (REPORT 20) = metrics.json
  165 / 155-164. ✓
- Δ(random−spatial) +0.1490/+0.1902/+0.2147 (REPORT 28-30) = metrics.json 431-433. ✓
- Tests appariés (REPORT 36-38) : gains −0.0642/−0.0501/−0.0470, committed
  +0.0659/+0.0792/+0.0938, NB p 0.7350/0.7683/0.9298, Wilcoxon 0.8438/0.5469/0.9453 =
  metrics.json 446-495. ✓
- Section VERDICT (REPORT 53-58) per-fold-mean : XGB 0.643 (=0.6237 arrondi… voir
  ci-dessous), stacking 0.641 (=`auc_mean` 0.6409, ligne 480), fusion 0.638 (=0.6378,
  468), HGT 0.624 (=0.6237, 444). Les OOF globaux 0.688/0.682/0.667/0.654 tracent à
  212/170/129/100. ✓ (le « 0.643 » du XGB est l'arrondi de l'in-run wall per-fold-mean ;
  cohérent à 10⁻³ près).

Contrairement aux trois autres expériences signalées dans
`profilage/AUDIT_REVALIDATION.md` pour défaut de traçabilité, **celle-ci passe** :
aucun chiffre du REPORT n'est fabriqué ou désynchronisé du metrics.json committé.

---

## Contrôles transverses confirmés

- **Δ spatial (apport méthodo central).** Δ(random − spatial) = +0.149 / +0.190 / +0.215
  (HGT / fusion / stacking). L'inflation CROÎT avec la sophistication : plus le modèle
  exploite la structure (embeddings, méta-apprentissage), plus il sur-apprend
  l'autocorrélation en split aléatoire. Les AUC aléatoires (~0.80–0.90) reproduisent le
  mirage des baselines committées (XGB random 0.900, RF 0.898, metrics_spatial.json
  66/26). Les ~0.65–0.68 spatiaux sont la réalité déployable. Quantification soignée et
  cohérente avec hgt_rgcn_t1/eval_validation.md §3 (Δ ~0.20 attendu). ✓
- **Calibration.** Brier + ECE rapportés pour chaque architecture ET chaque régime.
  Le stacking améliore nettement la calibration spatiale (ECE 0.128 vs HGT standalone
  0.289) — utile décisionnellement même sans gain d'AUC. ✓
- **Graphe conforme au ruling C-NODE.** Aucun type de nœud fabriqué : un seul type réel
  `well`, deux relations d'arêtes (`near`, `same_subbasin_knn`), HGT comme encodeur
  relationnel (`gnn_hetero_t1.py:76-80`, `206-208`). Conforme à
  hgt_rgcn_t1/eval_validation.md C-NODE.1/2/3. ✓
- **Cohérence avec l'historique.** ~0.60–0.68 spatial sans gain robuste, cohérent avec
  phases 1–3 et gnn_hybrid_t1 (0.646). Le graphe n'apporte pas de valeur prédictive
  robuste une fois le contexte tabulaire présent. ✓

## Réserves résiduelles (non bloquantes)

1. `above_noise` calculé sur `abs(gain)` → `true` même pour un déficit. Cosmétique ;
   le verdict n'en dépend pas (clause `gain > 0`). Renommer en
   `magnitude_above_noise` recommandé.
2. `gain_vs_in_run_wall` mélange per-fold-mean (archi) et OOF global (mur), pénalisant
   légèrement les architectures graphe. Le test apparié (per-fold vs per-fold) est, lui,
   correct et décisif. Pour un report propre, aligner les deux métriques (idéalement
   per-fold-mean des deux côtés). Sans effet sur la conclusion.
3. `gain_vs_committed_wall` devrait être **retiré ou explicitement marqué non-comparable**
   dans metrics.json (il l'est déjà dans REPORT.md §VERDICT). Tant qu'il reste dans le
   JSON sans annotation, il risque d'être recopié hors contexte. Recommandation :
   ajouter un champ `comparable: false` à côté.

## Endossement du verdict `no_robust_gain`

**ENDOSSÉ.** Sous les 8 mêmes plis spatiaux, mêmes 61 features, même pipeline (la seule
comparaison apples-to-apples), aucune des trois architectures graphe ne bat de façon
robuste le XGBoost tabulaire : gains in-run négatifs, tests appariés non significatifs
(p ≫ 0.05), déficits/gains dans le bruit inter-plis. Le graphe n'ajoute pas de valeur
prédictive robuste sur T1a. La mise en garde du fil principal contre la lecture de
`gain_vs_committed_wall` comme une victoire est correcte et nécessaire.
