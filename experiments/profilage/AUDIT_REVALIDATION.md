# AUDIT_REVALIDATION — re-audit méthodologique (session 2026-06-22)

> Auditeur : `eval-methodologist`. Mandat **AUDIT-ONLY** (aucun ré-entraînement, aucun
> run lourd). Vérification statique de `src/` + des artefacts d'expériences, recoupée au
> jeu `data/CA-PFAS-ASGWS.parquet` (46 338 × 201) via les artefacts de profilage déjà
> calculés. L'exécution Python directe sur le dataset a été refusée par le bac à sable ;
> les vérifications quantitatives s'appuient donc sur les `*_metrics.json` déjà produits
> (graine 42, déterministes) recoupés au code source — ce qui suffit pour ce mandat.

## Verdict global : **VALIDÉ SOUS CONDITIONS**

Le socle anti-fuite (`src/`) est **rigoureux et correct** sur les deux axes (cible,
spatial). La quantification de l'inflation spatiale — apport central du projet — est
**solide là où elle repose sur des runs complets** (baseline_t1, gnn_phase1/phase2,
gnn_hybrid_t1). Deux expériences présentent des **chiffres smoke encadrés comme s'ils
étaient comparables au mur** (hgt_rgcn_t1, baseline_t2), et le `metrics.json` du hybride
contenait un comparatif cassé (déjà réparé hors-bande par `gate5_paired_comparison.py`).
Aucune fuite de cible ni fuite spatiale réelle n'a été trouvée dans le code exécuté.

| Axe d'audit | Verdict | Sévérité |
|---|---|---|
| 1. Fuite de cible | **PASS** | — |
| 2. Fuite spatiale (splits, graphe, Moran) | **PASS** | — |
| 3. Seuil / calibration (OOF only) | **PASS** | — |
| 4. Traçabilité des chiffres | **PARTIAL FAIL** | baseline_t2, hgt_rgcn_t1 |

---

## 1. FUITE DE CIBLE — **PASS**

**Définitions des cibles vérifiées** (`src/targets.py`) :
- T1a = `PFOA>4 ∨ PFOS>4 ∨ HazardIndex≥1`, HI sur {PFHxS, PFNA, HFPO_DA, PFBS}, avec
  **garde-fou de détection** (`_guarded_conc`, `targets.py:20-24` : concentration mise à 0
  si `*_detected==False`). Correct (eval C1).
- T2 = multilabel hybride EPA-MCL / 2,0 ng/L + garde-fou (`build_T2`, `targets.py:49-57`).
  Recalculé depuis `*_ngL`+`*_detected`, **n'utilise jamais `label_*` brut** (qui est
  gonflé par la censure) — conforme à la découverte du profilage (§5).

**Blocklist (96 colonnes) vérifiée contre la structure réelle du jeu** :
- `config.py:48-50` : `LEAKAGE_BLOCKLIST = NGL_COLS(31) + DETECTED_COLS(31) +
  LABEL_COLS(31) + [sum_pfas_ngL, target_sum_gt70, pfas_class_assignment](3)` = **96**,
  assertion `assert len(...)==96` en place. Recoupé à `profile_metrics.json` :
  `family_counts` = 31 ngL + 31 detected + 31 label, plus les 3 dérivés → cohérent.
- **Garde-fou dur** : `FeaturePipeline.__init__` (`features.py:84-86`) **lève** si une
  colonne blocklistée entre dans les features. Toute expérience l'instancie, donc la
  fuite est bloquée *par construction*, pas seulement par convention.

**Re-dérivation du classement de corrélation à T1** (depuis `profile_metrics.json`,
`context_corr_with_T1_top20`, lignes 3270-3351) :
- Top 10 = **toutes des `*_detected`** (PFHpA_detected 0,696 … PFOS_detected 0,459),
  toutes blocklistées.
- **Rang 11 = `well_depth_ft` (−0,354)** — feature de contexte LÉGITIME (profondeur du
  puits, non dérivée d'une mesure PFAS), pas une fuite. Puis d'autres `*_detected`,
  `n_geotracker_within_1km` (0,178), etc.
- ⚠️ **Imprécision REPORT (mineure, non bloquante)** : `profilage/REPORT.md` §0.3 et §4
  affirment « les **16** colonnes les plus corrélées à T1 sont **toutes** des
  `*_detected` ». Faux au sens strict : `well_depth_ft` s'intercale au rang 11. La
  conclusion de fond (toute colonne fortement corrélée est soit `*_detected` blocklistée,
  soit un contexte non-fuitant comme well_depth/geotracker) **tient**. Corriger le
  libellé en « les 10 plus corrélées sont des `*_detected` ; au-delà n'apparaissent que
  des contextes non-fuitants (well_depth_ft #11, geotracker) ».
- `sum_pfas_ngL` (corr ≈ 1,0) n'apparaît pas dans le top car la liste est calculée
  **sur les seules colonnes de contexte** (`profile.py:220-221` exclut ngL/label/sum) —
  c'est attendu, la colonne est blocklistée par ailleurs.

**Features réellement passées aux modèles** : toutes les expériences appellent
`C.feature_columns(...)` (baseline_t1:43, baseline_t2:39, gnn_hybrid:141, gate5:30,
shap:22, hgt_rgcn meta:392-454) qui n'assemble QUE des familles non-fuitantes et
n'inclut **jamais** `gm_dataset_name` (confondeur de design C6) ni lat/lon en feature de
nœud (C-LOC.1). Le `gm_dataset_name` est exclu explicitement et tracé (baseline_t1
REPORT §3-§5 le documente comme la cause principale du 0,97 littérature).

**Certification SHAP (XGB-seul, `shap_xgb_core.json`)** : top driver
`n_geotracker_within_50km`, `dist_geotracker_km` (négatif = plus proche → plus de PFAS,
direction physique correcte), co-contaminants seulement 6,9 %, **`gm_dataset_name`
absent**. Le drapeau rouge « proxy échantillon-labo » est écarté. Deux réserves
honnêtes documentées (`gm_well_category=MONITORING` = sélection de site ; `year` = dérive
temporelle) — à divulguer, non bloquant.

---

## 2. FUITE SPATIALE — **PASS**

**Groupage par puits** (`src/splits.py`) :
- `spatial_block_folds` et `group_random_folds` partitionnent au **niveau puits**
  (`gm_well_id`), donc les pseudo-réplicats temporels d'un même puits ne traversent
  jamais train/test. `assert_no_group_leak` (`splits.py:70-75`) le vérifie et est
  appelé dans les runners (baseline_t1:113).

**CV par blocs spatiaux présente ET rapportée à côté de l'aléatoire** :
- KMeans k=8 au niveau puits (`spatial_block_labels`, `splits.py:30-36`), un puits = un
  bloc. `group_random_folds` (GroupKFold puits, mélange spatial) fournit le bras Δ.
- **Δ(aléatoire − spatial) quantifié partout** (cœur méthodologique) :
  - baseline_t1 (`metrics_spatial.json`, run complet k=8) : **RF ΔAUC +0,297**, XGB
    **+0,313** ; spatial 0,601/0,588 vs aléatoire 0,898/0,900. Δ par métrique fourni.
  - gnn_phase1/phase2 (run complet, smoke=false, 8 blocs) : GraphSAGE Δ **+0,193**,
    GCN **+0,204** (`metrics_p0.json:16,64`).
  - gnn_hybrid_t1 (run complet 8 blocs) : Δ **+0,202** (sain, dans la bande GNN-seul,
    sous les arbres ~0,30 → mémorise moins la carte).

**Construction du graphe — pas de traversée train/test par les arêtes** (`src/graph.py`) :
- k-NN spatial **capé à 1,5 km** (`knn_edges_km`), sous la portée d'autocorrélation
  (2-5 km) → la proximité ne ré-encode pas la carte longue portée.
- **Coupe inter-bloc C4** (`cut_cross_block`, `graph.py:165-174`) : toute arête dont les
  extrémités sont dans des blocs CV différents est supprimée → en LeaveOneBlockOut, les
  nœuds du bloc test sont **isolés** du train. Appliquée aux DEUX relations (spatial,
  subbasin_knn). Invariant **`n_cross_block_remaining == 0` assERTé** (`gnn.py:316,389`)
  et mesuré à 0 dans tous les artefacts (hgt_rgcn `n_cross_block_*: 0` sur tous plis ;
  phase3 `cross_bip=0 ET cross_well=0`).
- lat/lon **jamais** features de nœud ; la géographie n'entre que par les arêtes capées.
- Features de nœud `FeaturePipeline` **fit sur les nœuds FIT uniquement** (`gnn.py:229`,
  qui exclut val ET test ; gnn_hetero:271, gnn_bipartite:190).
- **KMeans/encodeurs** : KMeans ne sert QUE de partition CV (géographie non supervisée,
  non dérivée de la cible) → un fit « train-only » n'est pas requis pour la non-fuite ;
  les transformeurs *supervisés* (target encoder haute-cardinalité) utilisent un KFold
  interne OOF (`features.py:35-57`). Conforme.

**Moran's I** (`profile_metrics.json:3834-3863`) : **0,4262** (k=8 kNN binaire, attendu
sous H0 −0,00025), concordance T1 **0,7674 à 0-1 km** vs base 0,5403. Le REPORT arrondit
à 0,426 — **PASS**. *Caveat de transparence* : la valeur est estimée sur un échantillon
de 4 000 puits (sur 11 333), ce qui est **divulgué** dans la note du JSON ; acceptable.

**Alignement des plis pour le test apparié hybride** (`gate5_paired_comparison.py`) :
hybride et GNN-seul/XGB-core sont alignés par `sorted(set(fold))` sur KMeans k=8 graine
42. L'alignement repose sur le **déterminisme de KMeans** (même graine/données/version
sklearn → même assignation cluster→id). Valide ici, mais **sensible à la version**
sklearn — à figer si reproduction sur autre environnement.

---

## 3. SEUIL / CALIBRATION — **PASS**

- **Seuil F1-optimal toujours sur OOF/validation, jamais sur le test** :
  - baseline_t1 : `oof_threshold` via CV spatiale INTERNE sur le seul train
    (`run_spatial_t1.py:64-79`), seuil appliqué au test fold.
  - GNN : seuil calculé sur les **nœuds VAL** (`gnn.py:290-292`), test scoré une seule
    fois. hgt_rgcn : `threshold_used` varie par pli (0,25-0,55), tous d'origine OOF.
  - hybride : seuil + calibration Platt depuis les **probas OOF internes uniquement**
    (REPORT §1.4 ; guard 8 smoke : « threshold from OOF, not test »).
- **Calibration rapportée** : Brier + ECE + courbes de fiabilité présents
  (hgt_rgcn `reliability_curve` 10 bins ; hybride ECE 0,124 / Brier 0,231 spatial). La
  mauvaise calibration hors-distribution est **divulguée** avec recommandation de
  recalibration OOF avant décision opérationnelle. Bon réflexe.
- **Comparaisons** : IC95 bootstrap **par groupe (`gm_well_id`)** sur l'OOF poolé
  (évite des IC trop étroits par pseudo-réplicats) ; tests appariés Nadeau-Bengio
  (corrigé CV) + Wilcoxon sur les 8 plis ; règle de réalité « gain réel ⇔ significatif
  ET > 0,03 bruit inter-pli ». Méthodologie de comparaison **exemplaire**.

---

## 4. TRAÇABILITÉ DES CHIFFRES — **PARTIAL FAIL** (2 défauts)

### 4a. `gnn_hybrid_t1/metrics.json` — comparatif trois-bras cassé → **RÉPARÉ hors-bande**
Le bloc `three_way_comparison` du run committé est **non concluant par construction** :
- `paired_hybrid_vs_xgb` et `paired_hybrid_vs_gnn` = **`{}`** (aucun test apparié exécuté) ;
- `xgb_alone.spatial = 0,588` et `gnn_alone.spatial = 0,605` sont des **stubs codés en
  dur** (`run_hybrid_t1.py:92,98`), jamais rejoués sur les 8 mêmes plis ;
- `reality_rule.hybrid_gain_over_xgb_wall = 0,058` compare une moyenne 8-plis mesurée
  (0,646) à un scalaire stub (0,588) — pas une ablation appariée ;
- de plus, le mur 0,588 utilise un jeu de features PLUS LARGE (`cocontam="all"`+air) que
  le bloc tabulaire de l'hybride (`core`, 61) → ablation non propre de l'embedding.

➡️ **Déjà identifié et corrigé** par l'équipe : `gate5_paired_comparison.py` reconstruit
le bras **XGB-core sur les mêmes 8 plis** et lance Nadeau-Bengio + Wilcoxon. Résultat
honnête au REPORT §3 : hybride vs XGB-core **Δ=+0,050, Wilcoxon p=0,039 (sig.) MAIS
Nadeau-Bengio p=0,076 (n.s.)** → **BORDERLINE / sous-puissant** ; hybride vs GNN-seul
**+0,031 n.s.** Verdict REPORT « gain NON ROBUSTE ». **Conforme à ma règle.**
*Action restante* : `gate5_paired_comparison.py` **imprime** mais n'écrit pas de JSON →
le défaut original reste dans le `metrics.json` committé. **Corriger** : soit régénérer
`three_way_comparison` avec les vrais bras appariés, soit y inscrire un champ
`SUPERSEDED_BY: gate5_paired_comparison` pointant vers les chiffres du REPORT §3.

### 4b. `baseline_t2/metrics.json` est SMOKE, le REPORT cite des chiffres complets — **FAIL traçabilité**
- `metrics.json:2` → **`"smoke": true`** ; valeurs dégénérées (macro_AUROC 0,437,
  std 0,0, un seul pli). `metrics_incremental.json` aussi smoke (0,348). `full_run2.log`
  **vide (0 ligne)**.
- Le REPORT cite **BinaryRelevance spatial macro-AUROC 0,680**, Chain 0,667, Δ +0,222,
  Wilcoxon p=0,078 — introuvables dans tout `*_metrics.json` committé.
- Les seuls chiffres complets vivent dans `full_run.log`, et ils **ne concordent pas**
  avec le REPORT : log = **BR sp 0,698 / Chain 0,681** vs REPORT = **BR 0,680 / Chain
  0,667**. (La conclusion qualitative — chaînage n'aide pas, spatial ~0,68 vs aléatoire
  ~0,90, Δ~0,22 — reste robuste ; ce sont les chiffres exacts qui ne se tracent pas.)
- ➡️ **Corriger** : republier le `metrics.json` du run canonique T2 complet (8 plis,
  smoke=false) et réaligner le tableau du REPORT dessus ; sinon le « 0,680 » n'est pas
  reproductible et diverge de la seule trace complète existante (0,698).

### 4c. `hgt_rgcn_t1/metrics.json` est SMOKE mais encadré comme comparable au mur — **À CORRIGER**
- `meta.smoke = true`, `n_blocks = 3` (vs 8 pour les runs complets). Le REPORT présente
  pourtant les chiffres (spatial OOF 0,646, Δ random−spatial **0,016**) dans un tableau
  « comparable to the non-graph wall and gnn_phase1 » (REPORT ligne 10), et qualifie le
  Δ=0,016 de « the spatial-leakage inflation » (ligne 17).
- **Problème** : un Δ issu de **3 blocs sur sous-échantillon** n'est PAS comparable au
  Δ≈0,19-0,30 des runs complets 8 blocs ; le présenter sur le même pied induit en erreur.
  Le mécanisme anti-fuite (C4, `n_cross_block: 0`) est lui correct ; c'est le **statut
  smoke** qui doit être rendu impossible à manquer et le cadrage « vs mur » retiré tant
  qu'un run complet 8 blocs n'existe pas.
- *Contre-exemple de bonne pratique dans le même projet* : `gnn_phase3/REPORT.md` titre
  explicitement « RÉSULTAT NÉGATIF / inconclusif », divulgue le run interrompu/sous-
  entraîné et ne sur-revendique pas. hgt_rgcn doit s'aligner sur ce standard.

---

## Liste de corrections priorisée

**P1 — bloquantes pour publication de chiffres**
1. **baseline_t2** : republier le `metrics.json` canonique (run complet 8 plis,
   smoke=false) et réaligner le tableau du REPORT ; résoudre l'écart 0,680 (REPORT) vs
   0,698 (full_run.log). [§4b]
2. **hgt_rgcn_t1** : marquer SMOKE de façon non-ambiguë en tête de REPORT, retirer le
   cadrage « comparable au mur » et la lecture du Δ=0,016 comme inflation spatiale tant
   qu'un run complet 8 blocs n'a pas tourné. [§4c]
3. **gnn_hybrid_t1/metrics.json** : régénérer `three_way_comparison` avec les bras
   appariés réels (XGB-core, GNN-seul, 8 plis alignés) OU marquer le bloc
   `SUPERSEDED_BY` vers `gate5_paired_comparison`, et persister les sorties du gate5 en
   JSON (actuellement seulement imprimées). [§4a]

**P2 — exactitude rédactionnelle (non bloquant)**
4. **profilage/REPORT.md** : corriger « les 16 plus corrélées sont toutes `*_detected` »
   → « les 10 plus corrélées sont des `*_detected` ; well_depth_ft (#11) et geotracker
   suivent, contextes non-fuitants ». [§1]
5. **gate5** : figer la version sklearn (ou re-mapper les blocs par centroïde plutôt que
   par id) pour garantir l'alignement des plis à la reproduction. [§2]

**Aucune action requise** : `src/targets.py`, `src/features.py`, `src/config.py`,
`src/splits.py`, `src/graph.py`, `src/gnn.py`, `src/metrics.py` — anti-fuite cible et
spatiale corrects ; baseline_t1 (run complet) et gnn_phase1/phase2 exemplaires.

---

## Note finale
La contribution méthodologique centrale — **quantifier l'inflation spatiale** — est
correctement instrumentée et, sur les runs complets, **honnête et soignée** (Δ par
métrique, triplet aléatoire/spatial/Δ, tests appariés corrigés, IC bootstrap par
groupe, calibration). Les défauts trouvés sont des **problèmes de traçabilité/cadrage de
runs smoke**, pas des fuites. Aucune fuite de cible ni fuite spatiale réelle dans le code
exécuté. Verdict : **VALIDÉ SOUS CONDITIONS** (lever P1 avant tout chiffre publié).
