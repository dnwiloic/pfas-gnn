# Validation de protocole — V1 (GraphSAGE-hétéro inductif vs HGT), T1a

Méthodologiste : eval-methodologist. Boucle C (NO-GO) puis BOUCLE C — RE-CONFIRMATION
(audit du patch §3.8). Date révision : 2026-06-23. Mode : AUDIT-ONLY (lecture + contrôles
déterministes, AUCUN ré-entraînement). Smoke re-lancé par le fil principal après patch.

Code audité : `src/v1_inductive_sage_t1.py` (runner), `src/gnn_hetero_t1.py`
(encodeurs + `train_eval_fold`), socle `src/{graph,splits,metrics}.py`.
Socle réutilisé sans modification du cœur anti-fuite : `experiments/hgt_fusion_stacking_t1/
eval_validation.md` (VALIDÉ 6/6).

---

## VERDICT FINAL : **GO**

Le seul point bloquant de la boucle précédente (§3.8 — aucune courbe d'entraînement, risque
de sous-apprentissage masqué par l'early-stop) est **levé**. Le patch spécifié est en place,
**purement additif**, et ne touche aucun chemin anti-fuite. Les cinq axes anti-fuite restent
**PASS** (non ré-audités, conformément au mandat ; inchangés par le patch — voir Point
Pureté). Le protocole peut partir en run long Colab.

Re-confirmation point par point du correctif §3.8 ci-dessous (§ « LEVÉE DU BLOCAGE »).

---

## Points 1–5 (anti-fuite) — PASS, NON RÉ-AUDITÉS cette boucle

Validés à la boucle précédente et **non modifiés par le patch** (cf. Point Pureté) :

| Axe | Verdict | Preuve (boucle précédente) |
|---|---|---|
| 1. Embeddings/probas OOF (2 encodeurs) | PASS | v1:162-168 ; hetero:507-510,603 ; `_emb` jeté v1:164 |
| 2. CV spatiale k=8, groupage, coupe par relation | PASS | v1:138-143,169 ; hetero:139-148 ; config:146 |
| 3. Métrique = prédictif strict spatial ; mur in-run ; per-fold alignée | PASS | v1:309,383-388,398 ; mur in-run v1:177-183 |
| 4. Composants V1 sans fuite + diag overfit sans fuite | PASS | hetero:139-141,249,264,507-508,611-628 |
| 5. Seuil OOF jamais sur test | PASS | v1:204-206 ; hetero:631 |

(Détail de preuve conservé dans l'historique de ce fichier / boucle C initiale.)

---

## LEVÉE DU BLOCAGE — Courbes d'entraînement (CLAUDE.md §3.8) : **PASS**

### 1. Log par époque — PASS
`train_eval_fold` initialise `history = []` (`gnn_hetero_t1.py:563`) et **accumule par
époque** `(epoch, train_loss, val_auc, fit_auc)` :
`history.append((int(epoch), float(loss.detach()), float(vauc), float(fauc)))` (`:589`),
à l'intérieur de la boucle `for epoch in range(max_epochs)`. `fauc` est un AJOUT propre :
`roc_auc_score(yf, pf)` avec `pf = p[m_fit]`, `p` issu de `train_pack` (`:580`, inchangé) —
arêtes train-side, donc fit-AUC sans contamination test. Les clés sont exposées dans
`train_diag` (`:633-638`) : `history_epochs / history_train_loss / history_val_auc /
history_fit_auc`, plus `n_epochs_ran`, `max_epochs`, `early_stopped`. `train_diag` remonte
via `FoldResult.train_diag` (`:411` champ ajouté avec défaut ; `:659` passé au constructeur).
Le runner copie chaque `fr.train_diag` dans `fold_diag` par pli (`v1:170-173`), propagé en
`reg["fold_diag"]` (`v1:195,350`). Chaîne de données complète et inductive-propre.

### 2. Courbes PNG par encodeur, ligne au best_epoch — PASS
`_plot_training_curves(enc_results, exp_dir)` (`v1:458-490`) : pour CHAQUE encodeur de
`enc_results` (donc `hgt` ET `hetero_sage_v1`), trace val-AUC (trait plein) + fit-AUC
(pointillé) vs époque sur `ax1`, train-loss sur `ax2`, et une `axvline(best_epoch)` par pli
(`:481`). Écrit `training_curves_<enc>.png` (`:488`). Appelé inconditionnellement dans le
bloc `if write:` (`v1:421`), après `_write_metrics`, avant `_write_report`. Garde headless
gracieuse si matplotlib absent (`:464-466`) — n'invalide pas le run, conforme.

### 3. Diagnostic écrit de convergence / sous-apprentissage — PASS
`_convergence_diag(fold_diag)` (`v1:428-455`) calcule par encodeur :
`n_epochs_ran_mean/min`, `frac_folds_early_stopped`, `frac_folds_best_in_last_20pct`,
`under_training_flag`. Stocké dans `comparison.convergence` (`v1:413-415`) et **rendu** dans
la section « Convergence / under-training diagnostic (§3.8) » du REPORT (`_write_report`,
`v1:589-605`) sous forme de tableau pour les DEUX encodeurs.

**Logique du flag (vérifiée).** `still_rising` s'incrémente si `best_epoch >= 0.8 *
n_epochs_ran` (`v1:444`), c.-à-d. le meilleur epoch de validation tombe dans les **20 %
finaux** du run ; `under_training_flag = (still_rising / nfold >= 0.5)` (`:452`). C'est
exactement la signature de l'early-stop prématuré (piège P0) : si le pic val n'est atteint
qu'en toute fin avant déclenchement de la patience, la courbe val ne plafonnait pas → budget
d'époques/patience à augmenter. Le flag détecte donc bien le cas visé. Cohérence early-stop
↔ courbe : `early_stopped = (n_epochs_ran < max_epochs)` (`hetero:638`) recoupé avec
`frac_folds_best_in_last_20pct` dans le tableau du REPORT.

> Réserve mineure NON bloquante (à garder à l'œil sur le run réel, pas un correctif requis) :
> `under_training_flag = no` en smoke est ATTENDU (runs de quelques époques, trop courts pour
> que `be >= 0.8*nr` soit informatif). Sur le run complet (max_epochs 400 / patience 50), si
> le tableau affiche `under-training = YES` ou `best-in-last-20% ≥ 50 %`, le fil principal
> doit **relever max_epochs/patience et relancer** avant d'endosser les chiffres — c'est
> précisément ce que cette instrumentation rend désormais visible. L'instrumentation est le
> livrable exigé ; l'action corrective éventuelle dépend de ce que montrera le run réel.

---

## PURETÉ ADDITIVE DU PATCH — CONFIRMÉE

Le patch §3.8 n'ajoute QUE de l'instrumentation (collecte d'historique + figures +
diagnostic). Aucun chemin anti-fuite validé 6/8 n'est modifié :

- **Boucle d'entraînement** : la seule mutation est `history.append(...)` (`:589`) — lecture
  pure de `loss`/`vauc` déjà calculés + un `fauc` nouveau sur `m_fit` (train-side). `loss`,
  `opt.step`, sélection `best_state`/`best_epoch`, critère d'early-stop (`:592-598`) :
  INCHANGÉS.
- **`p = sigmoid(_fwd(model, train_pack))`** pour la validation (`:580`) : INCHANGÉ — la
  validation reste scorée sur arêtes train-side (inductif).
- **`_node_diag` / `train_diag`** (`:617-639`) : diagnostic en lecture seule, score
  `fit_nodes`/`val_nodes` via `p_train` = `train_pack` (`:615`). N'écrit ni `proba_node`, ni
  `thr`, ni `score_pack`, ni les masques, ni les coupes inter-blocs.
- **Seuil OOF** (`thr = _f1_threshold(y_well[val_nodes], proba_node[val_nodes])`, `:642`) et
  **scoring test** (`proba_node[test_row_mask]`, `:647-651`) : byte-identiques au V1.
- **Coupe inter-blocs par relation** (`hetero:139-148`), **inductif train-train**
  (`_build_edge_tensors(train_only_mask=...)`, `:507-508`), **DropEdge en aval de la coupe**
  (`:249,264`), **neighbor sampling depuis les arêtes train-train** (`:566`) : tous
  INCHANGÉS par le patch.
- **`FoldResult`** : seul ajout = champ `train_diag` (défaut vide, `:411`) ; le tuple
  retourné `(fr, proba_node, emb_node)` est inchangé.
- **Runner** : les ajouts (`_convergence_diag`, `_plot_training_curves`, section REPORT) sont
  des fonctions/sorties nouvelles appelées dans le bloc `if write:` ; aucune ligne
  `test_nodes / test_row / thr / val_nodes / cut_cross / gnn_proba` n'est modifiée
  (vérifié par grep ciblé sur le diff du runner : zéro occurrence touchée).

---

## Synthèse des verdicts par point (après patch)

| Axe | Verdict | Statut |
|---|---|---|
| 1. Embeddings/probas OOF (2 encodeurs) | PASS | inchangé |
| 2. CV spatiale k=8, groupage, coupe par relation | PASS | inchangé |
| 3. Métrique prédictif strict spatial ; mur in-run ; per-fold alignée | PASS | inchangé |
| 4. Composants V1 sans fuite + diag overfit sans fuite | PASS | inchangé |
| 5. Seuil OOF jamais sur test | PASS | inchangé |
| §3.8 Courbes d'entraînement (log/époque + PNG/encodeur + diagnostic écrit) | **PASS** | **levé par le patch** |
| Pureté additive (aucun chemin anti-fuite touché) | **PASS** | vérifié sur diff |

**VERDICT GLOBAL : GO.** Le blocage §3.8 est levé par un patch purement additif ; les six
contrôles anti-fuite restent valides. Conditions de suivi (non bloquantes) : sur le run
Colab complet, examiner les `training_curves_<enc>.png` et le tableau de convergence du
REPORT ; si `under_training_flag = YES`, relever max_epochs/patience et relancer avant
d'endosser les AUC finales.

---

## CLÔTURE BOUCLE C — RÉSULTATS (audit de traçabilité, lecture seule)

Méthodologiste : eval-methodologist. Date : 2026-06-24. Mode : AUDIT-ONLY (recoupement
déterministe REPORT.md ↔ metrics_summary.json + recalculs ; AUCUN ré-entraînement, aucune
relecture du metrics.json complet — non encore committé en local).

### VERDICT FINAL : **ENDOSSÉ — V1 REJETÉE (négatif honnête)**, sous condition de commit

Le verdict « V1 REJETÉE » du fil principal est **endossé**. Les chiffres headline sont
internement cohérents (REPORT ↔ summary ↔ recalculs per-fold), les comparaisons sont
valides (mur in-run, tests appariés per-fold, règle de réalité), et l'interprétation
méthodologique (goulot = extrapolation spatiale, pas sur-apprentissage de nœuds) est
soutenue par les chiffres.

**Condition suspensive (non bloquante sur le fond, bloquante sur la traçabilité finale) :**
l'endossement n'est définitif qu'après commit en local de :
- `metrics.json` complet (histories par époque, 16 plis) — artefact AUTORITAIRE ;
- `training_curves_hgt.png` et `training_curves_hetero_sage_v1.png` ;
- `config.yaml`.
Tant que ces fichiers ne sont pas versionnés, le résumé n'est pas indépendamment
reproductible : §3.8 (courbes examinées) et le recalcul OOF reposent sur un fichier non
présent. Verdict = **ENDOSSÉ-CONDITIONNEL**.

### Point 1 — TRAÇABILITÉ : **PASS**
Tous les chiffres headline du REPORT tracent au summary et se recalculent au rounding près :

| grandeur | recalcul | REPORT/summary |
|---|---|---|
| HGT pfm (8 plis) | 0,6178 | 0,6177 |
| SAGE pfm (8 plis) | 0,5820 | 0,5819 |
| SAGE−HGT pfm diff | −0,0358 | −0,0358 |
| SAGE−mur gain | −0,0609 | −0,0610 |
| HGT−mur gain | −0,0252 | −0,0252 |
| Δ(rd−sp) HGT (OOF−OOF) | 0,1525 | 0,1525 |
| Δ(rd−sp) SAGE (OOF−OOF) | 0,2036 | 0,2037 |
| réduction gap SAGE vs HGT | −0,0080 | −0,0080 |

n_plis = 8 = n_blocks (k=8) pour les deux encodeurs : cohérent.
**Réserve rédactionnelle mineure (non bloquante) :** le tableau du REPORT (§0) affiche
le Δ(rd−sp) sur la même ligne que la colonne spatiale « per-fold-mean / OOF ». Or le Δ
est calculé OOF−OOF (random n'est rapporté qu'en OOF-global), pas pfm−pfm. Cohérent et
correct, mais à expliciter pour éviter la lecture pfm−pfm (qui donnerait 0,1834 / 0,2494).
À clarifier d'une note de bas de tableau lors de la mise au propre — n'affecte pas le verdict.

### Point 2 — COMPARAISON VALIDE : **PASS**
- Le verdict s'appuie bien sur le **mur XGB IN-RUN** (pfm 0,6429 / OOF 0,6879), pas sur le
  mur committé 0,588 — comparaison sur les mêmes plis spatiaux, donc appariée et légitime.
- Tests appariés per-fold vs per-fold : SAGE−mur NB p=0,218 / Wilcoxon p=0,109 ; HGT−mur
  NB p=0,651 / Wilcoxon p=0,461 ; SAGE−HGT NB p=0,223 / Wilcoxon p=0,195. **Aucun p<0,05.**
- Règle de réalité (p<0,05 ET |Δ|>0,03 AUC) : aucune paire ne la franchit. Les deux verdicts
  `no_robust_gain` (vs mur) et `no_robust_diff` (SAGE vs HGT) sont **correctement justifiés**.
- Contrôle de bruit : l'écart SAGE−HGT (−0,036) est INFÉRIEUR à l'écart-type inter-plis du
  HGT (0,085) → gain sous le bruit inter-plis, conforme à la méfiance exigée (§5 du mandat).

### Point 3 — ENDOSSEMENT DU NÉGATIF : **PASS**
- (a) SAGE < HGT < mur : confirmé numériquement (0,582 < 0,618 < 0,643), tous écarts non
  significatifs → « ne bat ni le HGT ni le mur » est exact (négatif honnête, pas sur-vendu).
- (b) Réduction du gap fit−val = −0,008 (le SAGE a un gap PLUS GRAND, 0,057 ≥ 0,049) :
  l'objectif de V1 (réduire le sur-apprentissage) est réfuté, signe correct.
- (c) Inflation spatiale aggravée : Δ passe de 0,1525 (HGT) à 0,2037 (SAGE) : confirmé.
- **Interprétation méthodologique clé — VALIDÉE.** L'affirmation « l'écart fit−val (~0,05)
  sous-estime l'échec réel ; le vrai goulot est l'extrapolation spatiale » est soutenue par
  les chiffres : val_auc_mean ~0,914 (les deux) vs test spatial pfm ~0,60. L'effondrement
  **val→test = −0,296 (HGT) / −0,332 (SAGE)** écrase d'un facteur ~6 l'écart fit→val (~0,05).
  Le val étant des micro-blocs internes au train (proches), il est optimiste ; le test est un
  bloc KMeans tenu (région non vue). Conclusion correcte : une régularisation inductive qui
  cible fit−val ne PEUT PAS corriger un déficit d'extrapolation hors-région. C'est l'apport
  méthodologique majeur du projet (quantification de l'inflation spatiale) — soigné et juste.
  **Nuance à garder :** ce diagnostic est lui-même conditionné à la lecture des courbes du
  metrics.json complet (val_auc par époque, fit_auc par époque) — d'où la condition de commit.

### Point 4 — §3.8 (under-training) : **PASS**
`under_training_flag = false` pour les deux ; early_stopped_frac = 1,0. La logique du flag
(best_epoch dans les 20% finaux ⇒ rising) avait été auditée et validée (boucle précédente).
Headroom d'époques confortable (HGT 175/400, SAGE 114/400) : l'early-stop ne se déclenche pas
au plafond du budget, donc le pic val n'est pas tronqué. La réserve de suivi (relever
max_epochs/patience si under-training) est correctement **levée**. AUC **endossables** sur ce
critère — **sous réserve** d'inspection visuelle des deux PNG au commit (cohérence axvline
best_epoch ↔ plateau val), que je n'ai pu vérifier (PNG non committés).

### Synthèse de clôture
VALIDÉ sur les 4 points. Verdict V1 REJETÉE **endossé**. Endossement DÉFINITIF dès que
`metrics.json` (complet) + 2 PNG + `config.yaml` sont committés en local. Réserve
rédactionnelle (note OOF sur le Δ du tableau §0) à corriger à la mise au propre, sans impact
sur le verdict.
