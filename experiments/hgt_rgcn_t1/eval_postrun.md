# Audit post-run — HGT / R-GCN multi-relationnel T1

**Méthodologiste évaluation — verdict avec autorité de blocage.**
Date : 2026-06-22 · Seed run : 42 · Tâche : T1a (dépassement réglementaire binaire)

---

## VERDICT GLOBAL : **NON VALIDÉ — REFUSÉ (artefacts manquants / incohérents)**

L'audit demandé porte sur un run GPU Colab « 8 plis spatiaux, deux modèles HGT et
R-GCN » avec des fichiers `metrics_hgt.json` et `metrics_rgcn.json`. **Ces fichiers
n'existent pas dans le dépôt, et aucun run correspondant aux chiffres du brief n'y est
présent.** Je ne valide ni n'interprète des résultats que je ne peux pas recouper avec
un artefact versionné. Un score qu'on ne peut pas rouvrir et vérifier ne vaut rien.

Ce qui existe réellement dans `experiments/hgt_rgcn_t1/metrics.json` :

| Champ | Brief (demande) | Artefact réel présent |
|---|---|---|
| `meta.smoke` | (run complet GPU) | **`true` — SMOKE TEST** |
| `n_blocks` (plis) | 8 | **3** |
| modèles | HGT **et** R-GCN | **R-GCN seul** (`meta.model="rgcn"`) |
| fichiers | `metrics_hgt.json`, `metrics_rgcn.json` | **absents** — un seul `metrics.json` |
| AUC OOF spatial | 0.6472 | 0.6464 (proche, mais…) |
| AUC mean±std spatial | 0.5990 ± 0.0585 (8 plis) | **0.5441 ± 0.0635 (3 plis)** |
| plis spatiaux | [0.506,0.621,0.532,0.675,0.577,0.653,0.659,0.568] | **[0.614, 0.558, 0.460]** |
| Δ(random−spatial) | +0.1742 (R-GCN) / +0.1638 (HGT) | **+0.0161** |
| Brier spatial | 0.310 | **0.245** |
| ECE spatial | 0.264 | **0.167** |
| courbe fiabilité bin[0,0.1] | n=15 096 | **n=375** |

Les deux jeux de chiffres sont **mutuellement incompatibles** et ne peuvent provenir du
même run. Le brief décrit vraisemblablement un run réalisé sur Colab dont les sorties
**n'ont jamais été persistées dans le dépôt** (cf. CLAUDE.md §4 : espace Colab éphémère,
persistance explicite obligatoire en fin de run). C'est précisément le risque que la
politique « zéro Drive + commit en fin de run » est censée prévenir, et il s'est
matérialisé.

> Tant que les `metrics_*.json` du run 8-plis ne sont pas committés, **toute conclusion
> sur HGT vs R-GCN à AUC 0.644/0.647 est non auditable et donc non recevable.**

Le reste de ce rapport (1) documente ce que dit l'artefact réellement présent (run
smoke 3-plis R-GCN), et (2) fournit le protocole d'audit conditionnel à appliquer dès
que les vrais fichiers seront versionnés.

---

## 1. Fuite spatiale

### 1.1 Garde anti-fuite par arête — OK sur l'artefact présent
Sur le run smoke réel, l'assertion fonctionne : `n_cross_block_total = 0` dans les deux
régimes, et l'audit par pli détaille, par relation, la coupe effective des arêtes
inter-blocs :

- Régime **random** : `n_removed_cross_block_near = 62`, `n_removed_cross_block_subbasin
  = 78` par pli, résiduel `n_cross_block_* = 0`. La coupe est réellement exécutée (et
  pas seulement asserté à 0 sur un graphe déjà trivialement séparé).
- Régime **spatial** : `n_removed_* = 0` car les blocs géographiques rendent déjà
  presque toutes les arêtes intra-bloc ; résiduel = 0.

Le mécanisme de coupe **par relation séparément** (les deux types d'arête `near` et
`same_subbasin_knn` sont traités indépendamment) est correct et c'est le bon design pour
un graphe multi-relationnel : une seule coupe globale aurait pu laisser fuir une
relation. **Validé sur le principe.**

### 1.2 Le Δ du brief (+0.17) n'est PAS reproduit par l'artefact
Le brief affirme Δ(random−spatial) = +0.174 (R-GCN), à comparer aux phases 1-2
(GraphSAGE +0.196, GCN +0.218, **réels et vérifiés** dans `gnn_phase1/metrics.json`).
Mais l'artefact présent donne **Δ = +0.016** seulement.

Deux lectures, et aucune n'est rassurante :

- **Si le run 8-plis (Δ +0.17) est le vrai** : alors le smoke 3-plis (Δ +0.016) est
  non représentatif, ce qui est attendu (3 blocs ≠ 8 blocs, échantillon minuscule) —
  mais alors **on auditerait des chiffres absents**.
- **Quoi qu'il arrive**, un Δ de +0.16 à +0.22 sur cette tâche est **la signature d'une
  forte autocorrélation spatiale** : le split aléatoire gonfle l'AUC de ~0.20 par fuite
  spatiale via les voisins. C'est cohérent entre phases 1-2 et le brief. **C'est l'apport
  méthodologique central du projet et il est solide** : la quantification de l'inflation
  spatiale (~+0.16 à +0.22 d'AUC) est cohérente, large, et reproductible across modèles.
  Le split aléatoire est ici un artefact optimiste, pas une mesure de généralisation.

**Conclusion section 1** : garde anti-fuite OK ; quantification de l'inflation spatiale
robuste en tant que phénomène (~+0.2 AUC), MAIS la valeur précise +0.17 du brief n'est
pas dans l'artefact et reste à committer.

---

## 2. Comparaisons statistiques

### 2.1 Tests appariés demandés : **IMPOSSIBLES sur les données présentes**
Le brief demande un Wilcoxon/t apparié « sur les 8 plis spatiaux ». L'artefact n'a que
**3 plis (smoke)**. Un test apparié exige les **mêmes blocs géographiques** des deux
côtés. Or :

- R-GCN smoke : `n_blocks = 3` ;
- baselines GraphSAGE/GCN : `n_blocks = 8` (`gnn_phase1`).

Les blocs ne coïncident pas → **aucun appariement valide possible**. Faire un Wilcoxon
sur des vecteurs de longueurs différentes, ou tronquer, serait une faute. Je refuse de
le produire.

### 2.2 Ce que j'ai pu calculer honnêtement (code exécuté sur les vrais JSON)
Per-fold spatial réellement stockés :

```
R-GCN  (smoke, n=3) : [0.614, 0.558, 0.460]  mean 0.5441  std 0.0778
GraphSAGE  (n=8)    : [0.518,0.693,0.583,0.563,0.554,0.699,0.654,0.682]  mean 0.6184  std 0.0717
GCN        (n=8)    : [0.541,0.754,0.509,0.613,0.600,0.619,0.665,0.693]  mean 0.6243  std 0.0795
```

Comparaison **non appariée** (Welch, descriptive seulement, puissance quasi nulle à n=3) :

| Comparaison | Δ mean (R-GCN − base) | Welch t | p |
|---|---:|---:|---:|
| R-GCN(smoke) vs GraphSAGE | **−0.0743** | −1.44 | 0.235 |
| R-GCN(smoke) vs GCN | **−0.0802** | −1.51 | 0.210 |

**Lecture brutale** : sur l'artefact réellement présent, le R-GCN multi-relationnel est
**en-dessous** des baselines mono-relationnelles (0.544 vs 0.618/0.624), pas au-dessus.
C'est l'inverse exact du récit du brief (0.647 > 0.618). Aucune significativité (n=3),
mais la direction du signe est l'opposé de ce qui est revendiqué.

### 2.3 Sur les gains revendiqués au brief (+0.02–0.03)
À supposer même les chiffres 8-plis du brief exacts : gain R-GCN +0.023 et HGT +0.020
vs GraphSAGE, pour σ inter-plis spatial ≈ 0.06–0.07. **Le gain est ~3× plus petit que
l'écart-type inter-plis.** Règle C-CMP : tout gain < σ est dans le bruit. Donc même dans
le scénario le plus favorable au brief, **ces gains ne sont pas distinguables de zéro**.
Un test apparié sur 8 plis avec un effet de +0.02 et σ≈0.065 n'atteindra jamais p<0.05
(d de Cohen ≈ 0.3, puissance dérisoire à n=8).

**Conclusion section 2** : tests appariés non réalisables faute d'artefact ; sur les
données présentes, R-GCN ne bat pas les baselines ; et même au mieux du brief, les gains
sont sous le bruit inter-plis → non significatifs.

---

## 3. Calibration

### 3.1 Sur l'artefact présent (smoke 3-plis)
Brier spatial 0.245, ECE 0.167. Médiocre mais pas catastrophique. La courbe de fiabilité
montre déjà le bon motif d'alerte : bin [0,0.1] → n=375, conf=0.065, **frac_pos=0.261**
(26 % de positifs étiquetés <10 % de proba → sous-estimation des bas scores), et bins
hauts erratiques (peu de masse, n=1 à 0.9-1.0). Un seul seuil OOF est utilisé
(`threshold_used` varie par pli : 0.05 à 0.55) — l'optimisation de seuil est bien
out-of-fold, **conforme C-CAL/C-SEUIL**.

### 3.2 Sur les chiffres du brief (à confirmer)
Si ECE spatial = 0.264 (R-GCN) / 0.302 (HGT), c'est **sévère** : un ECE > 0.25 signifie
qu'en moyenne la probabilité annoncée est fausse de plus de 25 points. Le détail du brief
est sans appel :

- bin [0,0.1] : conf≈0.01–0.025 mais frac_pos≈0.30–0.32 → **30 % des « ~0 % » sont en
  fait positifs**. Sous-estimation massive des risques faibles. En décision, ces puits
  seraient classés sûrs à tort.
- bin [0.9,1] : frac_pos≈0.59–0.61 alors que conf>0.95 → **surconfiance** ; le modèle
  « certain » se trompe 4 fois sur 10.

Le contraste random (ECE 0.131 / 0.061) vs spatial (0.264 / 0.302) est lui-même
diagnostique : **la calibration s'effondre hors-distribution spatiale.** Un modèle calibré
sur des voisins fuités (random) se décalibre dès qu'on le confronte à des blocs
géographiques nouveaux. C'est cohérent avec le Δ AUC de fuite spatiale.

### 3.3 Recommandation
Une **bonne AUC mal calibrée est inexploitable en décision** — et ici l'AUC n'est même
pas bonne (0.64–0.65, à peine au-dessus du hasard sur certains plis). **Recommandation :**

1. Une recalibration post-hoc (Platt/isotonique) ajustée **uniquement sur les
   probabilités OOF, jamais sur le test**, est **nécessaire avant tout usage décisionnel**,
   et doit être réalisée **par pli spatial** (un calibrateur global réintroduirait de la
   fuite spatiale).
2. Mais la recalibration **ne corrigera pas le pouvoir discriminant** : elle aligne les
   probabilités, pas l'AUC. Avec une AUC spatiale ~0.64 et des plis à ~0.5, recalibrer
   ne rend pas le modèle décisionnellement utile ; cela rend juste ses probabilités
   honnêtes (et donc honnêtement médiocres).
3. **Reporter systématiquement Brier + ECE + courbe de fiabilité spatiale** (déjà fait
   dans le JSON — bonne pratique conservée).

---

## 4. Instabilité inter-plis

### 4.1 Constat
Artefact présent : plis R-GCN [0.614, 0.558, **0.460**] — un pli **sous le hasard**.
Brief : R-GCN pli 0 = 0.506 (hasard), HGT pli 0 = **0.463** (sous le hasard) ; étendue
~0.46 à ~0.69 ; σ ≈ 0.059–0.068.

### 4.2 Interprétation
Un AUC < 0.5 sur un bloc géographique **n'est pas du bruit neutre** : cela signifie que
la relation features→cible apprise ailleurs est **inversée** dans ce bloc. C'est la
signature d'une **non-stationnarité spatiale** du processus PFAS : les déterminants d'un
dépassement diffèrent d'une région à l'autre (hydrogéologie, sources industrielles
locales, pratiques agricoles). Le modèle global apprend une moyenne qui ne transfère pas
— voire s'inverse — sur certaines régions.

σ ≈ 0.06–0.07 sur 3 à 8 plis, avec une étendue de ~0.23 d'AUC, est **élevé** et
**inacceptable pour une revendication de généralisabilité géographique**. Concrètement :
déployer ce modèle sur une nouvelle région revient à tirer un AUC quelque part entre 0.46
et 0.69 — c'est-à-dire entre « pire que pile ou face » et « médiocre ». Aucune garantie
opérationnelle.

Ce n'est pas un défaut du modèle à corriger par tuning : c'est une **propriété des
données** (autocorrélation + non-stationnarité) que la CV spatiale révèle correctement.
Le mérite du protocole est précisément de l'exposer ; le split aléatoire la masquait.

---

## 5. Verdict sur l'apport des GNN multi-relationnels

**Aucun apport démontré. Résultat négatif à documenter comme tel.**

1. **Sur l'artefact réellement présent** : R-GCN spatial 0.544 < GraphSAGE 0.618 < GCN
   0.624. Les relations multiples **dégradent** (smoke, non concluant mais défavorable).
2. **Sur les chiffres du brief (au mieux)** : HGT 0.644 / R-GCN 0.647 vs GraphSAGE 0.618
   / GCN 0.624 → gain +0.02–0.03, soit **~⅓ de σ inter-plis (0.06–0.07)**. Sous le bruit.
   Non significatif par construction (puissance nulle à 8 plis pour un effet de cette
   taille).
3. **Mur tabulaire RF ~0.600** : ni les baselines GNN ni les multi-relationnels ne
   s'en détachent franchement. Le graphe (mono ou multi-relationnel) **n'achète quasi
   rien** au-dessus d'un RF sur features de contexte.

Toute la famille — GraphSAGE, GCN, R-GCN, HGT — converge vers **AUC spatiale ≈ 0.60–0.65
avec σ ≈ 0.06–0.07 et des plis sous le hasard**. La sophistication relationnelle
(HGT/R-GCN) n'élargit pas cette enveloppe. **L'hypothèse « les relations
hydrogéologiques multiples portent un signal exploitable par GNN » n'est pas soutenue.**

C'est un **résultat négatif propre et précieux** : il borne ce que les GNN apportent sur
cette tâche (rien de plus que le mur tabulaire, une fois la fuite spatiale neutralisée)
et il chiffre l'inflation spatiale (~+0.16 à +0.22 AUC) qui aurait fait croire au succès
sous split aléatoire. **C'est l'apport méthodologique du projet ; il faut le revendiquer
comme tel, pas chercher un gain de +0.02 qui n'existe pas.**

---

## 6. Recommandations pour la suite

**Bloquant (à faire avant tout nouvel audit « validable ») :**

1. **Committer les artefacts du run 8-plis** (`metrics_rgcn.json`, `metrics_hgt.json`,
   ou un `metrics.json` consolidé avec `smoke=false`, `n_blocks=8`, `per_fold` complet
   par modèle ET par régime, les `audit.*` par pli, les `reliability_curve`). Sans cela,
   les chiffres du brief restent non auditables → l'audit demandé **ne peut pas aboutir
   à VALIDÉ**. Vérifier que le notebook Colab persiste bien ses sorties (commit/download)
   en fin de run — la perte actuelle illustre la défaillance de cette étape (CLAUDE.md §4).
2. **Aligner les plis** : R-GCN/HGT et baselines GraphSAGE/GCN doivent partager
   **exactement les mêmes 8 blocs spatiaux** (mêmes IDs, même graine) pour autoriser des
   tests appariés. Stocker l'assignation bloc→puits.

**Une fois les artefacts présents — protocole d'audit conditionnel (déjà cadré) :**

3. **Tests appariés** sur 8 plis identiques : Wilcoxon signé (robuste, n=8) pour
   {HGT vs R-GCN, R-GCN vs GraphSAGE, HGT vs GraphSAGE, chacun vs mur RF}. Reporter Δ,
   IC95 bootstrap apparié, p. **Anticiper** : effets +0.02 / σ 0.065 → non significatifs ;
   le verdict « pas de gain » tiendra très probablement.
4. **Recalibration post-hoc OOF par pli spatial** (isotonique ou Platt) avant tout
   chiffre décisionnel ; re-reporter Brier/ECE post-calibration. Ne jamais calibrer sur
   le test.
5. **Investiguer le bloc à AUC<0.5** : caractériser ce bloc géographique (taille,
   prévalence, hydrogéologie, présence de sources) — la non-stationnarité y est la vraie
   histoire scientifique, plus intéressante que +0.02 d'AUC.

**Orientation stratégique :**

6. **Arrêter l'escalade architecturale GNN sur T1.** GraphSAGE → GCN → R-GCN → HGT n'a
   pas bougé l'enveloppe 0.60–0.65. Documenter en **résultat négatif** et rediriger
   l'effort vers : (a) l'enrichissement des features (le mur RF ~0.60 suggère que le
   plafond est dans le signal des données, pas dans l'architecture) ; (b) des modèles
   **spatialement adaptatifs** (effets régionaux, geographically-weighted) qui adressent
   la non-stationnarité plutôt que de l'ignorer ; (c) une honnête mise en avant de la
   **quantification de l'inflation spatiale** comme contribution principale.

---

### Résumé exécutif
- **NON VALIDÉ / REFUSÉ** : les artefacts du run audité (8 plis, HGT+R-GCN) sont absents
  du dépôt ; le seul fichier présent est un smoke R-GCN 3-plis incompatible avec le brief.
- Garde anti-fuite par arête : **OK** (coupe par relation, résiduel 0, vérifié).
- Inflation spatiale : phénomène **robuste** (~+0.16 à +0.22 AUC), apport clé du projet —
  mais la valeur +0.17 du brief n'est pas committée.
- Apport des GNN multi-relationnels : **nul** (gains +0.02–0.03 < σ inter-plis 0.06–0.07,
  non significatifs ; et défavorable sur l'artefact réel).
- Calibration spatiale (brief) : **sévère** (ECE 0.26–0.30, sous-estimation des bas
  scores + surconfiance) → recalibration OOF par pli **obligatoire** avant décision, sans
  espoir de gagner en discrimination.
- Action n°1 : **committer les vrais JSON 8-plis** ; sans cela, pas d'audit validable.
