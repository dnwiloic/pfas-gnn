# REPORT — Baseline multilabel T2 (quels PFAS dépassent, PFAS CA, mode prédictif strict)

> Agent : `multilabel-specialist` + run canonique Colab. Graine 42.
> Cibles = `src.targets.build_T2` (schéma hybride EPA‑MCL / analytique + garde‑fou
> détection C1) ; features = socle (encodage fréquentiel, aucune mesure PFAS en entrée) ;
> CV = socle spatial‑block (référence) + aléatoire groupé (Δ) ; **masquage de mesure par
> label** (MNAR). 5 modèles dont **FrequencyClassChain** (4 classes de fréquence + chaîne
> cascade façon Dong et al. 2024). Les 5 métriques imposées (AUC‑ROC, F1, accuracy,
> rappel, précision) calculées en micro/macro/par‑label.
>
> **Statut : RUN CANONIQUE COMPLET (Colab) — données complètes 46 338 lignes, 10 labels,
> 8 plis spatiaux + 8 aléatoires.** Source : `experiments/baseline_t2/metrics.json`.

## 0. Résumé exécutif

- **Binary Relevance reste la référence** (macro‑AUROC spatial 0.680) ; aucun modèle ne la
  bat significativement. L'Ensemble de chaînes est à égalité (0.677) ; **FrequencyClassChain
  (chaîne‑par‑classe, Dong) = 0.668**, au niveau des autres chaînes — **le chaînage n'apporte
  pas de gain** en mode prédictif strict.
- **L'inflation spatiale est le fait central** : macro‑AUROC **aléatoire ~0.90 vs spatial
  ~0.67**, soit **Δ ≈ +0.22 à +0.23**. Aucun modèle ne réduit ce Δ.
- Positionnement Dong et al. 2024 : nos scores **aléatoires (~0.90) rejoignent la
  littérature** (cf. point externe **macro‑AUC 0.966** en split aléatoire, §1ter) ; nos
  scores **spatiaux (~0.67)** mesurent la généralisation géographique réelle (jamais
  rapportée en protocole non spatial). L'écart **0.966 → 0.68** est l'inflation
  spatiale + censure + design. **Le chaînage n'aide pas, confirmé aussi côté externe** (§1ter).

## 1. Modèles × les 5 métriques (micro, CV spatiale) + AUROC aléatoire + Δ

AUROC = macro (sans seuil) ; F1/accuracy/rappel/précision = micro au seuil OOF par label.

| modèle | AUROC | F1 | accuracy | rappel | précision | AUROC(rd) | Δ AUROC |
|---|---|---|---|---|---|---|---|
| Prévalence (plancher) | 0.348 | 0.487 | 0.398 | 0.990 | 0.323 | 0.476 | +0.128 |
| **BinaryRelevance** | **0.680** | 0.542 | 0.655 | 0.709 | 0.439 | 0.902 | +0.222 |
| Chaîne (1 ordre) | 0.667 | 0.533 | 0.642 | 0.707 | 0.427 | 0.901 | +0.234 |
| Ensemble de chaînes | 0.677 | 0.541 | 0.638 | 0.739 | 0.427 | 0.904 | +0.228 |
| FreqClassChain (Dong) | 0.668 | 0.531 | 0.649 | 0.689 | 0.432 | 0.895 | +0.228 |

**Test apparié chaîne − BR (8 plis spatiaux)** : macro‑AUROC +0.020 (Wilcoxon p=0.078),
micro‑F1 −0.013 (p=0.95) → **non significatif**. ⇒ **BR de référence.**

## 1ter. Point externe — approche « class » de Dong et al. (split aléatoire)

Run T2 d'un notebook antérieur (`ca-pfas-ml/.../06_multilabel_class_improvement_fullrun`,
2026-06-21), approche **`class` = 4 chaînes indépendantes par palier de couverture**
(= la stratégie de Dong et al.), **split aléatoire 80/20**, **27 labels PFAS**, 97 features,
`drop_location=True`. ⚠️ **Régime aléatoire/optimiste**, à NE PAS comparer à notre spatial.

| approche | macro‑AUC | micro‑F1 | Hamming | EMR |
|---|---|---|---|---|
| global | 0.9589 | — | — | — |
| nested | 0.9604 | — | — | — |
| **class (baseline Dong)** | **0.9661** | 0.856 | 0.154 | 0.187 |
| + ordre sous‑famille chimique | 0.9658 | 0.852 | — | — |
| + hyperparams XGBoost renforcés | 0.9670 | 0.862 | — | 0.215 |
| **combo (meilleur)** | **0.9671** | 0.861 | 0.154 | 0.211 |

**Deux enseignements majeurs, cohérents avec notre travail :**
1. **Le chaînage/ordonnancement n'apporte quasi rien** : l'ordre chimique de Dong
   (PFCA→FTS→PFSA→sulfonamides) donne **0.9658 < 0.9661** (aucun gain) ; le combo gagne
   **+0.001** macro‑AUC. ⇒ **confirme indépendamment** notre conclusion (chaînes ≈ Binary
   Relevance ; FreqClassChain n'aide pas).
2. **Ce 0.966 est le point « littérature/Dong » de T2**, gonflé par : split **aléatoire**
   (pas de groupage puits ni de blocs spatiaux), **27 labels** dont des ultra‑rares
   (macro‑AUC moyennée), **pas de garde‑fou de détection** (labels `*_ngL`>seuil bruts,
   censure incluse — cf. notre §2 sur FTS/FOSAA). **Notre référence spatiale reste ~0.68.**

## 2. Par‑label — Binary Relevance, CV spatiale (les 5 métriques au seuil OOF)

| label | mesuré | prévalence | AUROC | F1 | accuracy | rappel | précision | AP |
|---|---|---|---|---|---|---|---|---|
| PFOS | 46 256 | 0.394 | 0.588 | 0.536 | 0.562 | 0.643 | 0.460 | 0.450 |
| PFBS | 44 083 | 0.392 | 0.632 | 0.570 | 0.526 | 0.800 | 0.442 | 0.487 |
| PFHxA | 44 400 | 0.384 | 0.656 | 0.574 | 0.545 | 0.799 | 0.448 | 0.519 |
| PFOA | 46 252 | 0.341 | 0.665 | 0.573 | 0.617 | 0.753 | 0.462 | 0.443 |
| PFHpA | 45 099 | 0.265 | 0.634 | 0.439 | 0.579 | 0.620 | 0.339 | 0.379 |
| PFBA | 25 816 | 0.410 | 0.728 | 0.665 | 0.673 | 0.792 | 0.573 | 0.592 |
| PFPeA | 25 738 | 0.410 | 0.689 | 0.625 | 0.638 | 0.737 | 0.542 | 0.559 |
| PFHxS | 44 115 | 0.154 | 0.660 | 0.363 | 0.739 | 0.484 | 0.290 | 0.288 |
| PFPeS | 25 948 | 0.158 | 0.721 | 0.412 | 0.755 | 0.544 | 0.332 | 0.367 |
| PFNA | 45 112 | 0.026 | 0.831 | 0.241 | 0.956 | 0.274 | 0.215 | 0.169 |

PFNA : AUROC élevé (0.831) mais **AP 0.169 et F1 0.24** — label rare (2.6 %), viser l'AP.
PFBA/PFPeS/PFOA sont les mieux prédits spatialement (~0.66–0.73).

## 3. FrequencyClassChain — 4 classes de fréquence (Dong et al. 2024 sur nos cibles)

Labels rangés du moins rare au plus rare, chaîne cascade dans cet ordre. Résultat : sur
données complètes, FreqClassChain (0.668) ≈ Chaîne simple (0.667) ≈ sous BR (0.680) →
**le découpage en classes de fréquence n'aide pas** : en mode prédictif strict la
co‑occurrence est déjà médiée par le contexte commun (proxys géo/hydro), donc propager les
prédictions n'ajoute pas d'information.

## 4. Déséquilibre / labels rares & semi‑supervision

- **SMOTE sur PFNA** : AUROC 0.831 (class_weight) → **0.873 (+SMOTE)** = **+0.042**. Ici
  SMOTE aide PFNA (run complet) ; à confirmer en AP (la métrique pertinente à 2.6 %).
- **Pseudo‑étiquetage** (labels à panel réduit) : PFBA −0.017, PFPeA +0.006, PFPeS −0.011
  → **apport nul**, cohérent avec le MNAR. Non activé.

## 5. Mur pour les GNN

- **Cible à battre = macro‑AUROC spatial ≈ 0.68** (BR), pas l'aléatoire ~0.90.
- Rapporter le triplet (aléatoire, spatial, Δ) ; un GNN n'aide que s'il **monte le spatial
  sans gonfler Δ ≈ 0.22**. Std inter‑plis ≈ 0.08–0.10 → un gain < ~0.03 sera dans le bruit.
- Vu la structure MNAR, prioriser la **complétion de matrice bipartite puits × analyte**
  plutôt que le graphe de labels (les chaînes échouent ici), en respectant C4/C5.

## 6. Artefacts

- `src/baselines_t2.py` — modèles (BR, chaînes, ECC, **FreqClassChain**, prévalence),
  masques par label, seuils OOF, métriques masquées, SMOTE/pseudo, test apparié.
- `experiments/baseline_t2/run_baseline_t2.py` — driver (toggle `SMOKE_TEST`).
- `experiments/baseline_t2/{metrics.json,full_run.log,config.yaml}` — **run canonique
  Colab** (source des chiffres §1‑§4) ; `metrics_incremental.json` — checkpoint par modèle.
