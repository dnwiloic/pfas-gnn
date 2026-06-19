# REPORT — Baseline multilabel T2 (quels PFAS dépassent, PFAS CA, mode prédictif strict)

> Agent : `multilabel-specialist`. Graine 42. Cibles = `src.targets.build_T2` (schéma
> hybride EPA-MCL / analytique + garde-fou détection C1) ; features = socle (encodage
> fréquentiel, aucune mesure PFAS en entrée) ; CV = socle spatial-block (référence) +
> aléatoire groupé (Δ) ; **masquage de mesure par label** (MNAR : on n'entraîne/évalue
> un label que sur ses lignes mesurées).
>
> ⚠️ **PROVENANCE / DÉVIATION.** Le run complet a été exécuté **localement sur CPU
> (~4 h)** — déviation de la discipline CLAUDE.md (les runs longs vont sur Colab GPU).
> Les **chiffres « full » ci-dessous (headline) sont vérifiés** dans
> `experiments/baseline_t2/full_run.log`. Le tableau **par-label détaillé est SMOKE**
> (500 puits, k=3) car le run complet n'a persisté que le headline. **À reproduire
> proprement sur Colab pour figer le canonique** (per-label complet, calibration).
> Module : `src/baselines_t2.py` ; smoke : `tests/test_baselines_t2.py` (VERT ~66 s CPU).

## 0. Résumé exécutif

Binary Relevance (un classifieur par label), Chaînes de classifieurs (1 ordre),
Ensemble de chaînes (ECC) et plancher de prévalence, comparés sous double CV.
**Conclusion : les chaînes ne battent pas la Binary Relevance** (gain < bruit, non
significatif) ⇒ **BR = baseline T2 de référence**. L'inflation spatiale (Δ ≈ 0.20 sur
macro-AUROC) est le fait méthodologique central et **aucun modèle ne la réduit**.

## 1. Headline — modèles × métriques (RUN COMPLET, full_run.log)

Données complètes (46 338 lignes, 10 labels), 8 plis spatiaux + 8 aléatoires.

| modèle | mAUROC sp | microF1 sp | Hamming sp | EMR sp | mAUROC rd | Δ mAUROC (rd−sp) | coût |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Prévalence (plancher) | 0.348 | 0.487 | 0.602 | 0.074 | 0.471 | +0.123 | 16 s |
| **Binary Relevance** | **0.698** | **0.551** | **0.337** | 0.156 | 0.903 | +0.205 | 757 s |
| Chaîne (1 ordre) | 0.681 | 0.545 | 0.365 | 0.137 | 0.903 | +0.222 | 2 467 s |
| Ensemble de chaînes (ECC) | 0.701 | 0.561 | 0.345 | 0.137 | 0.907 | +0.206 | 10 584 s |

**Test apparié chaîne − BR (8 plis spatiaux)** : macro-AUROC −0.0055 (Wilcoxon p=0.55) ;
micro-F1 −0.026 (p=0.11). **Non significatif.** ECC ≈ BR (+0.003 < bruit, std inter-plis
≈ 0.087) pour **14×** le coût. ⇒ **BR de référence ; ECC réservé à Colab/multi-cœur.**

## 1bis. Modèle « chaîne par classe » (Dong et al. 2024) + 5 métriques (ajout 2026-06-19)

À la demande, ajout du modèle **`FrequencyClassChain`** (`src/baselines_t2.py`) reproduisant
l'architecture Dong et al. 2024 **sur nos cibles** (hybride + garde-fou C1, CV spatiale) :
les labels sont rangés en **4 classes ordonnées par fréquence de présence** (du moins rare
au plus rare) et une **chaîne en cascade** parcourt cet ordre, si bien que les PFAS rares
sont prédits à partir des plus fréquents déjà prédits. Réutilise la `MaskedClassifierChain`
(priors OOF sans fuite). Le découpage en 4 classes (ordre de fréquence, train) est rapporté
dans `metrics.json`/`classes_`.

Et conformément à la consigne, **les 5 métriques — AUC-ROC, F1, accuracy, rappel,
précision — sont désormais calculées pour T1 ET T2** (T2 en micro, macro et par-label ;
module partagé `src/metrics.py`), en plus de Hamming/EMR.

**Aperçu smoke (7 labels, 800 puits, k=2 — indicatif ; chiffres canoniques à régénérer sur Colab)** :

| modèle | AUROC | F1 | accuracy | recall | precision |
|---|---|---|---|---|---|
| Prévalence | 0.433 | 0.422 | 0.383 | 0.981 | 0.269 |
| BinaryRelevance | 0.683 | 0.509 | 0.694 | 0.691 | 0.403 |
| Chain | 0.685 | 0.506 | 0.675 | 0.725 | 0.389 |
| Ensemble | 0.681 | 0.509 | 0.683 | 0.714 | 0.395 |
| **FreqClassChain** | 0.677 | 0.509 | 0.669 | 0.746 | 0.386 |

Le `FrequencyClassChain` est **au niveau des autres chaînes** (≈ BR), cohérent avec la
conclusion : en mode prédictif strict, la co-occurrence est déjà médiée par le contexte ⇒
pas de gain net du chaînage. Smoke VERT en ~158 s CPU (`tests/test_baselines_t2.py`).

## 2. Par-label — Binary Relevance (⚠️ SMOKE, 500 puits ; à régénérer sur Colab)

| label | n_meas | prévalence | AUROC sp | AP sp |
|---|---|---|---|---|
| PFOS | 3153 | 0.340 | 0.616 | 0.406 |
| PFBS | 3013 | 0.331 | 0.622 | 0.402 |
| PFHxA | 3025 | 0.338 | 0.625 | 0.446 |
| PFOA | 3152 | 0.286 | 0.597 | 0.364 |
| PFHpA | 3077 | 0.201 | 0.602 | 0.301 |
| PFBA | 1788 | 0.352 | 0.681 | 0.501 |
| PFPeA | 1781 | 0.360 | 0.676 | 0.513 |
| PFHxS | 3017 | 0.127 | 0.628 | 0.264 |
| PFPeS | 1799 | 0.147 | 0.755 | 0.357 |
| PFNA | 3077 | 0.026 | 0.843 | 0.251 |

Repères **run complet** (rapportés par l'agent) : AUROC spatial minimal = **0.598 (PFOS)** ;
max hors PFNA = **0.729 (PFBA)** ; PFNA AUROC 0.908 mais **AP seulement 0.254** (à 2.6 %
de prévalence, viser l'AP, pas l'AUROC).

## 3. Où les chaînes aident-elles ? (per-label AUROC, chaîne − BR)

Quasiment nulle part, et non significativement. Micro-gains seulement sur **PFHxS (+0.018)**
et **PFPeS (+0.010)** (la paire à co-occurrence ~0.84). Paradoxe : les labels à **plus
forte co-occurrence** sont les plus **dégradés** par la chaîne (PFHxA −0.054, PFHpA −0.055,
PFOA −0.044) — en mode prédictif strict la co-occurrence est déjà médiée par le **contexte
commun** (proxys géo/hydro partagés) ; le prior d'un label imparfait ne fait que
**propager son erreur**, amplifiée en CV spatiale.

## 4. Déséquilibre / labels rares

- PFNA (rare réglementé, ~2.6 %) : `class_weight='balanced'`. **SMOTE testé et REJETÉ**
  (AUROC 0.886 vs 0.908 sans) — il crée des positifs hors distribution spatiale.
- Tous les labels appris utilisent `class_weight='balanced'` (mesuré, pas supposé).

## 5. Semi-supervision (sonde de pseudo-étiquetage, labels à panel réduit)

Apport **nul, signes mélangés** (PFBA −0.014, PFPeA +0.022, PFPeS −0.011 ; tous dans le
bruit) — cohérent avec le MNAR (transport de distribution source→cible). **Non activé** ;
code réutilisable en cas de transfert futur vers un contexte à données rares.

## 6. Points de vigilance & positionnement

- **Inflation spatiale = contribution méthodologique** : Δ ≈ 0.20 macro-AUROC (jusqu'à
  +0.295 PFOS, +0.275 PFHxS). Le classement se fait **en spatial**.
- **vs Dong et al. (2024)** : nos AUROC **spatiaux** 0.60–0.73 sont ~0.15–0.30 **sous** les
  ~0.8–0.9 publiés ; nos scores **random** (0.90) **rejoignent la littérature** → l'essentiel
  de l'écart est l'inflation spatiale (+ mode prédictif strict + garde-fou détection).
- **Puissance** : k=8 LOBO, std inter-plis ≈ 0.087 → un gain GNN < 0.03 AUROC spatiale
  sera dans le bruit.
- **Recommandation GNN forte** : vu la structure MNAR, prioriser la **complétion de
  matrice bipartite puits × analyte** (réutiliser le masque de mesure comme matrice
  d'observation), **pas** le graphe de labels (les chaînes échouent ici). Respecter
  C4/C5 (k-NN spatial plafonné ~1–2 km, arêtes coupées aux frontières de bloc).

## 7. Artefacts

- `src/baselines_t2.py` — module (masques par label, BR / chaîne / ECC / prévalence,
  seuils OOF par label, métriques masquées, sonde pseudo, test apparié).
- `tests/test_baselines_t2.py` — smoke test (VERT, ~66 s CPU).
- `experiments/baseline_t2/run_baseline_t2.py` — driver (toggle `SMOKE_TEST`).
- `experiments/baseline_t2/full_run.log` — **headline du run complet (source des chiffres §1)**.
- `experiments/baseline_t2/{config.yaml,metrics.json}` — ⚠️ état SMOKE (à régénérer canonique sur Colab).
