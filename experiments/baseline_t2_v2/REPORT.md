# REPORT — Mur multilabel T2 sur le dataset v2 (enrichi)

> Run complet local, k=8, smoke=false, graine 42, CPU, wall 3816 s (~64 min). Cibles T2 =
> schéma hybride EPA-MCL/analytique + garde-fou détection (`src.targets.build_T2`), 10
> labels (`C.T2_LABELS`). **63 features** = mêmes flags que le baseline v1 T2
> (`cocontam="core", include_air=False`) + les 16 features hydrogéo v2 → isole le lift
> d'enrichissement. CV spatiale par blocs + aléatoire groupée, seuils par-label OOF,
> masquage de mesure par label (MNAR). Source : `metrics_v2_t2.json`,
> `figures/t2_per_label_auroc.png`.

---

## 0. Verdict : l'enrichissement v2 relève AUSSI le mur T2 (+0,05 macro-AUROC)

| modèle | macro-AUROC **v1** | macro-AUROC **v2** | lift | micro-F1 v2 | EMR v2 | Δ(rd−sp) v2 |
|---|---|---|---|---|---|---|
| **BinaryRelevance** (référence) | 0,680 | **0,7262** | **+0,046** | 0,575 | 0,214 | +0,187 |
| FreqClassChain (Dong) | 0,668 | **0,7315** | +0,064 | 0,581 | 0,256 | +0,176 |

Même cohérence que T1 : les 16 features hydrogéo v2 relèvent le mur multilabel de ~+0,05
macro-AUROC en généralisation spatiale stricte. Le récit du mémoire tient sur les **deux
tâches** (T1 binaire **et** T2 multilabel).

## 1. Chaînage de Dong vs BinaryRelevance — toujours pas de gain robuste

En v2, `FreqClassChain` passe **marginalement devant** BR (+0,005 macro-AUROC, +0,04 EMR,
+0,006 micro-F1) — alors qu'en v1 il était **derrière**. Mais cet écart **+0,005** est
**dans le bruit inter-plis** (cf. v1 : test apparié chaîne−BR p=0,078, non significatif).
Lecture honnête, identique à v1 : **le chaînage n'apporte pas de gain robuste** ; BR reste
la référence simple et défendable. L'exploitation des dépendances entre labels n'aide pas
en mode prédictif strict (le signal est porté par le contexte, pas par la co-occurrence).

## 2. Per-label — quels PFAS sont prédictibles (BR, CV spatiale)

| PFAS | AUROC | prévalence | n mesuré | lecture |
|---|---|---|---|---|
| **PFNA** | **0,919** | 0,026 | 45 112 | rare mais **très prédictible** — forte structuration source/spatiale |
| PFBA | 0,750 | 0,410 | 25 816 | chaîne courte fréquente, bien prédite |
| PFPeS | 0,719 | 0,158 | 25 948 | |
| PFPeA | 0,715 | 0,410 | 25 738 | |
| PFOA | 0,712 | 0,341 | 46 252 | réglementé, solide |
| PFHpA | 0,702 | 0,265 | 45 099 | |
| PFBS | 0,697 | 0,392 | 44 083 | |
| PFOS | 0,688 | 0,394 | 46 256 | réglementé phare |
| PFHxS | 0,687 | 0,154 | 44 115 | |
| PFHxA | 0,675 | 0,384 | 44 400 | le plus difficile |

- **Tous les labels > 0,67** en AUROC spatiale : signal exploitable et homogène, contraire-
  ment au plancher attendu sur certains rares.
- **PFNA = AUROC 0,919** (rare, prév. 2,6 %) à surveiller : importance probable d'un proxy
  source fortement localisé (à confirmer SHAP si on en fait une cible vedette) — plausible
  mécanistiquement (PFNA = signature industrielle/fluoropolymère ponctuelle), mais à ne pas
  survendre tant que la part « mémorisation spatiale » n'est pas isolée (cf. ablation T1).

## 3. Inflation spatiale (rappel honnêteté)

Δ(random − spatial) macro-AUROC ≈ **+0,18** (BR random 0,913 vs spatial 0,726). Le ~0,91
random rejoint la littérature (Dong et al. ~0,966 split aléatoire) ; le **0,726 spatial**
est la généralisation réelle. À citer systématiquement avec le triplet.

## 4. Lecture pour le mémoire

**Phrase défendable** : « Sur la tâche multilabel, le pipeline enrichi v2 porte la
macro-AUROC spatiale à **0,73** (+0,05 vs base), avec tous les PFAS individuels au-dessus de
0,67. Le chaînage des labels (approche Dong et al.) n'apporte pas de gain robuste : la
prédiction par label indépendant (BinaryRelevance) reste la référence, le signal étant
porté par le contexte hydrogéologique plutôt que par les dépendances inter-PFAS. »

## 5. Reproductibilité
```bash
SMOKE_TEST=1 PFAS_FORCE_CPU=1 python3 experiments/baseline_t2_v2/run_v2_t2.py   # ~45 s
PFAS_FORCE_CPU=1 python3 experiments/baseline_t2_v2/run_v2_t2.py                # ~64 min
```
