# REPORT — GNN phase 3 (P1+ hétérogène T2) — RÉSULTAT NÉGATIF / inconclusif

> Agent `gnn-researcher`. Graine 42. Étend `src/gnn_bipartite.py` (P1) en graphe
> **hétérogène** (bipartite puits×analyte ⊕ arêtes puits‑puits spatiales). Même protocole
> que le mur (`baseline_t2`), C1–C6. **⚠️ Run interrompu par la limite de session** — les
> chiffres ci‑dessous sont ceux effectivement produits (partiellement sous‑entraînés).

## 0. Verdict honnête

**P1+ n'améliore PAS T2 — il régresse.** Ajouter des arêtes puits‑puits spatiales et un
encodeur hétérogène au‑dessus de la complétion bipartite **n'aide pas** :

| modèle T2 | macro‑AUROC **spatial** | vs mur 0,680 | vs bipartite 0,681 |
|---|---|---|---|
| Mur BinaryRelevance | 0,680 | — | — |
| **Bipartite (P1, phase 2)** | **0,681** | +0,001 | référence |
| P1+ hetero_sage/mlp (run long) | **0,660** | −0,020 | **−0,021** |
| P1+ hetero_sage/mlp (run court 0,6 min) | 0,555 | −0,125 | −0,126 (sous‑entraîné) |

⇒ **La complétion bipartite parcimonieuse (0,681) reste le meilleur GNN T2.** Plus de
structure de graphe (arêtes puits‑puits, hétérogène) **dégrade** — résultat en soi
intéressant (rasoir d'Occam : le modèle relationnel le plus simple gagne ; les arêtes
spatiales n'apportent que du bruit/risque de fuite sans bénéfice).

## 1. Ce qui a été testé (et son honnêteté)

- **Graphe hétérogène** (`src/gnn_hetero.py`) : nœuds puits (contexte `FeaturePipeline`
  anti‑fuite) + analytes (embedding) ; arêtes bipartite (cellules mesurées) **⊕ k‑NN
  puits‑puits capé 1,5 km**. Encodeurs testés : **HGT, R‑GCN, hetero_sage** ; décodeurs
  MLP et **VGAE** (le smoke montre que les 3 encodeurs s'entraînent — perte décroît :
  hgt 0,77→0,24, rgcn 0,78→0,33, vgae 3,34→0,39 — donc **pas un bug**, juste pas de gain).
- **C4 audité, pas présumé** : `cross_bip=0` ET `cross_well=0` (arêtes puits‑puits coupées
  aux frontières de bloc) sur tous les plis. La fuite spatiale est bien contrôlée.
- **PFNA (rare)** : focal loss (γ=1,0) → AP toujours **0,079**, **aucune récupération**.
- Δ (run long, hetero_sage) ≈ sain mais sur un modèle plus faible (spatial 0,66, random plus bas).

## 2. Pourquoi c'est inconclusif (limite de session)

Le run canonique `metrics_p1plus.json` n'a tourné que **0,63 min** (sous‑entraîné →
macro‑AUROC 0,555, labels fréquents < 0,5). Le run plus long
(`metrics_p1plus_incremental.json`) atteint **0,660** mais **reste sous le bipartite
0,681**. La tendance (hetero ≤ bipartite) est cohérente entre smoke, run court et run long,
mais aucun run hétérogène **pleinement convergé** n'a été obtenu avant l'interruption. Un
run GPU complet pourrait raffiner le 0,66 mais **ne renverse pas** la conclusion à ce stade.

## 3. P6 (hybride GNN⊕arbres) — seulement esquissé

L'agent a amorcé la Priorité 6 et écrit un auto‑audit méthodologique
(`experiments/gnn_hybrid_t1/EVAL_PROTOCOL_HYBRID.md`, style eval‑methodologist) :
**triplet GNN seul / XGBoost seul / hybride APPROUVÉ** sous le protocole spatial, **MAIS**
arêtes mécanistes **REFUSÉES** par mesure : « même type de source géotracker » = 4 cliques
(jusqu'à 4 755 puits, 19,5 M arêtes, η avec T1a = **0,081** → pas un mécanisme mais une
partition de carte) ; clique de sous‑bassin idem. **Aucun run hybride exécuté.**
(`src/config.py` reçoit un constant `SGMA_SUBBASIN` pour contraindre d'éventuelles arêtes
mécanistes, jamais comme feature de nœud.)

## 4. Smoke-test CPU
**VERT** (~112 s CPU, `PFAS_FORCE_CPU=1`) : graphe hétérogène construit (bons types), les
3 encodeurs + VGAE s'entraînent (perte ↓), **C4 = 0 arête inter‑bloc (bipartite ET puits‑
puits)**, CV spatiale 3 blocs → 5 métriques masquées. Aucun process orphelin (vérifié).

## 5. Recommandation (honnête)

1. **Garder la complétion bipartite P1 (0,681) comme GNN T2 de référence** — la plus
   simple ET la meilleure. Ne pas sur‑investir l'hétérogène (résultat négatif clair).
2. **P6 hybride GNN⊕arbres** (le plus crédible pour un gain net) reste à exécuter : embedding
   bipartite concaténé aux features → stacking RF/XGB, sous le triplet spatial. Protocole
   déjà auto‑audité (arêtes mécanistes refusées).
3. Sinon, **conclure la phase GNN** : le message scientifique est robuste — les GNN
   **égalent** le mur spatial (T1 ~0,62, T2 0,68) **sans le battre**, et l'architecture la
   plus simple (bipartite) est la meilleure ; la contribution est l'évaluation spatiale
   honnête + la quantification de l'inflation, pas un GNN miracle.

### Artefacts
- `src/gnn_hetero.py` (HGT/R‑GCN/hetero_sage + VGAE) · `experiments/gnn_phase3/{run_p1plus_t2.py,
  config.yaml, metrics_p1plus.json, metrics_p1plus_incremental.json}` · `notebooks/gnn_phase3_colab.ipynb`
- `experiments/gnn_hybrid_t1/EVAL_PROTOCOL_HYBRID.md` (auto‑audit P6, run non exécuté)
- `tests/test_gnn_smoke.py` (étendu hétérogène, ~112 s)
