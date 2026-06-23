# HGT / R-GCN multi-relational T1 — résultats GPU (production)

Graphe homogène puits-puits, deux types d'arêtes :
- `('well','near','well')` : k-NN spatial, cap 1,5 km
- `('well','same_subbasin_knn','well')` : k-NN intra-sous-bassin SGMA, cap 2 km

Eval-validated design (experiments/hgt_rgcn_t1/eval_validation.md) :
pas de nœud source fabriqué ; HGT/R-GCN utilisés comme encodeurs RELATIONNELS sur un seul
type de nœud réel (C-NODE.1/2/3). Run complet GPU Colab, smoke=False, seed=42.

- features: 61 cols, include_location=False (C-LOC.1: lat/lon PAS en features de nœud)
- inductive (C-SPAT.4): True  — MP restreint aux arêtes TRAIN-TRAIN pendant l'entraînement
- relations: ['near', 'same_subbasin_knn'], hidden=128, layers=3, heads=4

## Résultats (row-level OOF, comparables au mur non-graphe et à gnn_phase1)

| modèle | régime | AUC OOF | IC 95% | AUC mean±std | F1@OOF | PR-AUC | bal.acc | Brier | ECE | xblock |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| R-GCN | spatial | **0.6472** | [0.628, 0.665] | 0.5990±0.0585 | 0.6399 | 0.5604 | 0.6016 | 0.3098 | 0.2644 | 0 |
| R-GCN | random  | 0.8214 | [0.806, 0.837] | 0.8272±0.0223 | 0.7419 | 0.7746 | 0.7580 | 0.1898 | 0.1313 | 0 |
| HGT   | spatial | **0.6440** | [0.623, 0.664] | 0.5961±0.0681 | 0.6064 | 0.5635 | 0.6124 | 0.3268 | 0.3019 | 0 |
| HGT   | random  | 0.8078 | [0.792, 0.823] | 0.8096±0.0175 | 0.7302 | 0.7593 | 0.7407 | 0.1815 | 0.0611 | 0 |

**Δ(random − spatial) : R-GCN +0.1742, HGT +0.1638**
(C-SPAT.6 : inflation spatiale de ~0.17 pt AUC. Le score aléatoire est un artefact d'autocorrélation
spatiale, sans valeur décisionnelle. Confirms that random split is INVALID on this dataset.)

## Leakage guard (C-SPAT.2 / C-SPAT.5)

Cross-block edges cut SEPARATELY per relation and asserted to 0 on every fold and regime.
Total residual cross-block edges: **0** pour les deux modèles, les deux régimes (must be 0).

Audit fold 0 (exemple) :
- removed_near=19, removed_subbasin=24 (spatial) → résiduel 0
- removed_near=27 399, removed_subbasin=28 844 (random, beaucoup plus d'arêtes inter-blocs)

## Comparaison avec les baselines (même protocole, k=8 blocs spatiaux)

| modèle | AUC spatial OOF | AUC mean±std | Δ rand−spat |
|---|---:|---:|---:|
| RF wall (tabulaire) | ~0.600 | — | — |
| GraphSAGE (phase 1, mono-rel) | 0.618 | ±0.067 | +0.196 |
| GCN (phase 1, mono-rel) | 0.624 | ±0.074 | +0.218 |
| **HGT (multi-rel, ce run)** | **0.644** | **±0.068** | **+0.164** |
| **R-GCN (multi-rel, ce run)** | **0.647** | **±0.059** | **+0.174** |

**Verdict C-CMP :** gains HGT et R-GCN vs GraphSAGE/GCN : +0.020–0.029. Ces gains sont
**inférieurs au bruit inter-plis** (σ spatial ≈ 0.06–0.07). Selon la condition C-CMP de
l'eval-methodologist, ils ne peuvent pas être revendiqués comme amélioration réelle.
Les tests statistiques appariés (rapport eval_postrun.md) confirmeront ou infirmeront.

## Calibration (C-CAL)

Les deux modèles sont **sévèrement mal calibrés en régime spatial** :
- ECE spatial : R-GCN 0.264, HGT 0.302 (vs random : 0.131 / 0.061)
- Bin [0,0.1] (spatial) : R-GCN 15 096 lignes, conf=2.5%, frac_pos=30.2% — massive sous-estimation
- Bin [0.9,1.0] (spatial) : R-GCN frac_pos=59.3%, HGT 60.6% — surconfiance (modèle croit à 96%, raison à 60%)
- La calibration se dégrade fortement quand le modèle doit extrapoler géographiquement
- Courbes de fiabilité complètes : voir metrics_hgt.json / metrics_rgcn.json

**Recommandation** : calibration post-hoc (Platt ou isotonic, ajustée OOF) avant usage décisionnel.

## Instabilité inter-plis

| pli | AUC R-GCN | AUC HGT |
|---|---:|---:|
| 0 | 0.506 | **0.463** |
| 1 | 0.621 | 0.606 |
| 2 | 0.532 | 0.550 |
| 3 | 0.675 | 0.578 |
| 4 | 0.577 | 0.611 |
| 5 | 0.653 | 0.687 |
| 6 | 0.659 | 0.686 |
| 7 | 0.568 | 0.587 |

Pli 0 (HGT 0.463 < hasard) et pli 2 (0.53) : probable région à régime hydrogéologique
différent (aquifères profonds, puits hors bassins alluviaux, dont les 1 475 puits sans
sous-bassin SGMA). L'instabilité (σ ≈ 0.06–0.07) est attendue et constitue un résultat
mécaniste valide (stationnarité spatiale fausse du signal PFAS).

## Verdict hydro-expert (hydro_expert.md)

| élément | verdict |
|---|---|
| Arête `near` (1.5 km) | PLAUSIBLE sous condition du cap. Recommande 1.0 km par défaut. |
| Arête `same_subbasin_knn` (2 km) | DISCUTABLE — quasi-redondante avec `near` à ces distances. |
| Distinction near vs subbasin | DISCUTABLE — explique structurellement le résultat nul multi-rel. |
| Source→puits en feature | NON SUFFISANT — isotrope ; manque la direction d'écoulement. |

**Conclusion hydro** : le faible apport du graphe est un résultat mécaniste cohérent, pas un
défaut d'implémentation. La physique exploitable par un graphe non orienté est déjà captée par
les features de voisinage. La seule évolution mécaniste réelle = arête orientée `flows_to`
(gradient hydraulique), qui rendrait les deux relations vraiment distinctes.

## Recommandations pour la suite

1. **Ne pas investir davantage dans HGT/R-GCN multi-rel** tant que les deux relations sont
   quasi-redondantes. Résultat négatif documenté, à revendiquer comme tel.
2. **Ablation obligatoire** : `near` seul vs `near + same_subbasin_knn`. Si Δ < σ inter-plis
   (~0.06), retirer `same_subbasin_knn` par parcimonie.
3. **Cap ablation** : tester `cap_km ∈ {0.5, 1.0, 1.5}` pour `near`.
4. **Prochaine arête mécani ste** : `flows_to` orientée par gradient hydraulique (DWR), si
   une couche piézométrique est intégrable. Seule relation vraiment nouvelle.
5. **Calibration post-hoc** à tester avant usage décisionnel (ECE 0.26–0.30 en spatial).
6. **Cartographier AUC par bloc** sur la carte géographique, croisé avec prévalence locale
   et densité de puits, pour documenter l'instabilité mécanistiquement.
7. Lire eval_postrun.md (eval-methodologist) pour les tests statistiques appariés formels.

## Positionnement littérature

Dong et al. (2024) : macro-AUC élevée avec split aléatoire (non comparable au split spatial
honnête). Le résultat central de ce projet est que le split aléatoire gonfle l'AUC de ~0.17–0.20 pt
— précisément l'écart entre le chiffre de la littérature et le chiffre honnête. C'est la
contribution méthodologique principale, indépendante de l'architecture GNN choisie.
