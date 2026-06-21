# REPORT — GNN phase 2 (P0 : stabilisation T1 · P1 : complétion bipartite T2)

> Agent `gnn-researcher`. Graine 42. Réutilise et ÉTEND le socle figé (`src/{config,
> data,targets,splits,features,metrics}.py`, `src/baselines_t2.py`). Même protocole que
> le MUR non-graphe (`baseline_t1`, `baseline_t2`), conditions C1-C6 (`EVAL_PROTOCOL.md`).
>
> Reproductible :
> `PFAS_FORCE_CPU=1 python3 experiments/gnn_phase2/run_p0_t1.py` → metrics_p0.json
> `PFAS_FORCE_CPU=1 python3 experiments/gnn_phase2/run_p1_t2.py` → metrics_p1.json
> Smoke : `PFAS_FORCE_CPU=1 python3 tests/test_gnn_smoke.py` (10 tests, ~58 s CPU).

## 0. Résumé exécutif (verdict honnête)

| tâche | métrique spatiale | MUR | GNN phase 2 | Δ vs mur | verdict |
|---|---|---|---|---|---|
| P0 T1a | AUC spatial | RF 0,601 | GCN 0,631±0,047 / SAGE 0,615±0,063 | +0,030 / +0,014 | ÉGALE le mur, entraînement stabilisé |
| P1 T2 | macro-AUROC spatial | BR 0,680 | Bipartite 0,681 | +0,001 | ÉGALE le mur, Δ sain, piste MNAR validée |

- P0 a corrigé l'instabilité d'early-stop (collapse des plis disparue : best_epoch 104-189 vs ≤9 ; std GCN 0,074→0,047 ; plus aucun pli GCN<0,55) mais SANS gain net : le plafond T1 est structurel.
- P1 est le résultat marquant : la reformulation MNAR en complétion bipartite puits×analyte ÉGALE le mur BinaryRelevance (0,681 vs 0,680) avec un seul encodeur partagé + embedding d'analyte, gagne sur 5/10 labels (PFOS +0,050, PFHpA +0,038), 0 fuite spatiale audité. Première architecture relationnelle qui tient le spatial sur T2 (valide HYDRO_CRITIQUE).
- Garde-fou Δ respecté : T1 GNN Δ 0,19-0,20 (mur RF 0,30). Le GNN mémorise moins la carte.

## P0 — Stabiliser l'entraînement T1

### Ce qui a changé (src/gnn.py)
1. Validation = plusieurs micro-blocs spatiaux assemblés (`_robust_val_mask_coords`, EVAL_PROTOCOL §2.4) au lieu du bloc unique de phase 1 : TRAIN découpé en n_val_micro=8 micro-blocs KMeans, holdout stratifié val_frac=0,18 de CHAQUE micro-bloc → validation couvrant tout le train, spatiale côté train (pas de fuite test, seuls les nœuds de perte masqués).
2. LR schedule ReduceLROnPlateau (max sur AUC val, factor 0,5, min_lr 1e-5) + patience 50, max 400 époques.
3. run_t1_cv (GraphSAGE+GCN) en CV spatiale ET aléatoire, 8 blocs, données complètes (46 338 lignes/11 333 puits), 33,4 min CPU.

### Triplet AVANT (ph1) / APRÈS (ph2)
| modèle | spatial avant | spatial après | random après | Δ(rd−sp) | std après | vs mur 0,601 |
|---|---|---|---|---|---|---|
| GraphSAGE | 0,618 | 0,615±0,063 | 0,808 | 0,193 | 0,063 | +0,014 |
| GCN | 0,624 | 0,631±0,047 | 0,835 | 0,204 | 0,047 | +0,030 |

### Per-fold (le vrai effet de P0)
| | min pli | max pli | #<0,55 | #<0,60 | best_epoch |
|---|---|---|---|---|---|
| GraphSAGE ph2 | 0,478 | 0,706 | 1 | 2 | 104-176 |
| GCN ph2 | 0,565 | 0,701 | 0 | 2 | 122-189 |

Per-fold spatial GCN ph2 : [0.600, 0.685, 0.565, 0.669, 0.606, 0.583, 0.636, 0.701].

### Lecture honnête P0
Instabilité corrigée (best_epoch ≤9 → 104-189 ; GCN plus aucun pli<0,55 ; std 0,074→0,047) : objectif technique atteint. Mais PAS de gain net (SAGE stationnaire 0,618→0,615 avec un pli résiduel 0,478 ; GCN +0,007, +0,030 sur mur = seuil de bruit). Plafond T1 structurel confirmé : mieux entraîner ne déplace pas le mur. Leviers restants = hybride GNN⊕arbres (P6), ablation d'arêtes (P2).

## P1 — T2 complétion bipartite puits×analyte (MNAR)

### Design (src/gnn_bipartite.py)
Pourquoi : matrice puits×analyte lacunaire MNAR (panel réduit : PFBA/PFPeA/PFPeS ≈26 k vs ≈45 k). Le mur BR = 10 modèles indépendants ignorant la matrice ; les chaînes n'aident pas. La complétion apprend UN encodeur partagé + embedding d'analyte : prédiction puits×a emprunte au contexte, aux autres analytes du puits, aux motifs d'autres puits (GAE/IGMC, prédiction inductive du LABEL d'arête).
Graphe : puits (gauche, contexte FeaturePipeline anti-fuite fit train-only, lat/lon hors C6), analytes (droite, embedding appris), arêtes = cellules mesurées (measurement_mask, disponibilité seule), label = dépassement majoritaire par puits (build_T2). Encodeur SAGE bipartite (analyte→puits & puits→analyte, message passing TRAIN-only → inductif), décodeur MLP [z_w;z_a;z_w⊙z_a], pos_weight par label.

### Contrôles de fuite (audités)
CV externe spatial-block niveau PUITS (train/test sans puits ni bloc partagé). 0 arête bipartite inter-bloc (une arête = un puits = un bloc, C4 vrai par construction, audité n_cross_block_edges=0). Évaluation niveau LIGNE, 5 métriques via multilabel_metrics avec masque par label, seuils OOF. Régime aléatoire pour le Δ.

### Résultat vs mur 0,680
| | macro-AUROC | micro-F1 | micro-recall | micro-precision | subset-acc |
|---|---|---|---|---|---|
| MUR BinaryRelevance | 0,680 | 0,542 | 0,709 | 0,439 | — |
| Bipartite completion | 0,681 | 0,547 | 0,694 | 0,451 | 0,329 |
| Δ (GNN−mur) | +0,001 | +0,005 | −0,015 | +0,012 | — |

**Triplet (régime aléatoire = Δ d'artefact)** : bipartite macro-AUROC **spatial 0,681 ·
aléatoire 0,843 · Δ(rd−sp) = 0,162**. C'est **plus sain que le mur BR** (random 0,902,
Δ 0,222) : comme en T1, le GNN **mémorise moins la carte** (random plus bas) tout en tenant
le spatial. Run complet (spatial+aléatoire) terminé, `metrics_p1.json`.

Par-label (AUROC spatial) :
| label | prév | GNN | mur BR | Δ |
|---|---|---|---|---|
| PFOS | 0,394 | 0,638 | 0,588 | +0,050 |
| PFBS | 0,392 | 0,641 | 0,632 | +0,009 |
| PFHxA | 0,384 | 0,669 | 0,656 | +0,013 |
| PFOA | 0,341 | 0,663 | 0,665 | −0,002 |
| PFHpA | 0,265 | 0,672 | 0,634 | +0,038 |
| PFBA | 0,410 | 0,706 | 0,728 | −0,022 |
| PFPeA | 0,410 | 0,698 | 0,689 | +0,009 |
| PFHxS | 0,154 | 0,657 | 0,660 | −0,003 |
| PFPeS | 0,158 | 0,702 | 0,721 | −0,019 |
| PFNA | 0,026 | 0,766 | 0,831 | −0,065 |

5/10 labels ≥ mur ; delta par-label moyen +0,001.

### Lecture honnête P1
ÉGALE le mur (0,681 vs 0,680, dans le bruit ; micro-F1 +0,005). Ne le bat pas au sens fort mais c'est la première architecture relationnelle qui tient le spatial sur T2 (toutes les chaînes du baseline étaient sous le mur). Profil sain : gains sur labels fréquents, pertes sur le rare PFNA (embedding partagé moins efficace que le mur SMOTE-é). Δ sain (lat/lon hors features, arêtes ne franchissant aucun bloc). Un seul encodeur remplace 10 modèles et complète nativement les cellules non observées. Levier suivant : arêtes puits-puits spatiales (hétérogène) + décodeur VGAE + SMOTE/AP ciblé PFNA.

## Smoke-test CPU (vert) & durées
10 tests, ~58 s CPU (<3 min). Vérifie : PyG CPU (torch 2.12.0+cpu/pyg 2.7.0) ; graphe T1 ≤1,5 km ; C4 2 coupées→0 ; perte T1 0,694→0,220 ; P0 validation ≥2 micro-blocs ; P1 matrice 14 199 cellules, perte bipartite 0,669→0,289, 0 arête train→bloc test, CV spatiale 3 blocs → 5 métriques masquées, n_cross_block_edges=0.
Durées runs complets CPU : P0 33,4 min ; P1 spatial ≈33 min (aléatoire plus lent, graphe non coupé). Colab GPU attendu <5 min.

## Verdict global
- T1 (P0) : ÉGALE le mur (GCN +0,030 au seuil de bruit), entraînement stabilisé (collapse éliminé, std GCN −36 %), PAS de gain net → plafond structurel.
- T2 (P1) : ÉGALE le mur (0,681 vs 0,680), Δ sain, première piste relationnelle tenant le spatial sur T2, profil par-label sain. Base architecturale prometteuse.

## Suite recommandée
1. P1+ (le plus prometteur T2) : graphe hétérogène bipartite ⊕ arêtes puits-puits capées 1,5 km (HGT/R-GCN), décodeur VGAE, récupérer PFNA (SMOTE/AP/focal). Cible 0,68→>0,71 spatial.
2. P6 (le plus crédible T1) : hybride GNN⊕arbres (embedding concaténé, stacking RF/XGB).
3. P2 : ablation d'arêtes (cap×k, k-NN features, source géotracker) jugée au triplet.
4. P3/P4 : catalogue (SGC, APPNP, GAT/GATv2, GIN/PNA), profondeur/agg/dropout d'arêtes.
Tout nouveau protocole (graphe hétérogène P1+) à valider par eval-methodologist avant run long.

## Points de vigilance
1. Gain non net : T1 +0,03 et T2 +0,001 dans le bruit (~0,03-0,06). Le GNN égale, ne bat pas.
2. P0 a corrigé l'instabilité, pas le plafond → levier T1 ≠ optimisation.
3. Δ = garde-fou permanent : juger sur spatial + Δ, jamais random.
4. PFNA (2,6 %) sous-performé (−0,065) : viser l'AP, traitement de rareté dédié.
5. Agrégation puits-majoritaire perd la variation temporelle intra-puits (≈16 %).
6. C4 audité jamais présumé (T1 152 coupées/0 restante ; T2 arêtes mono-puits/0 inter-bloc).
7. Reproductibilité GPU : notebook autonome (zéro Drive, GIT_REF="main"), vérifier import PyG (cellule 3).

### Artefacts
- experiments/gnn_phase2/metrics_p0.json (P0 triplet/per-fold/best-epoch)
- experiments/gnn_phase2/metrics_p1.json (+ metrics_p1_incremental.json) (P1 spatial vs mur, par-label, audit C4, Δ aléatoire)
- experiments/gnn_phase2/{run_p0_t1.py,run_p1_t2.py}
- src/gnn.py (P0), src/gnn_bipartite.py (P1)
- tests/test_gnn_smoke.py (10 tests, ~58 s)
- notebooks/gnn_phase2_colab.ipynb (Colab GPU autonome P0+P1)
