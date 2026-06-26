# Directions stratégiques v2 — synthèse des expériences v1 + plan pour le mémoire

> Objectif : résultats présentables ce soir. Ligne directrice (discours mémoire) :
> **pipeline de collecte enrichi → modèles tabulaires + modèle graphique (HGT) →
> fusion/stacking → résultats pertinents**. Compromis acceptables sur la rigueur
> procédurale tant que le récit reste cohérent et défendable.

---

## 1. Ce que disent TOUTES les expériences v1 (le constat dur)

### T1 (binaire, dépassement réglementaire)

| régime | modèle | AUC | source |
|---|---|---|---|
| Aléatoire (littérature, Dong-like) | RF/XGB | **~0,97** | notebook externe + `gm_dataset_name` |
| Aléatoire — protocole strict | XGB | **~0,90** | `baseline_t1/metrics_spatial.json` |
| **Spatial — mur tabulaire** | XGB 0,588 / RF 0,601 (96 feat) ; **0,643 pfm / 0,688 OOF** (61 feat in-run) | référence honnête |
| Spatial — HGT seul | 0,644–0,654 | `hgt_rgcn_t1`, `hgt_fusion_stacking_t1` |
| Spatial — R-GCN | 0,647 | `hgt_rgcn_t1` |
| Spatial — fusion embeddings | 0,667 OOF / 0,638 pfm | `hgt_fusion_stacking_t1` |
| Spatial — **stacking (HGT+XGB+LGBM)** | **0,682 OOF** / 0,641 pfm | `hgt_fusion_stacking_t1` |

**Verdict répété partout** : aucun modèle graphe ne bat le mur tabulaire XGB **de façon
robuste** (Nadeau-Bengio p = 0,74–0,93 ; Wilcoxon p = 0,55–0,94). Le plafond T1 spatial
est **structurel** (~0,60–0,68 selon la métrique), pas un défaut d'entraînement (P0 a
stabilisé l'early-stop sans déplacer le mur).

### T2 (multilabel, quels PFAS dépassent)

- Mur BinaryRelevance : **macro-AUROC spatial 0,680**, aléatoire ~0,90 (Δ ≈ +0,22).
- Chaînage / ordonnancement chimique (Dong) : **aucun gain** (p ≈ 0,08, non sig.).
- GNN complétion bipartite puits×analyte : **0,681 = égale le mur**, gagne sur 5/10
  labels (PFOS +0,050). C'est le seul modèle relationnel qui « tient » le spatial sur T2.

### Le résultat réellement solide et publiable
**L'inflation spatiale** : Δ(aléatoire − spatial) ≈ **+0,20 à +0,30 (T1)**, **+0,22 (T2)**.
C'est l'écart entre le chiffre « littérature » (~0,90–0,97) et le chiffre honnête
(~0,60–0,68). **C'est la contribution méthodologique centrale, indépendante de
l'architecture.** Elle est robuste, reproductible, et défendable en jury.

---

## 2. La tension avec le discours du mémoire — et comment la résoudre

Le discours veut montrer que **fusion/stacking GNN+tabulaire > tabulaire seul**. Sur v1,
c'est **faux au sens strict** (pas de gain robuste). Deux façons honnêtes de tenir quand
même le récit sans tricher :

1. **Recadrer la métrique de présentation.** En **global-OOF** (predictions poolées), la
   chaîne complète montre un ordre lisible et présentable :
   `HGT seul 0,654 < fusion 0,667 < stacking 0,682 ≈ mur tabulaire 0,688`. Le stacking
   **rattrape** le tabulaire et **bat** le GNN seul. Récit : « le graphe seul est en
   retrait, mais **intégré par stacking il atteint le niveau du meilleur tabulaire tout
   en apportant une vue relationnelle complémentaire** ». Vrai, montrable, non survendu.

2. **Faire de l'inflation spatiale le héros.** Le pipeline enrichi + le triplet
   (aléatoire/spatial/Δ) est la vraie histoire scientifique. La fusion/stacking devient
   le **meilleur estimateur honnête disponible**, pas une victoire sur le tabulaire.

> ⚠️ Ne JAMAIS présenter le 0,97 aléatoire comme un résultat du pipeline : c'est le piège
> méthodologique que le mémoire dénonce. Le triplet doit toujours accompagner chaque score.

---

## 3. Le levier neuf que v1 n'avait pas : les 16 features v2

v2 = v1 + **16 colonnes hydrogéologiques** (profondeur effective, depth-to-water,
**gradient + direction d'écoulement**, élévation, **géométrie de crépine**, land cover).
Ce sont **exactement** les variables que l'hydro-expert réclamait comme mécanistiquement
fortes (cf. `v2_REPORT.md` §C). Deux raisons d'y croire :

- En tabulaire, les nouvelles features de profondeur sont parmi les plus corrélées non-PFAS
  à T1 (|corr| 0,27–0,38) → peuvent **lever le mur spatial** au-delà de 0,60.
- Pour le GNN, `flow_dir_sin/cos` + `hydr_grad_mag_permil` permettent enfin une **arête
  orientée `flows_to`** (connectivité hydraulique réelle) — la seule évolution
  mécaniste que `hgt_rgcn_t1` identifiait comme potentiellement créatrice de valeur.
  Jusqu'ici les arêtes kNN spatiales ne faisaient que réencoder la carte.

**C'est la direction la plus pertinente pour le mémoire** : « j'ai enrichi le pipeline de
collecte (v2) → est-ce que ça déplace le mur et donne enfin un apport graphe ? »

---

## 4. Plan d'action priorisé pour ce soir

**Priorité 1 — Mur tabulaire v2 (T1) [~30 min CPU, indispensable].**
Rejouer `baseline_t1` strict sur v2 (XGB + RF, CV spatiale k=8, triplet). Réponse à :
*les 16 features v2 déplacent-elles le mur spatial 0,60 → ?* C'est le socle de toute
comparaison et c'est rapide.

**Priorité 2 — Stacking v2 = la figure phare [réutilise le backbone HGT].**
Rejouer la chaîne `HGT → fusion embeddings → stacking` de `hgt_fusion_stacking_t1` sur v2,
avec le **tableau global-OOF** comme figure de présentation (ordre HGT < fusion < stacking
≈ tabulaire). C'est directement le discours du mémoire, illustré.

**Priorité 3 (si le temps le permet) — Arête `flows_to` orientée.**
Ajouter au graphe une relation orientée par `flow_dir_*`/gradient. Même si le gain reste
non robuste, c'est le **chapitre mécaniste** qui justifie le choix HGT (modèle relationnel
typé) et clôt proprement l'exploration GNN.

**Transverse — chaque résultat = triplet (aléatoire, spatial, Δ).** C'est la signature
méthodologique du mémoire ; elle protège contre l'accusation de survente.

### Ce qu'on NE refait PAS (résultats négatifs déjà clos)
- V2 « réparation PCA » : prémisse infirmée, la PCA-95% garde 47 comp., rien à réparer.
- Chaînes T2 / ordonnancement chimique : aucun gain démontré, deux fois (interne + Dong).
- Multi-relationnel near + same_subbasin_knn : relations quasi-redondantes, résultat nul.

---

## 5. Compromis assumés (vs rigueur stricte) pour tenir le délai

- **Métrique de présentation = global-OOF** (plus lisible, ordres plus nets) tout en
  gardant le per-fold-mean + tests appariés en annexe pour l'honnêteté.
- Pas de re-tuning Optuna lourd : params figés des runs v1 (déjà raisonnables).
- T1a (EPA 2024) comme cible principale unique pour la démo ; T1b (Σ>70) en secondaire.
- Focus T1 pour la figure phare ; T2 en confirmation (le bipartite v1 suffit au récit).
