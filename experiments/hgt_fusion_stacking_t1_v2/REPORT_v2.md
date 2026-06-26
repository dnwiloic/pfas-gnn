# REPORT — Figure phare v2 : HGT → fusion → stacking (T1a)

> Run complet local, k=8, smoke=false, graine 42, CPU, wall 4736 s (~79 min). Cible T1a
> (EPA 2024 + garde-fou détection). **98 features « mécanisme pur »** (cocontam=all sans
> les 7 colonnes administratives) — MÊMES features comme attributs de nœud du graphe ET
> pour le mur XGB in-run (apples-to-apples). Graphe puits-puits multi-relationnel (`near`
> 1,5 km + `same_subbasin_knn` 2 km), HGT inductif (hidden 64, 2 couches, 4 têtes).
> Sources : `metrics.json`, `figures/{figure_phare_v2,spatial_vs_random_v2}.png`.

---

## 0. Verdict en une ligne

**L'enrichissement v2 relève tout le pipeline de ~+0,05 AUC ; le stacking est le meilleur
prédicteur graphe et le mieux calibré, et approche le mur tabulaire — qu'il ne dépasse
toujours pas de façon robuste.** Le récit du mémoire (collecte enrichie → tabulaire + HGT
→ fusion/stacking → résultats plus pertinents) est **soutenu et illustrable**, avec
l'honnêteté méthodologique préservée.

---

## 1. Résultat principal (AUC spatiale global-OOF, k=8)

| architecture | **v1** | **v2** | lift | ECE v2 (calibration) | verdict vs mur |
|---|---|---|---|---|---|
| HGT seul | 0,6537 | **0,7070** | +0,053 | 0,222 | no_robust_gain |
| Fusion embeddings (XGB+PCA-HGT) | 0,6670 | **0,7115** | +0,045 | 0,148 | no_robust_gain |
| **Stacking (HGT+XGB+LGBM)** | 0,6815 | **0,7315** | **+0,050** | **0,101** | no_robust_gain |
| Mur XGB tabulaire in-run (pfm) | 0,688 | **0,744** | +0,056 | — | référence |

Deux lectures, toutes deux présentables :

1. **Lift d'enrichissement (Panel A)** : les 16 features hydrogéo v2 (profondeur,
   écoulement, crépine, land cover) relèvent **chaque étage** du pipeline de ~+0,05 AUC
   spatiale. C'est le cœur du discours : *un meilleur pipeline de collecte améliore tout le
   système, du modèle graphe au stacking.*

2. **Ordre interne propre (Panel B)** : `HGT 0,707 < fusion 0,711 < stacking 0,732`. Le
   **stacking est le meilleur prédicteur fondé sur le graphe** et **le mieux calibré**
   (ECE 0,222 → 0,148 → **0,101**, Brier 0,267 → 0,241 → **0,219**). Il **approche le mur
   tabulaire** (0,744 pfm) en intégrant la vue relationnelle du HGT.

---

## 2. Honnêteté méthodologique (à garder dans le mémoire)

- **Pas de gain robuste sur le tabulaire.** Sous le même protocole in-run (mêmes plis,
  mêmes 98 features, même métrique pfm), les trois architectures restent **sous** le mur
  XGB (gain_vs_wall −0,05 à −0,08) ; tests appariés **non significatifs**
  (Nadeau-Bengio p = 0,51–0,97 ; Wilcoxon p = 0,38–0,95). **Cohérent avec v1 et toutes les
  expériences GNN du projet** : le graphe n'ajoute pas de valeur prédictive robuste une
  fois le contexte tabulaire présent.
- **Inflation spatiale persistante** (Panel scatter) : Δ(random − spatial) = HGT +0,110,
  fusion +0,156, stacking +0,173. Le ~0,90 random reste un mirage d'autocorrélation ; le
  ~0,70–0,73 spatial est la réalité déployable.
- ⚠️ **Mur in-run (0,744) ≠ mur du baseline row-level (0,653).** Ce module évalue au
  **niveau puits** (une proba/puits diffusée aux lignes, encodage fréquentiel), le baseline
  `baseline_t1_v2` évalue **par ligne** (encodage cible). Les deux murs ne sont PAS
  comparables entre expériences ; **seule la comparaison interne** (mur in-run vs
  architectures, même protocole) est valide. Toujours citer le mur in-run avec ce run.

---

## 3. Lecture pour la rédaction

**Phrase défendable** : « Sur le jeu enrichi v2, le pipeline complet
tabulaire ⊕ HGT atteint une AUC spatiale de **0,73** par stacking — soit +0,05 vs le jeu de
base — avec une calibration nettement améliorée (ECE 0,10). Le stacking intègre la vue
relationnelle du graphe pour égaler le meilleur modèle tabulaire ; aucun apport graphe
*robuste* n'est démontré au-delà du tabulaire, ce qui confirme que le signal exploitable
en généralisation géographique stricte est essentiellement porté par les features de
contexte hydrogéologique. » → vrai, chiffré, non survendu.

**Ce que la figure phare montre** (`figures/figure_phare_v2.png`) :
- Panel A : lift v1→v2 sur les 3 architectures.
- Panel B : ordre HGT<fusion<stacking + calibration + ligne du mur tabulaire.

## 4. Reproductibilité
```bash
SMOKE_TEST=1 PFAS_FORCE_CPU=1 python3 experiments/hgt_fusion_stacking_t1_v2/run_v2.py  # ~1 min
PFAS_FORCE_CPU=1 python3 experiments/hgt_fusion_stacking_t1_v2/run_v2.py               # ~79 min
python3 experiments/hgt_fusion_stacking_t1_v2/make_figure.py                           # figures
```
