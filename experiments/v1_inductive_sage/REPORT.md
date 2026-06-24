# REPORT — V1 : encodeur GraphSAGE-hétéro inductif vs HGT (T1a)

> Run Colab complet, smoke:false, graine 42, k=8, 61 features contexte (mode prédictif strict).
> Source des chiffres : `metrics.json` (run Colab, histories par époque pour les 16 plis) +
> `training_curves_{hgt,hetero_sage_v1}.png`. Résumé tracé dans `metrics_summary.json`.
> Protocole GO-validé par eval (Boucle C, `EVAL_PROTOCOL_V1.md`), socle anti-fuite VALIDÉ 6/6 inchangé.

## 0. Verdict : **V1 REJETÉE** (négatif honnête)

L'hypothèse V1 — *remplacer le HGT par un GraphSAGE-hétéro INDUCTIF (DropEdge + GraphNorm +
neighbor sampling) réduit le sur-apprentissage et améliore la généralisation* — est **réfutée
sur les deux volets**, sous protocole spatial strict :

| encodeur | AUC **spatiale** (per-fold-mean / OOF-glob) | AUC aléatoire | Δ(rd−sp) | gap fit−val | époques (moy) |
|---|---|---|---|---|---|
| **mur XGB tabulaire (in-run, mêmes plis)** | **0,6429 / 0,6879** | — | — | — | — |
| HGT (référence, V0) | 0,6177 / 0,6487 | 0,8012 | **0,1525** | **0,0492** | 175 |
| hetero-SAGE-v1 (inductif) | **0,5819 / 0,6278** | 0,8314 | **0,2037** | **0,0572** | 114 |

> Note tableau : la colonne « AUC spatiale » est *per-fold-mean / OOF-global*. La colonne Δ(rd−sp)
> est calculée **OOF−OOF** (random OOF − spatial OOF) : HGT 0,8012−0,6487=0,152 ; SAGE 0,8314−0,6278=0,204.
> En per-fold-mean−per-fold-mean l'écart serait 0,183 / 0,249 — même conclusion (inflation aggravée pour SAGE).

1. **Généralisation : SAGE < HGT < mur.** hetero-SAGE-v1 fait **−0,036 AUC vs HGT**
   (per-fold-mean 0,582 vs 0,618 ; NB p=0,22, Wilcoxon p=0,20 → **non significatif**), et les
   **deux** restent sous le mur XGB in-run (HGT −0,025 ; SAGE −0,061 ; aucun significatif).
   Verdict pour les deux : `no_robust_gain`.
2. **Sur-apprentissage : NON réduit.** L'écart fit−val du SAGE (0,057) est **≥** celui du HGT
   (0,049) → réduction **−0,008** (le signe attendu était positif). DropEdge/GraphNorm/sampling
   n'ont pas réduit l'écart visé.
3. **Inflation spatiale AGGRAVÉE.** Δ(aléatoire−spatial) passe de 0,152 (HGT) à **0,204** (SAGE) :
   le SAGE colle davantage à la structure spatiale du train.

## 1. Le vrai diagnostic : extrapolation spatiale, pas sur-apprentissage de nœuds

L'écart **fit−val** (~0,05, la métrique cible de V1) **sous-estime massivement** l'échec réel.
Pour les deux encodeurs : **fit AUC ≈ 0,96–0,97**, **val AUC ≈ 0,914**, mais **test spatial (OOF
bloc tenu) ≈ 0,63**. L'effondrement **val→test ≈ −0,30** écrase l'écart fit→val ≈ 0,05.

Raison : le **val** est découpé en micro-blocs spatiaux *internes au train* (donc proches du
train) → val optimiste à 0,91 ; le **test** est un bloc KMeans tenu (région géographique non
vue) → 0,63. **Le goulot est l'extrapolation à des régions non vues, pas le sur-apprentissage
au niveau des nœuds.** Une régularisation inductive qui vise l'écart fit−val ne peut donc pas
corriger ce problème — ce qui explique pourquoi V1 échoue. Les courbes (`training_curves_*.png`)
confirment une convergence propre, sans sous-apprentissage.

## 2. Diagnostic §3.8 (courbes d'entraînement)

`under_training_flag = false` pour les deux encodeurs ; 100 % des plis early-stoppés ; aucun pli
avec best_epoch dans les 20 % finaux. **L'early-stop est stable → les AUC sont endossables**
(la réserve de suivi de l'eval — relever max_epochs/patience si under-training — est **levée**,
inutile de relancer). Le SAGE converge plus vite (≈114 époques) que le HGT (≈175), cohérent avec
sa régularisation plus forte — mais cette convergence plus rapide ne se traduit PAS par une
meilleure généralisation spatiale.

## 3. Conclusion

V1 n'améliore pas V0 : sur ce dataset, en régime spatial strict, **l'encodeur tabulaire XGBoost
reste le mur** (per-fold-mean 0,643), au-dessus du HGT (0,618) et du hetero-SAGE inductif (0,582).
Résultat cohérent avec toutes les expériences GNN précédentes (`no_robust_gain`). La piste utile
n'est pas une meilleure régularisation de l'encodeur, mais le problème d'**extrapolation
spatiale** lui-même (features/arêtes portant un signal généralisable hors-région, ou cibles/protocoles
de transfert géographique explicite).

## 4. Artefacts à committer (depuis le téléchargement Colab)
- `experiments/v1_inductive_sage/metrics.json` (complet, histories par époque — autoritaire)
- `experiments/v1_inductive_sage/training_curves_hgt.png`
- `experiments/v1_inductive_sage/training_curves_hetero_sage_v1.png`
- `experiments/v1_inductive_sage/config.yaml`
- déjà en local : `metrics_summary.json`, ce `REPORT.md`, `EVAL_PROTOCOL_V1.md`
