# REPORT — V2 : fusion non destructrice (T1a)

> Run Colab complet, smoke:false, graine 42, k=8, mode prédictif strict, n_cross_block_total=0.
> Source : `metrics.json` (histories gating par époque) + `figures/{fold_auc_comparison,gating_training_curves,spatial_vs_random_scatter}.png`. Résumé tracé : `metrics_summary.json`.
> Backbone HGT 64-D OOF réutilisé (build_oof_backbone) ; fusions aval ; références XGB seul / stacking V0 / fusion-PCA-95% V0.

## 0. Verdict : **V2 NÉGATIF + PRÉMISSE INFIRMÉE**

Deux questions posées, deux réponses — dont une qui invalide la prémisse même de V2 :

### (1) « La PCA était-elle le coupable ? » → NON, et la prémisse ne tient pas ici.
La motivation V2 (papier §7.4) : *la PCA-95% ne retient qu'UNE composante (HGT_emb_0) → fusion dégradée.* **Sur CE dataset, la PCA-95% garde ~47-48 composantes sur 64** (note `pca_v0_n_components_per_fold` du run V0), PAS 1. **Il n'y a donc aucune destruction à réparer.** Logiquement, toutes les fusions se regroupent (AUC spatiale OOF ≈ 0,68 ; per-fold-mean ≈ 0,64-0,65) et **aucune ne bat fusion_pca95_v0 de façon robuste** :

| fusion | AUC sp (pfm / OOF) | gain vs PCA-95% V0 (pfm) | NB p | Wilcoxon p | verdict |
|---|---|---|---|---|---|
| fusion_pca95_v0 (réf) | 0,6400 / 0,6753 | — | — | — | — |
| v2a full-64D | 0,6509 / 0,6820 | +0,0109 | 0,584 | 0,313 | no_robust_gain |
| v2b PCA k=8 | 0,6422 / 0,6815 | +0,0023 | 0,925 | 0,945 | no_robust_gain |
| v2b PCA k=16 | 0,6544 / 0,6848 | +0,0145 | 0,503 | 0,383 | no_robust_gain |
| v2c gating→XGB | 0,6444 / 0,6715 | +0,0044 | 0,906 | 1,000 | no_robust_gain |

→ Le mécanisme de dégradation décrit dans le papier est un **artefact du dataset du papier** ; il ne se réplique pas ici. La fusion-PCA-95% V0 n'était pas « cassée » sur nos données.

### (2) « Une fusion bat-elle le mur de façon robuste ? » → NON.
| fusion | gain vs mur XGB in-run (pfm) | NB p | Wilcoxon p | verdict |
|---|---|---|---|---|
| mur XGB seul | 0,6429 (réf) | — | — | — |
| v2b PCA k=16 (meilleure) | +0,0115 | 0,638 | 0,547 | no_robust_gain |
| v2a full-64D | +0,0080 | 0,795 | 0,844 | no_robust_gain |
| v2c gating→XGB | +0,0014 | 0,975 | 0,945 | no_robust_gain |
| v2b PCA k=8 | −0,0007 | 0,981 | 0,945 | no_robust_gain |

Aucune fusion ne franchit la règle de réalité (p<0,05 ET >0,03 AUC). **Le mur XGB tabulaire tient** (per-fold-mean 0,643), comme en V0/V1.

## 1. Anomalie : la tête de gating (variante c) est INSTABLE

`v2c_gating` (proba directe du MLP de gating) : **AUC OOF 0,495 — SOUS le hasard**. Cause, visible dans `gating_training_curves.png` et les histories : sur **4 plis sur 8 (0, 3, 4, 5)** la `best_val_auc` est **< 0,5** (0,39 / 0,39 / 0,38 / 0,38) → le MLP apprend une relation **inversée** sur ces blocs spatiaux, et l'early-stop sélectionne quand même ce « meilleur » epoch < 0,5. `mean_gate_value = 0,136` (le gate n'accorde que ~14 % de poids au bloc graphe) → il passe surtout le tabulaire mais s'effondre quand même.

- Le flag §3.8 `under_training=false` **a manqué** ce problème : il teste le *sous-apprentissage* (best_epoch dans les 20 % finaux), pas l'**inversion de signe / val sous le hasard**. À corriger pour une future tête neuronale (ajouter un garde « best_val_auc < 0,5 → pli défaillant »).
- `v2c_gating_xgb` (XGBoost sur `fused_repr`) **récupère le signe** (0,644 pfm / 0,671 OOF) car l'arbre réapprend l'orientation indépendamment du MLP — mais reste sans gain. C'est la seule forme exploitable du gating, et elle ne fait pas mieux que le simple `xgb_seul`.

## 2. Δ(aléatoire − spatial) — inflation spatiale (rappel)
Toutes les fusions restent fortement inflationnées (Δ ≈ 0,16-0,21 ; gating 0,31). Le scatter `spatial_vs_random_scatter.png` montre tous les points loin au-dessus de la diagonale : random ≈ 0,86-0,90, spatial ≈ 0,64-0,68. Le goulot reste l'**extrapolation spatiale** (cohérent avec V1).

## 3. Conclusion
V2 ne « répare » rien parce qu'il n'y avait rien à réparer : la PCA n'était pas destructrice sur nos données. Aucune des trois fusions (64-D complet, PCA k fixe, gating) ne bat le mur XGB tabulaire ni la fusion-PCA-95% V0 de façon robuste. La tête de gating est en plus instable (signe inversé sur la moitié des blocs). **Le mur tabulaire (per-fold-mean 0,643) reste le plafond** — résultat cohérent avec V0 (stacking 0,647) et V1 (HGT 0,618, hetero-SAGE 0,582). Le levier utile reste l'extrapolation spatiale, pas l'ingénierie de fusion.

## 4. Artefacts à committer (depuis l'archive Colab)
- `experiments/v2_fusion/metrics.json` (complet, histories gating — autoritaire)
- `experiments/v2_fusion/figures/{fold_auc_comparison,gating_training_curves,spatial_vs_random_scatter}.png`
- `experiments/v2_fusion/config.yaml`
- déjà en local : `metrics_summary.json`, ce `REPORT.md`
