# REPORT — Baseline XGBoost T1 : confirmatoire ET prédictif × aléatoire ET CV spatiale

> Agent `tabular-ml-engineer` (scripts) + run canonique local (fil principal). Graine 42.
> Cible **T1a** = EPA 2024 NPDWR (`PFOA>4 ∨ PFOS>4 ∨ Hazard Index≥1`), prévalence 44,5 %.
> Réutilise le **socle figé** `src/{config,data,features,splits,metrics}.py` et le protocole
> de référence `experiments/baseline_t1/run_spatial_t1.py` (groupage `gm_well_id`, blocs
> spatiaux KMeans k=8 + aléatoire groupé, seuil F1 **OOF** jamais sur le test).
>
> **Tous les chiffres ci-dessous tracent vers un `metrics.json` committé `smoke:false`, k=8 :**
> - prédictif → `experiments/baseline_t1/metrics_spatial.json`
> - confirmatoire → `experiments/baseline_t1_confirmatory/metrics.json` (run complet 28 min, ce dossier)

---

## 0. Résumé exécutif

| mode | modèle | AUC **spatiale** | AUC **aléatoire** | Δ (rd−sp) | F1 sp | rappel sp | préc. sp | PR-AUC sp |
|---|---|---|---|---|---|---|---|---|
| **confirmatoire** | **XGB** | **1,000** ± 0,000 | 1,000 | **+0,000** | 0,995 | 0,997 | 0,994 | 1,000 |
| confirmatoire | RF | 0,999 ± 0,001 | 1,000 | +0,001 | 0,986 | 0,993 | 0,979 | 0,997 |
| **prédictif** | **XGB** | **0,588** ± 0,068 | 0,900 | **+0,313** | 0,520 | 0,750 | 0,411 | 0,474 |
| prédictif | RF | 0,601 ± 0,056 | 0,898 | +0,297 | 0,554 | 0,968 | 0,397 | 0,490 |

Deux constats, qui sont **le** message de ce baseline :

1. **Confirmatoire = plafond tautologique, PAS un modèle déployable.** T1a est *calculée* à
   partir des concentrations PFAS (`PFOA_ngL>4 ∨ PFOS_ngL>4 ∨ HI≥1`). En remettant les
   colonnes `*_ngL` dans les features, l'arbre **ré-apprend la définition du label** :
   AUC spatiale **1,000**, et l'inflation spatiale **s'effondre à Δ=0,000**. Aucune
   généralisation géographique n'est testée — la cible est déductible ligne à ligne. Ce
   chiffre borne l'**information chimique maximale disponible**, rien de plus. (Même mise en
   garde que Dong et al. 2024 lorsqu'ils incluent les concentrations.)
2. **Prédictif = le vrai mur (mode prédictif strict).** Sans aucune mesure PFAS, l'AUC
   spatiale tombe à **0,588 (XGB) / 0,601 (RF)** — à peine au-dessus du hasard en
   généralisation géographique stricte — avec une **inflation spatiale Δ ≈ +0,30** vs la CV
   aléatoire groupée (~0,90). **C'est ce 0,59-0,60 qui est le mur à battre pour les GNN**,
   pas le 0,90 aléatoire ni le 1,000 confirmatoire.

**Écart confirmatoire → prédictif (AUC spatiale XGB 1,000 → 0,588 ≈ −0,41)** = la valeur
informationnelle de la chimie PFAS de laboratoire. C'est la quantité que la priorisation
pré-prélèvement doit reconstruire à partir du **contexte seul**.

---

## 1. Protocole (identique aux deux modes, sauf le jeu de features)

- **Données** : `data/CA-PFAS-ASGWS.parquet`, 46 338 lignes × 11 333 puits, prév. T1a 44,5 %.
- **Découpage** : groupé par `gm_well_id` (zéro pseudo-réplicat temporel), **CV spatiale par
  blocs KMeans k=8** (référence) **et** CV aléatoire groupée k=8 (pour mesurer Δ).
  `assert_no_group_leak` vérifié sur les deux schémas.
- **Seuil** : F1-optimal estimé en **CV spatiale interne (k=3) sur le train seul** (OOF),
  jamais sur le test. → rappel élevé / accuracy basse, attendu.
- **Modèles** : XGBoost (livrable principal, `n_estimators=400, depth=6`), RF (`300` arbres,
  contexte). Pas d'Optuna (cohérent avec la référence spatiale stricte).
- **Métriques** : 5 standards + PR-AUC + Brier, agrégées sur les 8 plis (moyenne ± écart-type).

### Jeux de features
- **Prédictif — 96 colonnes** (`config.feature_columns(include_location=False, cocontam="all",
  include_air=True)`) : contexte uniquement. **Aucune** mesure PFAS, `*_detected`, `*_ngL`,
  agrégat ou label dérivé. lat/lon, county, `gm_dataset_name` exclus.
- **Confirmatoire — 127 colonnes = 96 + 31 `*_ngL`** : on **rouvre** les concentrations brutes
  (`log1p`). Restent bloqués :
  - les 31 `*_detected` (composants quasi-directs du label : `PFOA_detected ∧ PFOA_ngL>4 ⇒ T1a=1`) ;
  - les 31 `label_*` et les 3 dérivés (`sum_pfas_ngL`, `target_sum_gt70`, `pfas_class_assignment`).

  Garde-fou relâché de façon **explicite et contrôlée** (`ConfirmatoryFeaturePipeline`) :
  il autorise `*_ngL` mais lève toujours si un `*_detected`/`label_*`/dérivé entre dans les
  features (assertions en tête de `main()`).

---

## 2. Lecture

- **Pourquoi Δ≈0 en confirmatoire et Δ≈+0,30 en prédictif ?** L'inflation spatiale mesure ce
  que le modèle gagne en *trichant* sur l'autocorrélation spatiale plutôt qu'en apprenant un
  signal généralisable. En confirmatoire, le signal est *exact* (la cible est dans les
  features) : rien à gonfler. En prédictif, le contexte ne porte qu'un signal spatialement
  structuré → l'écart aléatoire/spatial explose. **L'inflation spatiale n'est donc visible que
  dans le régime opérationnel** — argument de méthode central du projet.
- **RF confirmatoire F1 0,986 < XGB 0,995** : RF, avec seuil OOF, sur-rappelle (rappel 0,993,
  préc. 0,979) ; écart sans portée — les deux sont au plafond.
- Le mur GNN reste **0,59-0,60 (AUC spatiale prédictive)**. Pour mémoire, l'hybride
  GraphSAGE⊕XGBoost committé (`gnn_hybrid_t1/metrics.json`) atteint 0,646 en spatial — au-dessus
  du mur tabulaire prédictif, mais gain **non robuste** au test apparié (Wilcoxon p=0,039,
  Nadeau-Bengio p=0,076 ; cf. `gate5_paired_comparison.py`).

## 3. Reproductibilité

```bash
# smoke (CPU < 2 min) :
SMOKE_TEST=1 python3 experiments/baseline_t1_confirmatory/run_confirmatory_t1.py
# run complet (CPU ~28 min, k=8) — écrit metrics.json smoke:false :
python3 experiments/baseline_t1_confirmatory/run_confirmatory_t1.py
```

Sources de tous les chiffres : `experiments/baseline_t1_confirmatory/metrics.json`
(confirmatoire, k=8, smoke:false, wall 1677 s) et `experiments/baseline_t1/metrics_spatial.json`
(prédictif, k=8, smoke:false). Aucune valeur de ce rapport n'est citée hors de ces deux JSON.
