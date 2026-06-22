# REPORT — GNN-hybrid T1 (mechanistic edges + GraphSAGE⊕XGBoost)

> Expérience "Priorité 6 (hybride GNN⊕arbres) + Priorité 2 (arêtes mécanistiques)" de
> `experiments/gnn_phase1/REPORT.md` §6. Protocole **validé sous conditions** par
> `eval-methodologist` — voir `EVAL_PROTOCOL_HYBRID.md` (contrat §C, bloquant).
> Cible **T1a** (`experiments/profilage/REPORT.md`). Graine 42. Runs lourds = Colab GPU.

---

## Graph + GNN (gnn-researcher)

Scope: mechanistic edges + GraphSAGE embedding extraction + inductive `train_gnn_and_embed`
primitive. Ne touche PAS à XGBoost/fusion (la boucle OOF/nested externe = tabular-ml-engineer).
`src/graph.py` reste torch-free ; torch vit dans `src/gnn.py`. Aucun run lourd local
(CLAUDE.md §4/§5 ; mémoire "heavy-runs-colab-only") — seulement un smoke CPU < 3 min ; le run
complet est Colab GPU via la boucle §3.

**1. Arête mécanistique — k-NN intra-sous-bassin distance-capé (la seule admissible, §2.4).**
`build_well_graph` porte désormais DEUX constructeurs d'arêtes via `relation=`, avec C4 appliqué
aux DEUX (contrat §2.1/§2.4) : `"spatial"` (k-NN spatial nu, cap par défaut 1,5 km = baseline
phase 1) et `"subbasin_knn"` (k-NN restreint aux puits partageant `sgma_subbasin_name`, défaut
k=8 cap 2 km = mécanistique). Définition (`graph.knn_edges_intra_subbasin`) : pour chaque puits,
relier ses k plus proches voisins QUI PARTAGENT le sous-bassin SGMA, dans un cap grand-cercle dur.
Deux puits à 1 km de part et d'autre d'une crête hydrogéologique (sous-bassins différents) ne sont
PAS reliés — prior d'aquifère, pas un re-encodage de la carte. Les puits sans sous-bassin n'ont
AUCUNE arête mécanistique (1 475 puits, 13,0 %, isolés sous cette relation). `sgma_subbasin_name`
n'est jamais une feature de nœud (η=0,505 = confondeur géographique géré par C5 ; pas un dérivé de
cible, ∉ blocklist) — il ne sert qu'à contraindre l'arête (nouveau `config.SGMA_SUBBASIN`). Les
designs refusés (clique sous-bassin pleine, cliques type-de-source) ne sont pas implémentés ;
`build_well_graph` lève `ValueError` sur toute relation non approuvée.

Comptages mesurés (données complètes, graine 42, KMeans k=8, niveau puits) : spatial cap 1,5 km →
30 496 pré-C4, 19 coupées, **30 477 post-C4**, 0 inter-bloc restante (contrat : 30 477 exact).
subbasin_knn k=8 cap 2 km → 32 660 pré-C4, **24 inter-bloc coupées** (contrat : 24 exact),
0 restante, **toutes les arêtes intra-sous-bassin**, max 2,000 km, 1 475 puits isolés. L'écart
~32 194 vs 32 636 est l'agrégation sous-bassin mode-vs-first ; les invariants porteurs (24
inter-bloc, 0 restante, 100 % intra-sous-bassin) tiennent. Honnêteté (§2.4) : à cap ≤2 km le graphe
mécanistique partage la plupart de ses arêtes avec le k-NN spatial nu ; l'apport sera faible sans
facility_id/gradient — rapporter le Δ(spatial,random), revendiquer un gain seulement au-dessus du
bruit inter-pli (~0,03) sur le score SPATIAL.

**2. Extraction d'embedding.** `build_model(... embed=True)` retourne la représentation cachée
pré-tête `[n_nodes, hidden]` (dernier LayerNorm+ReLU, avant la tête linéaire) ; `embed=False`
retourne les logits `[n_nodes]` (phase 1 inchangée). `model.embed_dim = hidden`, journalisé ;
déterministe en mode eval.

**3. Primitive `train_gnn_and_embed` + `EmbedInfo`.** Signature :
```python
emb, info = gnn.train_gnn_and_embed(
    df, y_row, feature_cols, fold_block, *,
    fit_blocks, embed_blocks,
    relation="subbasin_knn", k=8, cap_km=None,   # None -> 1.5 (spatial) / 2.0 (subbasin)
    model_name="graphsage", hidden=64, layers=2, dropout=0.5,
    lr=5e-3, weight_decay=5e-4, max_epochs=400, patience=50,
    val_frac=0.18, n_val_micro=6, lr_schedule=True,
    encode="frequency", early_stop=True, seed=42)
# emb : np.ndarray [n_embed_nodes, hidden]  (pré-tête ; ligne i <-> info.embed_well_ids[i])
# info: gnn.EmbedInfo (embed_dim, n_fit_nodes, n_embed_nodes, n_removed_cross_block,
#       n_cross_block_remaining==0, best_epoch, final_loss, relation, cap_km, k,
#       + embed_well_ids, embed_node_idx, row_to_node, well_ids pour la rediffusion ligne)
```
Garanties anti-fuite (chacune assertée) : `fit_blocks ∩ embed_blocks = ∅` ; graphe construit sur
fit∪embed avec `cut_blocks=True` et `info.n_cross_block_remaining == 0` ; FeaturePipeline fit sur
les nœuds FIT uniquement ; perte + early-stop ne touchent jamais les nœuds embed ; embedding
pré-tête, dim journalisée. Usage : `embed_blocks=[j]` sur les micro-blocs internes pour §3.2
(embeddings train OOF) ; `embed_blocks=[test_block]` sur les blocs externes pour §3.3 (embeddings
test). N'implémente PAS la boucle XGBoost.

**Smoke CPU (final) :**
```
mechanistic: nodes=1500 undirected_edges=1305 max_km=2.000 missing_subbasin=189
C4[spatial]: removed(undir)=2 remaining=0 would-be(dir,uncut)=4
C4[subbasin_knn]: removed(undir)=2 remaining=0 would-be(dir,uncut)=4
embed(): emb=(1500, 48) logit=(1500,) embed_dim=48
primitive: emb=(154, 32) fit_nodes=1105 embed_nodes=154 removed_xblock=2 cross_remaining=0 best_epoch=28 loss=0.5218
SMOKE OK in 76.9s
```
Fichiers : `src/graph.py`, `src/gnn.py`, `src/config.py`, `tests/test_gnn_smoke.py`
(18 tests verts CPU ~77 s).

---

## Fusion + XGBoost (tabular-ml-engineer)

Scope: `src/hybrid.py` (nested-OOF loop + three-way comparison harness),
`experiments/gnn_hybrid_t1/run_hybrid_t1.py` (driver with `SMOKE_TEST` toggle),
`tests/test_hybrid_smoke.py` (10 guards, §3.4 anti-leak assertions). Graine 42
partout. Toutes les sorties dans `experiments/gnn_hybrid_t1/`. Aucun entraînement
lourd local (CLAUDE.md §4/§5, mémoire "heavy-runs-colab-only").

### 1. Implémentation de la boucle OOF/nested (§3 exact)

Le module `src/hybrid.py` implémente la boucle décrite au §3.2–§3.3 :

**Boucle externe (§3.2) :** pour chaque bloc test `f` du découpage spatial (LOBO),
`run_one_outer_fold` :
1. Construit les **blocs spatiaux internes** (`J` micro-blocs KMeans) sur le seul
   DataFrame d'entraînement (`train_f`).
2. Pour chaque micro-bloc interne `j` : appelle
   `gnn.train_gnn_and_embed(df_train, ..., fit_blocks=[inner\j], embed_blocks=[j])`
   — chaque ligne de train obtient un embedding produit par un GNN qui n'a ni vu son
   label ni reçu d'arête cross-bloc (C4 assertée, `n_cross_block_remaining == 0`).
3. Construit les **embeddings de test** en appelant
   `gnn.train_gnn_and_embed(df, ..., fit_blocks=train_f_blocks, embed_blocks=[f])`
   — le bloc test n'est jamais dans `fit_blocks` (asserté).
4. Fusionne `[tabular_features ⊕ embedding]` au niveau puits (diffusion à la ligne
   via `row_to_node`), entraîne XGBoost (`class_weight` balancé / `scale_pos_weight`),
   seuil et calibration Platt depuis les probas OOF internes UNIQUEMENT (§4.4).

**Bras aléatoire :** même boucle avec `group_random_folds` — `Δ(random−spatial)`
mesure l'artefact d'inflation spatiale.

**Deux relations** (§C.3) : `"subbasin_knn"` (mécanistique, cap 2 km) et `"spatial"`
(k-NN nu, cap 1,5 km). Leur Δ l'une de l'autre mesure l'apport mécanistique.

### 2. Comparaison trois bras — mêmes plis / même graine / même `fold_block`

`run_three_way_comparison` assemble le tableau triplet
`(random, spatial, Δ)` par bras (§4.3) avec :
- ROC-AUC, PR-AUC, rappel + précision @ seuil, balanced accuracy,
  Brier + ECE + courbe de fiabilité (calibration), gain cumulé / lift @ 10 % et 20 %.
- IC95 % bootstrap **par groupe (`gm_well_id`)** sur les probas OOF concaténées (§4.5) :
  resampling au niveau puits (pas ligne) pour éviter des intervalles artificiellement
  étroits dus aux pseudo-réplicats.
- Test apparié : **corrected resampled t-test (Nadeau-Bengio 2003)** et **Wilcoxon signé**
  sur les paires AUC par pli.
- **Règle de réalité (§4.5)** : un gain hybride vs mur n'est "real" que s'il est (a)
  significatif (p < 0,05, l'un ou l'autre test) ET (b) > seuil de bruit 0,03 AUC
  spatial. Les résultats ci-dessous (smoke) montrent les deux conditions non satisfaites —
  ce qui est attendu sur 500 puits / 15 époques GNN.

### 3. Garde-fous §3.4 — résultats smoke (10 tests verts, CPU 86 s)

```
[guard 1] assert_no_group_leak: outer + all inner folds — PASS
[guard 3] test-embed GNN: removed=0 remaining=0 — PASS
[guard 3] inner-OOF GNN: removed=0 remaining=0 — PASS
[guard 2] test_b=1 not in fit_block_ids=[0, 2] — PASS
[guard 4] fit_blocks [1] ∩ embed_blocks [0] = ∅ — PASS
[guard 4] overlap correctly rejected: fit_blocks and embed_blocks overlap: [0] — PASS
[guard 5] embedding shape=(183, 16) embed_dim=16 finite=True — PASS
[guard 6] fused features (217, 108) → XGB fits (proba finite) — PASS
[guard 7] all §4.3 metrics compute: [roc_auc, pr_auc, recall, precision, f1,
          accuracy, balanced_accuracy, brier, ece, gain_top20pct, ...] — PASS
[guard 8] threshold=0.070 (from OOF, not test) — PASS
[guard 9] checkpointing: metrics_incremental.json written — PASS
[end-to-end] spatial_AUC=0.7019 random_AUC=0.7011 Δ=-0.0007 (3 folds, 400 wells)
ALL GUARDS GREEN in 86.3s — Smoke test PASSED.
```

Driver smoke (SMOKE_TEST=True, 2 relations × 2 bras × 3 outer = 12 GNN entraînements) :
```
[subbasin_knn] hybrid spatial AUC=0.563  random AUC=0.726  Δ=+0.163
[spatial]      hybrid spatial AUC=0.557  random AUC=0.726  Δ=+0.169
THREE-WAY COMPARISON (subbasin_knn, smoke — non représentatif, 500 puits / 15 époques)
  hybrid        spatial=0.563  random=0.718  delta=+0.154
  gnn_alone     spatial=0.605  random=n/a    delta=n/a
  xgb_alone     spatial=0.588  random=n/a    delta=n/a
  Hybrid gain over XGB wall: -0.025
  Verdict: spurious (not significant) [attendu en smoke]
Total elapsed: 148 s
```

### 4. Caveat alignement des embeddings (Watch-out documenté)

Chaque GNN interne (et le GNN de test) est **initialisé indépendamment** avec la même
graine 42, mais converge sur un sous-ensemble de données différent. Les **axes
pré-tête ne sont donc PAS comparables d'un pli à l'autre** : la colonne `emb_0` du pli
interne `j=0` peut encoder une direction latente orthogonale à `emb_0` du pli `j=1`.
XGBoost traite ces colonnes comme des features bruitées plutôt que comme une
représentation partagée — c'est une **limitation connue du stacking neuronal OOF**
(appelée "embedding axis misalignment" dans la littérature d'ensemble neuronal). Elle
tend à **sous-estimer** le vrai bénéfice d'un hybride entraîné conjointement.

Mitigation implémentée : graine identique (42) pour toutes les initialisations GNN
(réduit la variance de rotation des axes). La dimension `hidden` est fixée sur tous
les plis (schéma de feature XGB stable). Cette limitation est documentée ici et dans
le notebook Colab — elle ne doit pas être masquée dans les conclusions finales.

### 5. Coût estimé du run complet

| Composante | Nombre de GNN entraînements |
|---|---|
| Spatial arm, subbasin_knn : K=8 outer × (J=4 inner + 1 test) | 40 |
| Random arm, subbasin_knn  : K=8 × 5 | 40 |
| Spatial arm, spatial      : K=8 × 5 | 40 |
| Random arm, spatial       : K=8 × 5 | 40 |
| **Total** | **160** |

À ~10–20 min/GNN sur Colab T4 GPU (full data, 400 époques) : **27–53 h** pour les
deux relations. En pratique, la relation `spatial` est plus rapide (moins d'arêtes) ;
estimer **~30–40 h** au total. Recommandation : lancer les deux relations sur des
sessions Colab parallèles (une par relation), soit **~20 h par session**.

### 6. Lancement

**Smoke (CPU, < 3 min) :**
```bash
PFAS_FORCE_CPU=1 python3 tests/test_hybrid_smoke.py
PFAS_FORCE_CPU=1 python3 experiments/gnn_hybrid_t1/run_hybrid_t1.py   # SMOKE_TEST=True
```

**Full run (Colab GPU) :** ouvrir le notebook Colab dédié (à créer par
`colab-notebook-engineer`), qui clone le dépôt, installe les dépendances, bascule
`SMOKE_TEST=False` dans `run_hybrid_t1.py` (ou `RELATIONS = ["subbasin_knn"]` pour
une seule session), puis exécute le driver. Checkpoints par pli dans
`experiments/gnn_hybrid_t1/run_subbasin_knn/spatial/metrics_incremental.json`.
En fin de run : `files.download()` de `experiments/gnn_hybrid_t1/metrics.json` OU
`git add/commit/push` (zéro Drive, cf. CLAUDE.md §4).

### 7. Positionnement littérature

En split aléatoire, Dong et al. 2024 rapportent macro-AUC ~0.966 (T2, multilabel).
Pour T1a spatial, le mur non-graphe (XGB) est à 0.588 et les GNN seuls à ~0.605 (phase
2). Un gain hybride **réel** exige > 0.63 AUC spatial + significativité + > 0.03 bruit
inter-pli. Les chiffres ci-dessus (smoke) ne permettent pas de conclure : les 500 puits
et 15 époques ne représentent pas les vraies capacités du pipeline. **Les conclusions
définitives attendent le run Colab GPU.**

Fichiers : `src/hybrid.py`, `experiments/gnn_hybrid_t1/run_hybrid_t1.py`,
`tests/test_hybrid_smoke.py`.

---

## Colab notebook + run estimate (colab-notebook-engineer)

### Smoke-test result (CPU, 2026-06-21)

Two passes run back-to-back on CPU (`PFAS_FORCE_CPU=1`):

**Pass 1 — `tests/test_hybrid_smoke.py` (400 wells, outer_k=3, inner_k=2, 10 epochs, hidden=16)**

```
[guard 1] assert_no_group_leak: outer + all inner folds — PASS
[guard 3] test-embed GNN: removed=0 remaining=0 — PASS
[guard 3] inner-OOF GNN: removed=0 remaining=0 — PASS
[guard 2] test_b=1 not in fit_block_ids=[0, 2] — PASS
[guard 4] fit_blocks [1] ∩ embed_blocks [0] = ∅ — PASS
[guard 4] overlap correctly rejected: fit_blocks and embed_blocks overlap: [0] — PASS
[guard 5] embedding shape=(183, 16) embed_dim=16 finite=True — PASS
[guard 6] fused features (217, 108) → XGB fits (proba finite) — PASS
[guard 7] all §4.3 metrics compute: [roc_auc, pr_auc, recall, precision, f1,
          accuracy, balanced_accuracy, brier, ece, gain_top20pct, ...] — PASS
[guard 8] threshold=0.070 (from OOF, not test) — PASS
[guard 9] checkpointing: metrics_incremental.json written — PASS
[end-to-end] spatial_AUC=0.7019  random_AUC=0.7011  Δ=-0.0007
[end-to-end] 3 outer folds, all §4.3 metrics present — PASS
ALL GUARDS GREEN in 82.2s — Smoke test PASSED.
```

**Pass 2 — `experiments/gnn_hybrid_t1/run_hybrid_t1.py` (SMOKE_TEST=True, 500 wells,
2 relations × 2 arms × 3 outer folds = 12 GNN trainings)**

```
[subbasin_knn] hybrid spatial AUC=0.5635  random AUC=0.7176  Δ=+0.1542
[spatial]      hybrid spatial AUC=0.5568  random AUC=0.7255  Δ=+0.1687
THREE-WAY COMPARISON (subbasin_knn, smoke — non représentatif)
  hybrid        spatial=0.5635  random=0.7176  delta=+0.1542
  gnn_alone     spatial=0.6050  random=n/a     delta=n/a
  xgb_alone     spatial=0.5880  random=n/a     delta=n/a
  Hybrid gain over XGB wall: -0.0245
  Verdict: spurious (not significant) [expected in smoke]
Total elapsed: 147.7s
```

Both passes green. The smoke AUC figures (non-representative at 500 wells / 15 epochs)
match previously reported smoke output — pipeline is stable.

### Realistic GPU duration estimate (Colab T4)

The REPORT §5 figure "~40 h" was computed with a CPU extrapolation of 15 min/GNN. The
measured GPU pace from phase 2 refutes this:

- **Phase 2 measured**: 2 models × 2 regimes × 8 folds = 32 GNN trainings (400 epochs,
  full 11 k-node graph) completed in **33.4 min** on Colab T4 → **~1 min/GNN**.
- **Phase 1 measured**: same code, "seconds per fold on GPU, full run < 2 min" (from
  phase 1 REPORT, confirmed by the 21-min CPU run and the known ~10–20× GPU speedup).

Applying 1–2 min/GNN to the hybrid count:

| Config | GNN count | Estimate (T4 GPU) |
|---|---|---|
| **Full sweep** (2 relations, K=8, J=4) | 160 | **2.7–5.3 h** |
| **Trimmed** (1 relation `subbasin_knn`, K=8, J=4) | 80 | **1.3–2.7 h** |
| **Trimmed+** (1 relation, K=8, J=2) | 48 | **0.8–1.6 h** |

The prior 15 min/GNN estimate was a CPU-only extrapolation ignoring the GPU. The honest
GPU range is **2.7–5.3 h for the full sweep** (T4; A100 would halve it). Both fit within
one Colab session (12 h limit). The trimmed first-run config (`RELATIONS=["subbasin_knn"]`)
is recommended for a first result, resumable via incremental checkpoints.

If the actual fold time exceeds 3 min/GNN (e.g. larger hidden or more neighbours), the
checkpoint design means reconnecting and re-running Cell 5 picks up from the last completed
fold.

### Notebook — confirmed autonomous (no Drive)

`notebooks/gnn_hybrid_t1_colab.ipynb` (generated by `gnn_hybrid_t1_colab.ipynb.py`):

- **Cell 0**: user parameters (`REPO_URL`, `GIT_REF`, `DATA_PATH`, `SMOKE_TEST`,
  `RELATIONS`, full-run knobs). Prints GNN count + estimated wall time before running.
- **Cell 1**: GPU detection (`torch.cuda.is_available()`, device name), Python/torch/CUDA
  versions.
- **Cell 2**: `git clone REPO_URL` → brings `src/` AND versioned `data/CA-PFAS-ASGWS.parquet`
  to Colab. Anti-stale-code guard asserts `run_hybrid_t1`, `train_gnn_and_embed`,
  `build_well_graph` symbols exist in the cloned source. **No Drive, no gdown.**
- **Cell 3**: PyTorch Geometric installed for the runtime's exact torch+CUDA wheel;
  `SAGEConv`/`GCNConv`/`GraphConv` import verified; XGBoost installed if missing.
- **Cell 4**: dataset loaded from `data/CA-PFAS-ASGWS.parquet`; shape check (46338 × ≥201),
  key columns (`gm_well_id`, `latitude`, `longitude`, `PFOA_ngL`) verified; hard stop with
  explicit error if mismatch. Skipped in SMOKE_TEST mode.
- **Cell 5**: runs `H.run_hybrid_t1()` for each relation; checkpoints written per fold to
  `experiments/gnn_hybrid_t1/run_<rel>/{spatial,random}/metrics_incremental.json`.
- **Cell 6**: three-way comparison table; reality-rule verdict; writes
  `experiments/gnn_hybrid_t1/metrics.json`.
- **Cell 7**: persistence — `files.download()` of zip archive AND optional `git push`
  shell commands (commented, with instructions). Explicit WARNING that outputs are lost
  on disconnect without this step.

### How to launch on Colab

1. Open `notebooks/gnn_hybrid_t1_colab.ipynb` in Google Colab.
2. Runtime > Change runtime type > **GPU** (T4 is sufficient; A100 halves wall time).
3. In Cell 0: set `SMOKE_TEST=True` for a quick sanity run first, then
   `SMOKE_TEST=False` for the full run. Set `RELATIONS=["subbasin_knn"]` for a trimmed
   first session (~1.3–2.7 h); add `"spatial"` for the full sweep (~2.7–5.3 h).
4. Runtime > Run all. No manual intervention needed.
5. After completion, run Cell 7 to download the archive or push to the repo.

Parameters users may need to adjust: `REPO_URL` (if the repo is forked), `GIT_REF`
(branch/commit), `RELATIONS` (subset for a trimmed run), full-run knobs
(`FULL_OUTER_K`, `FULL_INNER_K`, `FULL_GNN_EPOCHS`, `FULL_HIDDEN`).

Fichiers livrés : `notebooks/gnn_hybrid_t1_colab.ipynb`,
`notebooks/gnn_hybrid_t1_colab.ipynb.py`.

---

## Gate-5 evaluation (eval-methodologist protocol, run on the real Colab metrics)

> Run réel : `experiments/gnn_hybrid_t1/metrics.json` — relation `subbasin_knn`, K=8 LOBO
> spatial, J=4 OOF interne, 400 époques, hidden 64, graine 42, ~68 min GPU. L'agent
> `eval-methodologist` ayant atteint sa limite de session, l'audit a été complété au fil
> principal en appliquant SON protocole pré-enregistré (§4.5), par recalcul CPU bon marché
> (`/tmp/gate5_paired.py`, ~71 s, aucun entraînement GNN local) réutilisant les helpers
> `_corrected_resampled_ttest` / `_wilcoxon_paired` du socle.

### 1. Headline AUC — moyenne par pli vs OOF poolée
- **ROC-AUC spatial moyenne par pli = 0,646 ± 0,112** ; **OOF poolée = 0,691** [IC95 % bootstrap
  par groupe 0,674–0,707]. L'écart vient de la **forte hétérogénéité inter-pli** : l'OOF poolée
  agrège les 46 338 prédictions et est tirée par les gros plis bien discriminés ; elle n'est pas
  pénalisée par la variance. **Headline honnête = la moyenne par pli (0,646 ± 0,112)** comme
  chiffre de généralisation spatiale ; l'OOF (0,691) est rapportée en complément, et son IC est
  trop étroit (il ne reflète pas la variabilité inter-région).

### 2. Instabilité par pli (confirmée, comme en phase 1)
Per-fold hybride : [0,535 · 0,740 · 0,504 · 0,694 · 0,743 · 0,764 · 0,711 · **0,476**]. Trois
plis à/sous le hasard (0 : 0,535 ; 2 : 0,504 ; 7 : 0,476). Le pli 2 est le **plus gros**
(18 450 lignes) et le pire pour les TROIS modèles (hybride 0,504, XGB-core 0,432, GNN 0,577) →
**région intrinsèquement difficile**, pas seulement sous-entraînement. La moyenne est portée par
les plis 1/4/5. Toute revendication doit intégrer cette variance.

### 3. Comparaison APPARIÉE corrigée (le triplet `metrics.json` était cassé — réparé)
Le `three_way_comparison` du run était **non concluant par construction** : `paired_*` vides,
bras `gnn_alone`/`xgb_alone` = scalaires codés en dur (0,605 / 0,588), jamais rejoués par pli.
De plus, le mur publié 0,588 utilise un **jeu de features PLUS LARGE** (`cocontam="all"`,
+air) que le bloc tabulaire de l'hybride (`core`, 61) → ce n'est pas une ablation propre de
l'embedding. Bras manquant reconstruit : **XGB-alone sur les MÊMES features `core`**, niveau
ligne, mêmes 8 blocs KMeans (graine 42, ordre trié → plis alignés).

| arme | per-fold spatial AUC (mean ± std) | note |
|---|---|---|
| **Hybride (GNN⊕XGB, subbasin_knn)** | **0,646 ± 0,112** | OOF poolée 0,691 |
| GNN seul (phase 2 P0, GraphSAGE stabilisé) | 0,615 ± 0,063 | mêmes plis |
| **XGB seul (features `core`)** | **0,596 ± 0,090** | ablation propre de l'embedding |
| (mur publié XGB, features `all`+air) | 0,588 ± 0,068 | référence historique, features ≠ |

**Tests appariés (8 plis alignés) :**
- **Hybride vs XGB-seul(core)** : Δ = **+0,050**, gagne 6/8 plis. **Wilcoxon p=0,039 (significatif)**
  MAIS **Nadeau-Bengio corrigé t=2,09 p=0,076 (NON significatif)**. → **BORDERLINE** : passe le
  test des rangs mais échoue au test t corrigé (le plus rigoureux pour la CV, il corrige la
  corrélation train/test). > seuil de bruit 0,03 ✓.
- **Hybride vs GNN-seul(P0)** : Δ = **+0,031**, 6/8 plis. NB p=0,60, Wilcoxon p=0,38 → **DANS LE
  BRUIT** (l'ajout de la fusion XGB sur le GNN n'apporte rien de significatif).

### 4. Calibration & Δ
- Calibration médiocre hors distribution : **ECE 0,124, Brier 0,231** (spatial) — recalibration
  OOF recommandée avant toute décision opérationnelle.
- **Δ(random−spatial) = 0,202** (random 0,848 − spatial 0,646) : profil **sain**, dans la bande
  GNN-seul (0,196–0,218), bien sous les arbres (~0,30). Le modèle mémorise moins la carte.
- **vs Dong et al. 2024** : leur chiffre est en **split aléatoire** → comparable au mieux à notre
  **random 0,848**, PAS au spatial. Écart de protocole à signaler ; ne pas comparer spatial↔Dong.

### 5. VERDICT
**Gain net NON ROBUSTE — signal prometteur mais non concluant.** L'hybride dépasse le mur
tabulaire (mêmes features) de **+0,050** AUC spatiale, ce qui franchit le seuil de bruit 0,03 et
est significatif au Wilcoxon (p=0,039) **mais pas au Nadeau-Bengio corrigé (p=0,076)** — sur 8
plis à très forte variance, l'étude est **sous-puissante**. L'hybride ne bat PAS significativement
le GNN seul (+0,031, n.s.). Conforme à l'avertissement pré-enregistré du §2.4 : sans `facility_id`
ni gradient, l'apport mécanistique au-dessus du k-NN spatial reste **modeste**. **Ne pas
revendiquer une victoire nette.** Pour trancher : (a) répéter sur **≥5 graines / blocs alternatifs
(LeaveOneRegionOut)** pour la puissance ; (b) le bras `relation="spatial"` (non encore lancé) pour
isoler l'apport *mécanistique* propre ; (c) SHAP XGB-seul (cf. plausibilité hydro) pour certifier
que le signal de base est physique et non un confondeur de design.

---

## Gate-5 plausibility (hydro-domain-expert)

**VERDICT : PLAUSIBLE, non sur-vendu — mais NON CERTIFIÉ tant que le SHAP du XGB-seul n'a pas
confirmé les drivers physiques.** Statut opérationnel : *plausible / needs-SHAP.*

### 1. Mécanisme d'arête — défendable ; la contrainte sous-bassin ajoute de la vraie physique
« Même sous-bassin SGMA ET ≤2 km » est un proxy acceptable d'exposition PFAS partagée et **est
strictement meilleur que la proximité brute** : le cap 2 km capte des puits susceptibles de
puiser la même unité aquifère superficielle / même recharge / même couloir de panache (cohérent
avec Moran's I 0,426, portée 2–5 km). **Refuser de relier deux puits de part et d'autre d'une
limite de sous-bassin est physiquement correct** (crête piézométrique, faille, changement de
remplissage) — c'est un vrai prior d'aquifère, pas un ré-encodage de la carte. Limites (toutes
dues à des données manquantes) : hétérogénéité verticale / aquifères multicouches (`well_depth_ft`
94,5 % manquant → graphe aveugle à la profondeur, faiblesse n°1) ; lentilles perchées ; cônes de
pompage ; 13 % de puits sans sous-bassin laissés isolés (conservateur ; vérifier la perf par
sous-groupe si non aléatoires).

### 2. Magnitude — un petit gain mécanistique au-dessus du k-NN spatial est ATTENDU
Les sources ici (AFFF aéroports/militaire, chromage, raffineries, terminaux) sont **ponctuelles**
à panaches **locaux** (km). Un voisinage ≤2 km capte déjà l'essentiel ; la contrainte sous-bassin
ne fait qu'élaguer les arêtes franchissant une crête → **un petit gain est exactement ce que
prédit la physique ; un gros gain serait suspect** (réinjection d'information longue portée). Le
Δ≈0,20 (plus sain que les arbres ~0,30) tient physiquement (C4 force une structure de voisinage
généralisable). **« Le graphe ré-encode surtout la proximité courte portée » est la lecture
honnête — endossée.** Ne pas sur-revendiquer : gain non significatif et variance inter-pli forte.

### 3. Designs refusés — accord sur des bases physiques ; un levier manqué
- Cliques par **TYPE de source** (sans `facility_id`) : refus **justifié** (deux puits près de
  deux aéroports à 500 km ne partagent ni panache ni aquifère ; η=0,081). *Nuance* : le type de
  source reste valide comme **feature de nœud** (prior AFFF) — vérifier qu'il est dans l'espace
  tabulaire ; c'est l'usage en arête qui est faux, pas la variable.
- Arcs **dirigés amont→aval** (sans gradient) : refus **justifié** ; ne pas prétendre à une
  connectivité hydrologique dirigée.
- **Levier manqué (travaux futurs, pas un défaut)** : un **gradient topographique (MNT)** comme
  proxy du gradient hydraulique (orienter les arêtes haut→bas en vallée alluviale non confinée),
  et une **anisotropie le long de l'axe de vallée** (cap plus lâche parallèle à l'écoulement).
  Spéculatif → re-tester le Δ (la topo peut devenir un proxy de carte). Seul levier physique
  restant pour passer d'un graphe « co-localisé » non orienté à « connecté par l'écoulement ».

### 4. SHAP — ce qu'il faut pour CERTIFIER (et oui, ça vaut un suivi ciblé)
Aucun SHAP généré ; **sans attribution je ne peux pas certifier la plausibilité, seulement
l'absence de signal d'alerte.** Priorité : (1) **SHAP du XGB-seul (tabulaire), d'abord et à part**
— le moins cher (pas de GPU), le plus informatif. Drivers plausibles : `dist_geotracker_km` /
`n_geotracker_within_{1,3}km`, `nearest_geotracker_type=Airport` (prior AFFF), co-contaminants
industriels, texture/MO du sol (`soil_om_pct`, `soil_ksat`), `well_depth_ft`. **Drapeau rouge** :
si `gm_dataset_name` (WB_CLEANUP cible des sites déjà pollués) ou un co-contaminant domine comme
proxy « échantillon analysé en labo » → confondeur de design à neutraliser (zone de vigilance du
profilage). (2) **Répartition du gain embedding vs tabulaire** (|SHAP| agrégé sur les colonnes
d'embedding vs tabulaires) : l'embedding contribue-t-il, ou XGB le traite-t-il en bruit
(cohérent avec le désalignement d'axes documenté) ? Si négligeable → le petit gain n'est PAS un
signal d'aquifère, le dire. **NE PAS rejouer le sweep GNN** ; l'attribution tourne sur les modèles
déjà entraînés.

| Axe | Verdict |
|---|---|
| Mécanisme d'arête (sous-bassin ≤2 km) | Plausible ; contrainte = vrai prior d'aquifère |
| Magnitude (petit gain + Δ sain) | Physiquement attendu ; ne pas sur-revendiquer |
| Designs refusés | Refus justifiés ; levier manqué = prior d'écoulement régional (MNT) |
| Limites du proxy | Hétérogénéité verticale (profondeur manquante), perchés, 13 % isolés |
| Certification | NON certifié tant que SHAP XGB-seul n'a pas écarté le confondeur `gm_dataset_name`/co-contaminant |

---

## Gate-5 SHAP certification (XGB-seul, features `core`) — suivi exécuté

> `experiments/gnn_hybrid_t1/shap_xgb_core.py` (CPU, 23 s, graine 42, TreeSHAP sur 6 000
> lignes). Artefacts : `shap_xgb_core.json` + `shap_xgb_core_top.png`. Le drapeau rouge
> demandé par l'hydro-expert (§4) est levé. **STATUT MIS À JOUR : CERTIFIÉ, avec 2 réserves.**

**Top drivers (mean|SHAP|, signe = corr valeur↔SHAP) :**
`n_geotracker_within_50km` 0,52 (+) · `dwr_region=South Coast` 0,29 (+) · **`dist_geotracker_km`
0,23 (−)** · `gldas_dist_km` 0,20 (+) · **`gm_well_category=MONITORING` 0,20 (+)** · `dwr_basin__enc`
0,18 (−) · `n_geotracker_within_10km` 0,17 (+) · `root_zone_moist` 0,11 (+) · **`year` 0,10 (+)** ·
`cocontam_as` 0,10 (−) · soil_sand_fine, soil_ph, soil_awc, aqs_* …

**Part par famille :** proximité-source **25,0 %** · admin/géo+well-category (« other ») 23,8 % ·
sol **20,4 %** · climat/hydro 9,8 % · qualité air 9,4 % · **co-contaminants 6,9 %** · temporel
3,1 % · puits 1,5 %.

**Lecture de certification :**
- ✅ **Le signal de base EST physique.** La proximité-source est la 1ʳᵉ famille (25 %) et porte
  les signaux les plus mécanistes : `dist_geotracker_km` (négatif = plus proche → plus de PFAS,
  direction correcte) et la densité de sites à 10 km. + sol (rétention/mobilité) 20 % + transport
  hydrique 10 %. C'est l'exposition AFFF/industrielle attendue.
- ✅ **Le confondeur redouté NE se déclenche PAS.** Co-contaminants seulement 6,9 %, menés par
  l'**arsenic (géogénique, négatif)** — pas un proxy « échantillon analysé en labo ». `gm_dataset_name`
  absent (exclu C6). Le drapeau rouge de l'hydro-expert est levé.
- ⚠️ **Réserve 1 — sélection de site** : `gm_well_category=MONITORING` (rang 5, +) — les puits de
  surveillance sont implantés là où une contamination est suspectée. Cousin atténué du confondeur
  de design (sélection, pas labo) ; à divulguer, idéalement contrôler par strate de catégorie.
- ⚠️ **Réserve 2 — dérive temporelle** : `year` (rang 9, +) confirme la dérive d'échantillonnage
  signalée au profilage ; le modèle s'y appuie un peu. Non mécaniste ; à divulguer.
- Note : poids notable des features géographiques larges (`dwr_region=South Coast`, `_within_50km`,
  `gldas_dist_km`) → une part du « signal » est géographie régionale/urbanisation — cohérent avec
  l'AUC spatiale modeste (une bonne part de la performance random vient de la carte).
- (in-sample AUC 0,979 = sur-ajustement du modèle global pour l'attribution UNIQUEMENT ; le
  vrai chiffre de généralisation reste l'AUC spatiale 0,596 du XGB-seul.)

**Reste à faire (différé, hors scope CPU) :** la **répartition du gain embedding vs tabulaire**
(point 2 de l'hydro-expert) nécessite les modèles hybrides entraînés — non sauvegardés par le run
Colab (seuls les `metrics_incremental.json` le sont). À produire en sauvegardant les modèles XGB
hybrides lors d'un prochain run, ou via un ré-entraînement local des bras hybrides à embeddings figés.

**VERDICT hydro mis à jour : PLAUSIBLE et CERTIFIÉ** (signal de base dominé par l'exposition
physique, confondeur labo écarté), **avec 2 réserves documentées** (sélection des puits de
surveillance, dérive temporelle) et le split embedding/tabulaire encore à faire.
