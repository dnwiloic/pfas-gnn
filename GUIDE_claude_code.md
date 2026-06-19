# Guide — Claude Code pour la recherche PFAS (v2)

Dépose `CLAUDE.md` + `.claude/` à la racine de ton dépôt. Claude Code charge `CLAUDE.md`
à chaque session et découvre les 8 sous-agents (`/agents` pour vérifier).

Cette version respecte 4 exigences :
1. exploration **exhaustive** des variantes GNN ;
2. livrables = **notebooks Colab** (pas de GPU local) ;
3. **mini-tests** (smoke-tests) avant chaque run long ;
4. **aucune conclusion préalable** sur le jeu de données — les agents l'analysent
   eux-mêmes et justifient leurs choix.

---

## 1. Les 8 agents

| Agent | Rôle | Modèle |
|---|---|---|
| `data-analyst` | Profile le jeu de données DE ZÉRO ; identifie cibles, fuites, blocs spatiaux | opus |
| `hydro-domain-expert` | Plausibilité mécaniste, interprétation SHAP, arêtes de graphe | opus |
| `tabular-ml-engineer` | Baselines RF/XGBoost (le mur à battre) | sonnet |
| `gnn-researcher` | Exploration exhaustive des GNN ; gardien anti-fuite spatiale | opus |
| `multilabel-specialist` | Baseline multilabel/semi-supervisée (T2) | opus |
| `eval-methodologist` | Gardien de l'évaluation ; peut REFUSER un protocole | opus |
| `colab-notebook-engineer` | Notebooks Colab + smoke-tests + checkpoints | sonnet |
| `scientific-writer` | Rédaction, positionnement, zéro chiffre inventé | sonnet |

Les agents ne se parlent pas entre eux : ils rapportent au fil principal. Chaînage =
chacun écrit `experiments/<id>/REPORT.md` que le suivant relit. Tu orchestres.

**Deux gardiens à utiliser à chaque cycle :** `eval-methodologist` (avant/après un run,
rigueur), `colab-notebook-engineer` (avant toute exécution Colab, smoke-test).

---

## 2. Flux recommandé (toujours commencer par l'analyse)

```
Étape 0  data-analyst         → profilage, cibles, audit fuite, blocs spatiaux
Étape 1  hydro-domain-expert  → critique mécaniste de l'espace de features
Étape 2  tabular-ml-engineer  → baseline non-graphe (mur à battre)
         multilabel-specialist→ baseline multilabel (T2)
Étape 3  gnn-researcher       → exploration GNN, comparée aux baselines
chaque run :  eval-methodologist (valide AVANT) → colab-notebook-engineer
              (smoke-test + notebook) → run Colab → eval-methodologist (contrôle APRÈS)
Étape 4  scientific-writer    → rédaction à partir des REPORT.md
```

---

## 3. Prompts d'orchestration (à coller dans le fil principal)

### A. Analyse autonome du jeu de données (à faire en premier)
```
Délègue à data-analyst : analyse le jeu de données dans <chemin> SANS aucune hypothèse
préalable. Profilage complet, identification des colonnes-cibles et des colonnes qui
fuitent la cible (avec preuves), proposition de définition(s) de cible pour T1 et T2,
caractérisation de l'autocorrélation spatiale et proposition de blocs géographiques.
Écris experiments/profilage/REPORT.md. Puis demande à hydro-domain-expert de critiquer
l'espace de features proposé sur le plan mécaniste.
```

### B. Baselines (le mur à battre)
```
Sur la base de experiments/profilage/REPORT.md :
1) tabular-ml-engineer : baseline T1 (RF + XGBoost) avec validation spatiale par blocs
   EN PLUS du split aléatoire ; rapporte l'écart.
2) multilabel-specialist : baseline T2 (compare classifieurs indépendants vs chaînes).
Code dans src/, smoke-testé, puis notebook Colab. Chacun écrit son REPORT.md.
```

### C. Boucle d'expérience fiabilisée (réutilisable pour CHAQUE variante)
```
Nouvelle expérience : <méthode>.
1) eval-methodologist valide le protocole (fuite cible + fuite spatiale) AVANT.
2) l'agent compétent implémente la logique dans src/.
3) colab-notebook-engineer écrit/lance le smoke-test sur CPU (<3 min) et n'emballe le
   notebook Colab QUE si le smoke-test est vert ; le notebook est AUTONOME (bootstrap du
   code + téléchargement du dataset depuis Drive) ; il estime la durée du run complet.
4) [tu lances le notebook sur Colab GPU]
5) eval-methodologist contrôle les résultats ; hydro-domain-expert juge la plausibilité.
Ne passe à l'étape suivante que si la précédente est VALIDÉE.
```

### D. Rédaction
```
scientific-writer : lis tous les experiments/*/REPORT.md et rédige "Résultats et
positionnement", en comparant aux chiffres réellement obtenus et à Dong et al. (2024).
N'invente aucun chiffre.
```

---

## 4. Bibliothèque de prompts — Exploration GNN exhaustive

> Chaque prompt : partir de l'analyse, valider sous **CV spatiale par blocs**, justifier
> les arêtes mécaniquement, comparer au mur des baselines, passer par smoke-test +
> notebook Colab. On ne suppose AUCUN modèle préexistant.

### 4.1 Socle commun
**GNN-0 — Graphe & audit de fuite spatiale**
```
gnn-researcher : à partir de experiments/profilage/REPORT.md, construis un premier
graphe de puits (k-NN spatial) et un GraphSAGE inductif. Évalue en split aléatoire ET
CV spatiale par blocs ; rapporte l'écart comme mesure de fuite spatiale. eval-method.
valide avant, colab-notebook-engineer smoke-teste avant le run.
```

### 4.2 Tâche 1 — balayage complet
**G1.1 — Ablation des types d'arêtes** (le cœur scientifique)
```
gnn-researcher : compare sur le MÊME modèle et la MÊME CV spatiale 4 graphes :
(a) k-NN spatial, (b) connectivité hydrologique amont→aval (MNT), (c) source commune,
(d) k-NN dans l'espace des features. Quelles arêtes apportent un gain qui SURVIT à la CV
spatiale ? hydro-domain-expert juge la plausibilité de chaque graphe.
```
**G1.2 — Balayage exhaustif d'architectures**
```
gnn-researcher : sur le meilleur graphe de G1.1, balaie TOUT le catalogue de ton fichier
d'agent — GCN, GraphSAGE, GAT/GATv2, GIN, SGC, APPNP, TAGCN, ARMA, PNA, GPS, et en
hétérogène R-GCN, R-GAT, HAN, HGT, HEAT. Même protocole/métriques. Tableau récapitulatif
+ calibration + gain cumulé @k.
```
**G1.3 — Ablation de conception**
```
gnn-researcher : sur les 2-3 meilleures archis, ablate profondeur (sauts), agrégation,
normalisation, dropout nœuds/arêtes, résiduels, neighbor sampling, inductif vs
transductif. Documente l'effet de chaque dimension.
```
**G1.4 — Hybride GNN ⊕ arbres**
```
gnn-researcher + tabular-ml-engineer : embeddings du meilleur GNN concaténés aux features,
entraîne un XGBoost. Compare GNN seul / arbres seuls / hybride. Le signal relationnel
survit-il une fois le contexte déjà capté par les arbres ?
```

### 4.3 Tâche 2 — balayage complet
**G2.1 — Graphe de corrélation de labels (ML-GCN)**
```
gnn-researcher : les labels PFAS comme NŒUDS, arêtes = co-occurrence conditionnelle
(mesurée par data-analyst). Compare aux chaînes de classifieurs (multilabel-specialist),
macro-AUROC et AUROC par sous-groupe, surtout les labels rares.
```
**G2.2 — Graphe bipartite puits–PFAS = complétion de matrice** (piste forte)
```
gnn-researcher : modélise la matrice de mesures lacunaire en graphe bipartite
puits–PFAS ; traite la prédiction comme une prédiction de liens / complétion (GAE, VGAE,
ou complétion inductive type IGMC). Reformulation absente de Dong et al. : compare
frontalement à la baseline multilabel et discute sa valeur en mode prédictif strict.
```
**G2.3 — Semi-supervision sur graphe à effectifs décroissants**
```
gnn-researcher : remplace le pseudo-étiquetage par seuil par une propagation de labels
sur le graphe de puits. Compare les deux à volume d'étiquettes décroissant (grand →
moyen → petit) pour préfigurer un régime à données rares. eval-method. contrôle la fuite.
```
**G2.4 — Encodeur GNN partagé + têtes par label/sous-groupe**
```
gnn-researcher : un encodeur GNN partagé alimente des têtes de classification par
label/sous-groupe. Compare aux baselines multilabel ; surveille le transfert vers les
labels rares.
```

---

## 5. Bonnes pratiques

- **Toujours analyser avant de modéliser** (`data-analyst` d'abord). Aucun agent ne
  recopie une conclusion : il l'établit.
- **Logique dans `src/`, orchestration dans le notebook** : le même code est
  smoke-testé sur CPU puis exécuté en entier sur Colab GPU.
- **Notebooks AUTONOMES** : chaque notebook est lancé seul sur Colab. Ses cellules de
  tête, dans l'ordre, détectent le GPU, installent les dépendances, **amènent le code**
  (`git clone` ou copie de `src/` depuis Drive) et **téléchargent le dataset depuis
  Drive** (cellule paramétrée `DRIVE_DATA_PATH` ou `DRIVE_FILE_ID`, avec instruction
  claire). Aucun fichier n'est supposé « déjà présent ».
- **Smoke-test obligatoire** : tu ne lances jamais 4 h sans un test CPU < 3 min vert.
- **Checkpointing sur Drive** : une déconnexion Colab ne doit pas coûter le run.
- **`eval-methodologist` à chaque cycle** : c'est lui qui protège la crédibilité (fuite,
  inflation spatiale).
- **Tout dans `experiments/<id>/`** : c'est le canal entre agents et la base de la
  rédaction.
- **Versionne `.claude/` dans git** : les agents s'améliorent avec le projet.
