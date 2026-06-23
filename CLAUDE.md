# CLAUDE.md — Mémoire projet (PFAS / eaux souterraines)

> Chargé automatiquement à chaque session et hérité par TOUS les sous-agents.
> Fixe les principes de travail. Il ne contient **volontairement aucune conclusion
> sur le jeu de données** : c'est aux agents de l'analyser et de justifier leurs choix.

## 1. Objet de la recherche

Prédire les PFAS en eaux souterraines en **mode prédictif strict** : prédire un
dépassement de seuil à partir de variables de contexte, **sans utiliser de
concentration PFAS mesurée (ni aucune colonne qui en dérive) comme entrée**. Deux
types de tâche :

- **T1 — dépassement réglementaire (binaire)** : le puits dépasse-t-il un seuil ?
- **T2 — PFAS individuels (multilabel)** : lesquels dépassent leur seuil ?

Objectif transverse : un pipeline **rigoureux, reproductible, exécutable sur Google
Colab**, qui explore **systématiquement l'espace des architectures GNN**.

## 2. Principe directeur : PARTIR DE ZÉRO SUR LES DONNÉES

Aucune connaissance préalable sur le jeu de données n'est fournie ni présumée.
Chaque agent doit, à partir de sa propre expertise :

- **Analyser le jeu de données lui-même** avant toute décision : structure, types,
  dimensions, distributions, taux de valeurs manquantes, structure spatiale,
  candidats-cibles, corrélations.
- **Ne RIEN supposer** : ni les noms de colonnes, ni les dimensions, ni la définition
  des cibles, ni quelles colonnes fuient, ni l'existence de regroupements/paliers, ni
  quelles sources de features existent, ni quel modèle aurait déjà été entraîné.
- **Justifier chaque choix** (cible, features, seuils, découpage, architecture) par
  l'analyse et l'expertise, en le documentant.

Si un agent a besoin d'un fait sur les données, il l'établit par analyse — il ne le
récite pas.

## 3. Principes méthodologiques (et non conclusions)

1. **Vigilance fuite — cible.** En mode prédictif strict, aucune mesure PFAS ni
   colonne dérivée de la cible ne doit être une feature. **C'est à l'agent
   d'identifier ces colonnes** (par analyse : corrélation, sémantique des noms,
   dérivation logique) et de justifier la liste d'exclusion.
2. **Vigilance fuite — spatiale.** Les puits proches sont autocorrélés ; un découpage
   aléatoire peut gonfler les scores. Toujours rapporter une **validation croisée
   spatiale par blocs** à côté du découpage aléatoire, et l'écart entre les deux.
   Décider après analyse s'il faut retirer les variables de localisation pure.
3. **Déséquilibre** traité explicitement et justifié. Optimisation de seuil uniquement
   sur probabilités out-of-fold (jamais sur le test).
4. **Métriques orientées décision** (l'agent choisit les pertinentes et le justifie) :
   p. ex. ROC-AUC, rappel, balanced accuracy, gain cumulé, calibration pour T1 ;
   macro-AUROC, micro-F1, Hamming, EMR, métriques par label/sous-groupe pour T2.
5. **Interprétabilité** (SHAP ou équivalent) + contrôle de plausibilité mécaniste.
   Une importance non interprétable est un signal d'alerte (fuite/artefact possible).
6. **Reproductibilité.** Graine fixée partout. Chaque expérience écrit dans
   `experiments/<id>/` : `config.yaml`, `metrics.json`, `REPORT.md`.
7. **Positionnement** vs la littérature, dont Dong et al. (2024), sans présumer de
   chiffres : l'agent rédacteur compare aux résultats réellement obtenus.
8. **Diagnostic d'entraînement — COURBES SYSTÉMATIQUES.** Tout entraînement itératif
   (époques GNN/réseaux, tours de boosting XGBoost, arbres RF) DOIT **journaliser puis
   afficher/sauvegarder les courbes de PERTE et de MÉTRIQUE (accuracy/AUC) au fil des
   itérations, en TRAIN ET VALIDATION**, par pli. On ne rapporte JAMAIS un score final
   sans avoir regardé la dynamique : ces courbes servent à **diagnostiquer** convergence,
   sur-/sous-apprentissage, point d'arrêt précoce, instabilité inter-pli, et l'agent en
   **tire des conclusions explicites dans `REPORT.md`** (p. ex. « pli k sous-entraîné,
   early-stop à l'époque 9 » — exactement le défaut corrigé en P0). Historiques sauvegardés
   (p. ex. `history.json`) et figures dans `experiments/<id>/figures/`
   (`{loss,metric}_curves*.png`). Pas de courbe possible (modèle non itératif) → le justifier.

## 4. Environnement d'exécution : Google Colab (pas de GPU local)

- **Tout entraînement lourd tourne sur Colab GPU.** Les livrables de modélisation sont
  des **notebooks `.ipynb` prêts pour Colab**.
- **Architecture du code** : la logique (chargement, features, graphe, modèle,
  entraînement, métriques) vit dans des **modules `src/` importables et testables sur
  CPU** ; le notebook ne fait qu'**orchestrer** (config, GPU, run complet, sauvegarde).
  Ainsi le MÊME code est mini-testé localement puis exécuté en entier sur Colab.
- **Politique de stockage : ZÉRO Google Drive.** Ni pour le code, ni pour le dataset, ni
  pour les sorties. Le dataset est **versionné dans le dépôt** (`data/CA-PFAS-ASGWS.parquet`)
  et le code dans `src/` : un `git clone` ramène les deux. Toutes les sorties s'écrivent
  **directement dans l'espace de travail** (le dépôt cloné sur Colab), dans
  `experiments/<id>/`.
- **Chaque notebook est AUTONOME.** Il sera ouvert et lancé SEUL sur Colab, sans dépôt
  local ni état partagé. Il doit donc, dans ses premières cellules et sans intervention
  manuelle au-delà du choix du runtime (et **aucune autorisation Drive**) :
  1. **détecter/activer le GPU** et imprimer les versions (Python, torch, CUDA) ;
  2. **installer les dépendances** épinglées et adaptées au runtime (PyTorch Geometric
     notamment, vérifier l'import) ;
  3. **récupérer le code ET le dataset depuis le dépôt** : comme `src/` et `data/` ne sont
     pas présents sur Colab, le notebook les amène lui-même par **`git clone` du dépôt**
     (variable `REPO_URL` + `GIT_REF` commit/branche). Le dataset versionné arrive AVEC le
     clone — **pas de Drive, pas de `gdown`** ;
  4. **charger le dataset depuis le dépôt cloné** : lire `data/CA-PFAS-ASGWS.parquet`
     (chemin relatif paramétré par `DATA_PATH`) et **vérifier l'intégrité** (forme/colonnes
     attendues) ; sinon arrêt avec message d'erreur explicite ;
  5. exposer le toggle `SMOKE_TEST` en tête.
- Cellules **idempotentes** (réexécutables sans casse) ; artefacts, modèles et
  **checkpoints** écrits **dans l'espace de travail** sous `experiments/<id>/`.
- ⚠️ L'espace de travail Colab est **éphémère** (perdu à la déconnexion). Sans Drive, le
  notebook doit, en fin de run, **proposer une persistance explicite** des sorties
  (`files.download()` d'une archive `experiments/<id>/`, et/ou `git add/commit/push`), en
  avertissant que sinon elles sont perdues à la fermeture du runtime.
- Un notebook ne doit jamais supposer qu'un fichier (code ou données) est « déjà là » :
  s'il manque, il le récupère par le clone ou s'arrête avec un message d'erreur explicite.

## 5. Discipline de MINI-TEST (obligatoire avant tout run long)

Aucun notebook destiné à un run long n'est livré sans avoir passé un **smoke-test**.

- Tout pipeline expose un mode `SMOKE_TEST` : sous-échantillon minuscule (~quelques
  centaines de lignes), 1 pli de CV, très peu d'époques/estimateurs, plus petit modèle,
  **exécutable sur CPU en < ~3 min**.
- Le smoke-test **vérifie le bout-en-bout** : les données se chargent, les formes
  concordent, le graphe se construit (bon nombre de nœuds/arêtes), la passe avant
  s'exécute, la perte est finie (et idéalement décroît), les métriques se calculent,
  les artefacts s'écrivent. Il vérifie aussi que les **courbes d'entraînement** (cf.
  §3.8) sont bien produites : historiques perte/métrique train+val **non vides** et de
  longueur = nombre d'époques/tours, figures écrites.
- Il **estime la durée du run complet** (extrapolation) et alerte si elle est longue.
- Règle : *smoke-test vert sur CPU → seulement ensuite run complet sur Colab GPU*.
  On ne découvre jamais une erreur de forme après 4 h de calcul.
- **Checkpointing** systématique pour les runs longs (sauvegarde incrémentale par
  époque/pli dans l'espace de travail `experiments/<id>/`, p. ex. `metrics_incremental.json`)
  afin qu'une déconnexion Colab ne perde pas le travail déjà fait ; persistance hors-session
  par download/commit en fin de run (cf. §4, zéro Drive).

## 6. Exploration GNN : exhaustive et systématique

L'agent GNN doit parcourir **tout l'espace des variantes** (catalogue dans son
fichier d'agent), pas un échantillon : familles convolutives, attentionnelles,
hétérogènes, transformers de graphe, graphes de labels, complétion de matrice
bipartite, semi-supervision sur graphe — avec ablation des dimensions de conception
(construction d'arêtes, profondeur, agrégation, normalisation, échantillonnage,
inductif vs transductif, hybride GNN⊕arbres). Chaque variante est comparée sous le
même protocole d'évaluation (§3) et au mur des baselines non-graphe.

## 7. Conventions de dépôt & sous-agents

- `src/` (modules testables CPU) · `notebooks/` (Colab) · `experiments/<id>/` · `tests/`.
- Les sous-agents **ne communiquent pas entre eux** : ils rapportent au fil principal.
  Chaînage = chacun écrit un `REPORT.md` que le suivant relit ; le fil principal
  orchestre.
- Faire **valider tout nouveau protocole par `eval-methodologist`** avant un run long,
  et faire **smoke-tester par `colab-notebook-engineer`** avant toute exécution Colab.
