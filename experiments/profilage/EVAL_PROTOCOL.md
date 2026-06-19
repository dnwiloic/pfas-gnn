# EVAL_PROTOCOL — Audit méthodologique du contrat d'évaluation (PFAS / CA, mode prédictif strict)

> Étape 3. Agent `eval-methodologist` (gardien méthodologique). J'AUDITE et je VALIDE
> ou REFUSE le contrat proposé par les étapes 1-2 (`REPORT.md`, `HYDRO_CRITIQUE.md`).
> **Je ne présume aucune blocklist ni aucun découpage : je vérifie par moi-même.**
> Tous les chiffres ci-dessous sont reproductibles (graine 42) :
> - `experiments/profilage/eval_audit.py` → `eval_audit_metrics.json`
> - vérification de portée d'autocorrélation : recalcul exhaustif par paires (cf. §2.3).
>
> **VERDICT : VALIDÉ SOUS CONDITIONS.** Le contrat est globalement solide ; 6 conditions
> bloquantes (C1-C6) doivent être levées avant tout run long. Détail au §7.

---

## 0. Synthèse exécutive (les 5 chiffres qui pilotent le verdict)

| Constat | Chiffre établi | Conséquence |
|---|---|---|
| **Inflation spatiale** (apport central) | AUC random-par-puits **0,727** → AUC spatial LOBO k=8 **0,557** → **Δ = 0,170** | Un split aléatoire surévalue massivement. La CV spatiale est **obligatoire** et devient la métrique de référence. |
| **Fuite pseudo-réplicats** | AUC row-KFold 0,746 → GroupKFold puits 0,727 → **Δ = 0,020** (>> bruit inter-plis 0,004-0,010) | Grouper par `gm_well_id` est **nécessaire et démontré**, pas une précaution de principe. |
| **Garde-fou détection** | **521 positifs T1a (2,5 %)** déclenchés sur un `>seuil` d'analyte **non détecté** (limite de rapport) | Garde-fou **à appliquer** ; prévalence T1a 45,6 % → **44,5 %**. |
| **Blocklist cible** | Après retrait des 96 col., **aucune** colonne résiduelle n'est une dérivation de la cible (top = proxys géo/mécaniste) | Blocklist **complète** sur le critère fuite-cible. |
| **Portée d'autocorrélation** | lift de concordance +0,236 (0-0,5 km) → +0,035 (2-5 km) → ~0 (>5 km) | Portée effective **~2-5 km** ; graphe k-NN à plafonner ~1-2 km ; buffer pertinent. |

---

## 1. AUDIT FUITE — CIBLE

### 1.1 La blocklist (96 col.) est-elle complète ? — OUI, vérifié empiriquement

J'ai recalculé, **indépendamment**, l'association de **chacune des 103 colonnes candidates
restantes** (après retrait de la blocklist proposée) avec les cibles **T1a (garde-fou
appliqué)** et **T1b** — corr. de Pearson pour le numérique, **η (rapport de corrélation)**
pour le catégoriel (le profilage ne testait que le numérique : **je comble ce trou**).

**Top associations résiduelles T1a** (rien au-dessus de 0,51, et aucune dérivation logique) :

| Colonne | assoc. | nature | Verdict |
|---|---|---|---|
| `dwr_basin` / `sgma_subbasin_name` | 0,505 (η) | **proxy géographique** (aquifère) | Pas une fuite-cible ; **confondeur spatial** (voir §2 + C5) |
| `county` | 0,403 (η) | proxy géo/administratif | confondeur, pas fuite |
| `well_depth_ft` | −0,347 | **mécaniste** (profondeur) | Légitime |
| `dwr_region` / `regional_board` | 0,33 (η) | proxy géo/admin | confondeur |
| `n_geotracker_within_50km` | 0,256 | proxy urbanité régionale | confondeur (cf. hydro) |
| `gm_dataset_name` | 0,252 (η) | **confondeur de design** (WB_CLEANUP) | À neutraliser (C6) |

**Conclusion fuite-cible** : **aucune colonne résiduelle n'est dérivée de la cible**. Les
plus fortes associations sont des **proxys géographiques** (qui causent l'inflation
spatiale, §2) et un **confondeur de design** (`gm_dataset_name`) — ce sont des problèmes
de *confondeur*, traités par la CV spatiale et l'audit SHAP, **pas** des fuites de cible.
La **blocklist de 96 colonnes est VALIDÉE et complète** sur ce critère.

**Note de cohérence (non bloquante)** : `dist_geotracker_km` et les comptes géotracker, les
`cocontam_*` et le sol restent admissibles côté fuite-cible. Le contrôle SHAP « importance
qui s'effondre en CV spatiale = artefact » du `HYDRO_CRITIQUE` §7 reste exigé en aval.

### 1.2 Garde-fou détection — IMPACT CHIFFRÉ, à appliquer

Le `*_ngL` stocke la **limite de rapport** quand l'analyte n'est pas détecté (établi au
profilage §2). Le MCL PFOA/PFOS = 4 ng/L est **du même ordre** que des LD observées →
risque de **faux positifs de cible**. J'ai chiffré l'impact (`detection_guardrail`) :

- **T1a sans garde-fou** : prévalence 0,4565 (21 154 pos).
- **T1a avec garde-fou** (`(X>seuil) & X_detected`) : prévalence **0,4453** (20 633 pos).
- **521 positifs (2,46 % des positifs bruts)** ne sont déclenchés **que** par un `>seuil`
  sur un analyte **non détecté** → ce sont des **artefacts de censure**.
- Décomposition : PFOA>4 mais non détecté = **515** lignes ; PFOS>4 mais non détecté =
  **532** ; PFNA>10 non détecté = 278 ; PFHxS>10 non détecté = 112. (GenX, PFBS : 0.)

**Décision** : le **garde-fou détection est OBLIGATOIRE** (C1). Définition retenue, à figer
comme contrat de cible :

```
T1a = ((PFOA_ngL > 4)  & PFOA_detected)
   OR ((PFOS_ngL > 4)  & PFOS_detected)
   OR ( HI_guarded >= 1 )
HI_guarded = (PFHxS·[PFHxS_detected])/10 + (PFNA·[PFNA_detected])/10
           + (HFPO_DA·[HFPO_DA_detected])/10 + (PFBS·[PFBS_detected])/2000
```

Pour T1b (Σ>70), le garde-fou est moins critique (somme de 31 analytes, dominée par des
détections réelles) mais doit être **documenté** comme limite : la somme inclut des LD de
non-détects. Garder T1b en **cible secondaire** (étiqueter « repère historique », cf. hydro
§5.3).

---

## 2. AUDIT FUITE — SPATIALE & GROUPES

### 2.1 Grouper par `gm_well_id` — NÉCESSITÉ DÉMONTRÉE (pas un principe)

J'ai mesuré la fuite par **pseudo-réplicats temporels** : même baseline
(LogisticRegression, 86 features numériques, lat/lon retirées), mêmes 5 plis :

- **KFold aléatoire au niveau ligne** (un puits peut être des deux côtés) : AUC = **0,746 ±0,004**
- **GroupKFold par `gm_well_id`** (split aléatoire mais puits indivisible) : AUC = **0,727 ±0,010**
- **Δ = +0,020**, soit **2 à 5× le bruit inter-plis**. 84,9 % des lignes appartiennent à
  des puits multi-prélèvements → la fuite est structurelle.

⇒ **Tout split, à TOUS les niveaux (CV externe, CV interne, tuning, seuil), groupe par
`gm_well_id`.** Validé et désormais quantifié. **(C2)**

### 2.2 CV spatiale par blocs — k=8 KMeans DÉFENDABLE, mais le LOBO seul est insuffisant

Inflation spatiale mesurée (apport central du projet), baseline identique :

| Protocole | AUC moyenne | Δ vs random-par-puits |
|---|---|---|
| Random GroupKFold (puits) | 0,727 | — (référence non-spatiale) |
| **Spatial Leave-One-Block-Out, KMeans k=8** | **0,557 ±0,032** (min 0,515 / max 0,598) | **−0,170** |
| Spatial LOBO, KMeans k=5 | 0,580 ±0,032 | −0,147 |

**Interprétation** : ~**0,17 d'AUC** de la baseline non-graphe est de la **structure
spatiale mémorisée**, pas du mécanisme transférable. C'est exactement le chiffre que le
projet doit publier comme contribution méthodologique. **La métrique de référence pour
TOUTE comparaison de modèles est la CV SPATIALE**, jamais l'aléatoire.

**Dégénérescence par bloc** : aucun bloc dégénéré. k=8 → prévalence par bloc **0,21-0,58**,
n_positifs **≥ 606** par bloc-test. k=5 idem. Les deux schémas sont **utilisables**.

**Limite de k=8 / k=5 pour la comparaison statistique** : un LOBO ne fournit que **8 (ou 5)
points de mesure** → un test apparié sur 5-8 plis est **sous-puissant** et instable
(l'AUC par bloc varie 0,51-0,60). **Recommandation (C3)** : utiliser un schéma à **plus de
plis** pour la *comparaison* des modèles tout en restant spatial — voir §2.4.

### 2.3 Contamination inter-blocs & buffer — quantifiée, KMeans déjà propre

Portée d'autocorrélation **recalculée exhaustivement** (toutes paires < distance, via
KDTree — et non sur 129 paires comme le profilage, base trop fragile) :

| Bande | lift de concordance T1 vs base (0,554) |
|---|---|
| 0-0,5 km | **+0,236** |
| 0,5-1 km | +0,135 |
| 1-2 km | +0,084 |
| 2-5 km | +0,035 |
| 5-10 km | +0,008 |
| 10-20 km | +0,003 |

⇒ Portée effective **~2-5 km** ; l'essentiel est < 2 km. **Fuite de frontière potentielle**
si des puits proches tombent dans des blocs différents. J'ai mesuré, pour KMeans k=8
(niveau puits), la part de puits dont le plus proche voisin est dans un **autre** bloc :

| Buffer | puits avec voisin < d | dont **cross-block** | frac. cross-block |
|---|---|---|---|
| 0,5 km | 8 255 | **0** | 0,0000 |
| 1 km | 10 038 | **0** | 0,0000 |
| 2 km | 10 812 | **1** | 0,0001 |
| 5 km | 11 139 | 6 | 0,0005 |

**KMeans produit des blocs spatialement compacts** → la contamination de frontière est
**négligeable** (0 puits à <1 km de part et d'autre). Un buffer formel n'est donc **pas
strictement nécessaire avec KMeans**. **MAIS** :

- **(C4)** Le buffer redevient **obligatoire dès que le graphe GNN relie des puits par
  k-NN spatial** : une arête qui franchit la frontière du bloc fait **transiter
  l'information train→test** (message passing). Règle à imposer : **toute arête spatiale
  > seuil de distance OU traversant la frontière de bloc CV est coupée à l'évaluation**,
  et le k-NN spatial est **plafonné à ~1-2 km** (cohérent avec la portée). Sans cette
  règle, la CV spatiale est **invalidée pour les GNN** (le buffer du dataset ne suffit pas
  si le graphe recrée le pont).
- **(C5)** Les arêtes / features « même `dwr_basin` / `sgma_subbasin` » entrent en
  **collision directe** avec la CV par blocs : ces colonnes ont η≈0,51 avec la cible et un
  bloc KMeans peut couper un bassin en deux. Si on relie par bassin (ou si on garde le
  bassin en feature catégorielle one-hot/OOF), **le schéma de blocs CV doit être ⊇ bassins**
  (LeaveOneRegionOut) **OU** le bassin doit être retiré des arêtes. Documenter le choix.

### 2.4 Schéma de blocs recommandé (synthèse)

- **Référence principale (rapportée partout)** : **CV spatiale par blocs**. Pour gagner en
  puissance statistique tout en restant spatial, utiliser un **Spatial-K-Fold** :
  générer **~20-40 micro-blocs** (KMeans k=20-40 ou grille spatiale ~25-50 km au niveau
  puits), puis **assembler en K=5 plis** en regroupant des micro-blocs **non adjacents**
  (assignation par bloc, jamais par ligne). On obtient **5 plis spatiaux comparables** →
  test apparié valide, tout en gardant la séparation spatiale. Vérifier qu'aucun pli n'a
  de prévalence dégénérée (toutes entre ~0,2 et 0,6, déjà le cas).
- **Variantes de robustesse rapportées** : LeaveOneBlockOut k=8 (interprétable, 8 régions),
  et **LeaveOneRegionOut** par `regional_board` (9) ou `dwr_region` (10) — utile pour la
  collision bassin (C5) et pour une lecture « généralisation à une nouvelle région ».
- **Toujours** grouper par `gm_well_id` à l'intérieur de chaque bloc (C2).

---

## 3. OPTIMISATION DE SEUIL & CV IMBRIQUÉE SANS FUITE

### 3.1 Principes non négociables

1. **Seuil de décision optimisé UNIQUEMENT sur probabilités out-of-fold** (concaténation
   des prédictions de validation interne), **jamais sur le test/le bloc externe**. Le seuil
   est un **hyperparamètre** : il se choisit dans la boucle interne.
2. **Imputation, scaling, encodage catégoriel haute cardinalité (target/OOF encoding)** :
   **fittés sur le train du pli uniquement**, appliqués au test. Le target-encoding doit
   lui-même être **out-of-fold à l'intérieur du train** (sinon fuite de cible via
   l'encodage). C'est le point le plus souvent raté : `dwr_basin` (239), `sgma_subbasin`
   (237), `county` (58) ont une cardinalité élevée et une η forte → un target-encoding
   naïf fuite massivement.
3. **Sélection d'hyperparamètres** : sur la métrique en **CV interne SPATIALE-groupée**
   (mêmes contraintes que l'externe), pas sur le test.
4. **Tout split (externe, interne) est à la fois GROUPÉ (`gm_well_id`) ET SPATIAL (blocs).**
5. **Une seule passe** sur le pli de test externe, à la toute fin, avec le seuil et les
   hyperparamètres figés par la boucle interne.

### 3.2 Pseudo-code de la CV imbriquée (cf. §6 pour la version finale au fil principal)

Voir §6.

---

## 4. MÉTRIQUES

### 4.1 T1 (binaire) — jeu validé, orienté décision

| Métrique | Rôle | Rapportée en |
|---|---|---|
| **ROC-AUC** | discrimination, comparable littérature | random **ET** spatiale |
| **PR-AUC (AP)** | adaptée au (dés)équilibre — utile surtout pour T1b (prév. 25 %) ; T1a quasi-équilibrée (44,5 %) | random ET spatiale |
| **Rappel @ seuil** + **précision/PPV** | décision (manquer un dépassement = coûteux) | spatiale (réf.) |
| **Balanced accuracy** | équilibre des classes | spatiale |
| **Brier score** + **ECE** + **courbe de fiabilité** | **calibration** — une AUC non calibrée est inexploitable en décision | **spatiale obligatoire** ; recalibration (Platt/Isotonic) **fittée OOF** |
| **Gain cumulé / lift @ k%** | priorisation opérationnelle (quels puits tester d'abord) | spatiale |

- **Calibration sous CV spatiale** : c'est là qu'elle se dégrade le plus (le modèle est
  sur-confiant hors distribution spatiale). Rapporter Brier/ECE **en spatial** est non
  négociable.
- **Baseline de référence** obligatoire : prévalence/stratifiée + LogisticRegression
  non-graphe (le « mur des baselines » de CLAUDE.md §6). Tout GNN se compare à ce mur **en
  CV spatiale**.

### 4.2 T2 (multilabel, 15 labels) — jeu validé

- **macro-AUROC** (moyenne sur labels, traite labels rares à égalité) et **macro-AP**
  (PR-AUC, plus honnête pour les labels rares : PFDA 8 %, NMeFOSAA 9 %).
- **micro-F1** et **macro-F1** (les deux : micro domine par les labels fréquents, macro
  expose les rares).
- **Hamming loss**, **Exact Match Ratio (EMR)** (sévère, ~25 % de lignes à 0 label).
- **Par-label** : AUROC + AP **par analyte**, et **par sous-groupe spatial** (bloc) pour
  détecter une généralisation hétérogène.
- **Seuils par label** optimisés **OOF indépendamment** (un seuil global est sous-optimal
  vu l'hétérogénéité de prévalence 8-49 %).

### 4.3 Rapporter l'ÉCART random↔spatial = test d'artefact (exigence CLAUDE.md §3.2)

Pour **chaque modèle** et **chaque métrique principale**, rapporter le **triplet** :
`(score_random, score_spatial, Δ = random − spatial)`. **Δ grand = le modèle exploite la
structure de carte** (artefact spatial), Δ petit = signal transférable. La baseline donne
**Δ_AUC ≈ 0,17** : tout modèle dont le Δ est **plus grand** que la baseline est **plus
artefactuel**, pas meilleur — même si son AUC random est plus haute. **Le classement des
modèles se fait sur le score SPATIAL**, et le Δ est reporté comme diagnostic d'honnêteté.

---

## 5. COMPARAISON STATISTIQUE DES MODÈLES

1. **Plis identiques** : tous les modèles (baselines, variantes GNN) évalués sur **les
   mêmes plis spatiaux** (mêmes graines, même assignation bloc→pli). Indispensable pour
   l'appariement.
2. **Test apparié sur les plis** : pour K=5 plis spatiaux (cf. §2.4), utiliser le
   **corrected resampled t-test** (Nadeau-Bengio, qui corrige la corrélation train/test
   inhérente à la CV) **et/ou** le **Wilcoxon signé** sur les paires de scores par pli.
   **Ne pas** sur-interpréter un t-test naïf à 5 points.
3. **Intervalles de confiance bootstrap** : sur les prédictions **out-of-fold concaténées**
   du test spatial, bootstrap **par groupe (`gm_well_id`, voire par bloc)** — pas par ligne
   (sinon IC trop étroits par pseudo-réplicats). Rapporter IC95 % de l'AUC/AP.
4. **Seuil de signification pratique** : un gain GNN vs baseline est **réel** seulement si
   (a) significatif au test apparié, **et** (b) supérieur au **bruit inter-plis** (std des
   AUC par bloc, ~0,03 en spatial ici). Un gain < 0,03 d'AUC spatiale est **dans le bruit**
   → ne pas le revendiquer.
5. **Positionnement littérature** (Dong et al. 2024 et al.) : comparer **score spatial**
   au score publié en notant le protocole de la littérature (souvent random → non
   comparable directement ; signaler l'écart de protocole).

---

## 6. PSEUDO-CODE — CV IMBRIQUÉE, GROUPÉE-SPATIALE, SANS FUITE

```
ENTRÉES : df (lignes = puits×prélèvement), y (cible T1a garde-fou), 
          groups = gm_well_id, coords = (lat,lon) niveau puits
PRÉ : construire blocs spatiaux AU NIVEAU PUITS (jamais ligne) :
      micro_blocks = KMeans(k=20..40).fit_predict(coords_well)   # graine fixée
      outer_folds  = assembler micro_blocks en K=5 plis (micro-blocs non adjacents)
      propager bloc→ligne via gm_well_id   # un puits = un seul bloc = un seul pli

POUR chaque pli externe f dans 1..K (CV SPATIALE EXTERNE) :
    test_idx  = lignes des blocs assignés à f
    train_idx = lignes des blocs ≠ f
    # train_idx et test_idx ne partagent AUCUN gm_well_id (par construction) NI bloc spatial

    # ---- BOUCLE INTERNE (sélection HP + seuil), SUR LE TRAIN SEULEMENT ----
    micro_in   = KMeans/grille sur les PUITS du train          # blocs internes spatiaux
    inner_folds = assembler en J=4 plis (groupés gm_well_id + spatiaux)
    POUR chaque combinaison d'hyperparamètres h :
        oof_proba = vecteur vide aligné sur train
        POUR chaque pli interne j :
            tr_j, val_j = split interne (groupé+spatial)
            # tout le PRÉTRAITEMENT est fitté sur tr_j uniquement :
            imputer      = fit(tr_j)            # médiane/par-bassin + indicateurs de manque
            scaler       = fit(tr_j)
            cat_encoder  = fit_OOF(tr_j)        # target/OOF-encoding interne au tr_j (anti-fuite)
            # GNN : graphe construit sur tr_j ∪ val_j MAIS arêtes spatiales
            #       PLAFONNÉES À ~1–2 km ET COUPÉES si elles franchissent la
            #       frontière tr_j/val_j (sinon message passing = fuite) [C4]
            modèle_h = entraîner(h) sur tr_j (transductif : masque val_j en perte)
            oof_proba[val_j] = prédire(val_j)
        score_interne[h] = métrique(y[train], oof_proba)         # ex. AP ou AUC spatiale
    h* = argmax score_interne
    # seuil de décision choisi UNIQUEMENT sur oof_proba de h* (jamais sur test) :
    τ* = argmax_τ  métrique_décision(y[train], oof_proba_h* ≥ τ)
    # recalibration (Platt/Isotonic) AUSSI fittée sur oof_proba de h* :
    calibrateur = fit(oof_proba_h*, y[train])

    # ---- RÉ-ENTRAÎNEMENT FINAL sur tout le train du pli, h* figé ----
    prétraitements = fit(train_idx)            # imputer/scaler/cat_encoder OOF sur train
    # graphe externe : mêmes règles d'arêtes (plafond distance + coupe frontière bloc) [C4]
    modèle_final = entraîner(h*) sur train_idx
    proba_test   = calibrateur( prédire(test_idx) )
    # UNE SEULE évaluation du test externe, seuil et HP figés :
    enregistrer  ROC-AUC, PR-AUC, Brier, ECE, rappel/précision@τ*, gain cumulé
                 EN VERSION SPATIALE (= ce pli) ; idem en version random pour le Δ

AGRÉGER sur les K plis : moyenne ± écart-type ; IC bootstrap PAR GROUPE/bloc.
COMPARER modèles : mêmes plis → corrected resampled t-test (Nadeau-Bengio)
                   + Wilcoxon signé ; gain réel ssi significatif ET > bruit inter-plis.
RAPPORTER pour chaque modèle : (score_random, score_spatial, Δ).
```

**Points anti-fuite incarnés dans le pseudo-code** (à vérifier à l'implémentation) :
- aucun `gm_well_id` partagé entre train/test (groupes) — **et** aucun bloc spatial partagé ;
- imputation / scaling / encodage catégoriel **fittés intra-fold** (et target-encoding OOF) ;
- seuil **et** recalibration **fittés sur OOF du train**, jamais sur le test ;
- arêtes GNN spatiales **plafonnées en distance** et **coupées aux frontières de bloc** [C4] ;
- une **seule** évaluation du pli de test externe.

---

## 7. VERDICT : VALIDÉ SOUS CONDITIONS

Le contrat partagé est **méthodologiquement solide** : granularité, blocklist cible (96),
définition T1a EPA-2024, structure spatiale et schéma de blocs sont **corrects et
empiriquement confirmés**. La fuite-cible est propre, la fuite spatiale est **quantifiée**
(Δ=0,17, contribution majeure). **Aucun motif de REFUS.** Mais 6 conditions bloquantes
doivent être levées **avant tout run long** :

| # | Condition bloquante | Statut |
|---|---|---|
| **C1** | **Appliquer le garde-fou détection** à T1a (`(X>seuil) & X_detected`, HI sur analytes détectés). Prévalence cible 44,5 %. Documenter T1b comme secondaire/censuré. | À FAIRE |
| **C2** | **Grouper TOUS les splits par `gm_well_id`** (externe, interne, tuning, seuil). Démontré : Δ=0,020 > bruit. | À FAIRE |
| **C3** | **CV spatiale = métrique de référence**, avec un schéma à **≥5 plis spatiaux comparables** (micro-blocs assemblés) pour permettre le test apparié ; LOBO k=8 et LeaveOneRegionOut en robustesse. Le score random ne sert qu'au calcul du Δ. | À FAIRE |
| **C4** | **GNN** : k-NN spatial **plafonné à ~1-2 km** (portée mesurée 2-5 km) **ET arêtes coupées aux frontières de bloc CV** (sinon message passing = fuite spatiale, invalide la CV). | À FAIRE |
| **C5** | **Résoudre la collision `dwr_basin`/`sgma_subbasin` (η≈0,51) × CV par blocs** : soit blocs ⊇ bassins (LeaveOneRegionOut), soit retirer le bassin des arêtes/features de localisation. Documenter. | À FAIRE |
| **C6** | **Neutraliser `gm_dataset_name`** (confondeur de design, η 0,25-0,36) : exclu des features ; gardé pour stratification/audit. Idem lat/lon en feature de nœud (tester avec/sans, cf. hydro). | À FAIRE |

**Conditions de qualité (fortement recommandées, non strictement bloquantes)** :
- Target/OOF-encoding catégoriel **interne au train** (anti-fuite via l'encodage).
- Recalibration (Brier/ECE) et seuil **fittés OOF**, rapportés **en spatial**.
- Bootstrap des IC **par groupe/bloc**, pas par ligne.
- Pour chaque modèle, rapporter le **triplet (random, spatial, Δ)** comme test d'artefact.
- Audit SHAP croisé random↔spatial (toute importance qui s'effondre en spatial = artefact).

**Une fois C1-C6 intégrées au pipeline `src/` et smoke-testées (CLAUDE.md §5), le protocole
passe de VALIDÉ SOUS CONDITIONS à VALIDÉ.** Tant que C1, C2, C4 ne sont pas en place, **tout
run long est REFUSÉ** (fuite de cible par censure, fuite par pseudo-réplicats, ou fuite
spatiale par les arêtes GNN — chacune invaliderait les scores).

---

### Artefacts de cet audit
- `experiments/profilage/eval_audit.py` — script déterministe (graine 42).
- `experiments/profilage/eval_audit_metrics.json` — tous les chiffres ci-dessus.
- Portée d'autocorrélation recalculée exhaustivement (§2.3), corrige l'échantillon de 129
  paires du profilage par un calcul sur 10⁴-10⁶ paires (KDTree).
