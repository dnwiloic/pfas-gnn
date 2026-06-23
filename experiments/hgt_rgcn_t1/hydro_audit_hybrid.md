# Audit mécaniste du notebook `04_hgt_xgboost_hybrid_epa2024`

Auteur : hydro-expert (hydrogéochimie PFAS). Date : 2026-06-23.
Périmètre : cohérence scientifique et méthodologique du notebook hybride HGT⊕XGBoost
(cible EPA 2024, CA-PFAS-ASGWS, mode prédictif strict).
Sources confrontées au CODE (pas seulement au PDF) :
`ca-pfas-ml/notebooks/04_hgt_xgboost_hybrid_epa2024.ipynb` (lignes citées),
mon rapport pfas-gnn `hydro_expert.md`, profilage spatial pfas-gnn.

> Vérification faite sur le code source, pas seulement le PDF. Les faits structurants
> ci-dessous sont confirmés par les lignes du notebook (split l.335-338 ; features
> géo l.282 ; arêtes identité l.567 ; geo_cluster l.579-584 ; `sample.x` = toutes les
> features l.565 ; KMeans fitté sur train l.556-558).

---

## Synthèse des verdicts

| Élément | Verdict |
|---|---|
| Graphe hétérogène à 5 types de nœuds | **PROBLÉMATIQUE** — pour l'essentiel une reformulation de la couche d'entrée, pas un graphe mécaniste |
| Arêtes identité 1:1 (env/facility/water) | **PROBLÉMATIQUE** — réencodent les features sur elles-mêmes, aucune information relationnelle |
| Arêtes geo_cluster (KMeans lat/lon) | **DISCUTABLE→PROBLÉMATIQUE** — canal de fuite spatiale supplémentaire sous split aléatoire |
| Split aléatoire 60/20/20 | **PROBLÉMATIQUE** — AUC 0.95 non défendable, inflation attendue ~0.20 pt |
| lat/lon en features baseline | **PROBLÉMATIQUE** — proxy géographique, pas mécanisme |
| SHAP « structure relationnelle » | **DISCUTABLE** — les drivers sont géographiques, pas relationnels |
| Conclusion « combine structure + boosting » | **DISCUTABLE** — non soutenue par les SHAP |

---

## 1. Conception du graphe — cohérence mécaniste : PROBLÉMATIQUE

**Le nœud = une ligne d'échantillonnage, pas un puits.** Le code pose
`data['sample'].x = X_data.values` (l.565) : chaque ligne du dataset devient un nœud.
Pour modéliser le transport PFAS — un phénomène attaché à une position dans l'aquifère —
l'objet physique pertinent est le PUITS (point d'observation de la nappe), pas
l'événement d'échantillonnage. Si plusieurs lignes correspondent au même puits à des
dates différentes, elles deviennent des nœuds distincts sans arête entre elles : la
continuité temporelle/spatiale d'un même point de nappe est perdue. C'est l'inverse du
graphe puits-puits de pfas-gnn, qui relie des points de nappe distincts par proximité
hydrogéologique capée à 1,5 km.

**Les arêtes env/facility/water sont des arêtes identité strictes.** Le code construit
`ei = torch.stack([arange(n), arange(n)])` (l.567) : le sample i est relié au nœud env i,
facility i, water i — et ces nœuds portent des sous-ensembles des MÊMES colonnes déjà
présentes dans `sample.x` (l.565, l.574). Il n'y a aucune mise en relation entre
échantillons via ces types : c'est une copie de la couche d'entrée éclatée en 4
sous-vecteurs, puis recollée par message-passing. Ce n'est pas un graphe hétérogène au
sens mécaniste (aucune entité « facility » réelle partagée par plusieurs samples) ; c'est
un MLP par blocs déguisé en HGT. Aucune connectivité hydrogéologique n'est encodée.

**Les arêtes geo_cluster sont le SEUL vrai canal relationnel** — et c'est précisément le
plus dangereux. Le KMeans à 20 clusters sur (lat, lon) (l.556, l.579-583) connecte tous
les samples d'un même pavé géographique via un centroïde. Mécaniquement, un centroïde
géographique n'est PAS une unité hydrogéologique : il ne respecte ni les limites de
bassin, ni le sens d'écoulement, ni une source commune. Deux samples du même cluster
peuvent être de part et d'autre d'une divide, dans des aquifères distincts. L'agrégation
via le centroïde revient à un lissage de la cible par pavé géographique — exactement le
réencodage de la carte que la critique d'arête doit refuser (cf. pfas-gnn : la clique de
sous-bassin avait été rejetée pour ce motif, Cramér V(block,subbasin)=0,98).

**Comparaison avec le graphe puits-puits capé à 1,5 km :** l'approche pfas-gnn est
nettement plus défendable. Son arête `near` approxime une connectivité de panache réelle
(autocorrélation mesurée Moran I=0,43, concordance 77 % à 0-1 km, calibrée sur la portée
du signal et bornée par un cap dur + coupe inter-blocs). Le graphe du notebook 04 n'a, en
pratique, qu'une seule relation informative (geo_cluster) et elle est non capée, non
hydrogéologique, et opère sur tout un pavé KMeans — donc plus proche de la fuite que de la
connectivité.

---

## 2. Fuite spatiale — analyse critique : PROBLÉMATIQUE

**Le split est aléatoire stratifié** (l.335-338, `train_test_split(..., stratify=y)`),
sans aucune validation spatiale par blocs. Sur CE MÊME dataset, pfas-gnn a mesuré un Δ
d'inflation de ~0,17-0,22 pt AUC entre split aléatoire (~0,82-0,84) et split spatial
honnête (~0,60-0,65). Les AUC du notebook (XGBoost 0,952 ; Stacking 0,956) sont donc
**très vraisemblablement gonflées par autocorrélation spatiale** et ne mesurent pas la
capacité prédictive hors-zone.

**lat/lon en features.** Le groupe Geospatial (l.282) injecte directement les coordonnées,
et le SHAP les place 2e (latitude) et 3e (longitude). Ce n'est pas un signal mécaniste :
c'est l'apprentissage d'un gradient géographique de prévalence (où la contamination est
fréquente en Californie). Sous split aléatoire, un point test hérite de la cible de ses
voisins train via ses coordonnées — c'est de la mémorisation de carte, pas de la
prédiction de contexte.

**Double canal géographique.** lat/lon comme features ET geo_cluster comme arêtes
empilent deux fois la même information spatiale. Un sample test dont le cluster KMeans
contient beaucoup de samples train contaminés voisins récupère leur signal par
message-passing HGT, EN PLUS du gradient lat/lon. C'est un mécanisme de fuite spatiale
direct et identifiable, qui explique pourquoi HGT_emb_0 domine la fusion (cf. §3).

**Estimation corrigée.** En transposant le Δ mesuré dans pfas-gnn (~0,20), l'AUC
spatialement honnête attendue pour ce pipeline se situe autour de **0,74-0,78** au mieux
(XGBoost/Stacking), possiblement plus bas car le canal geo_cluster ajoute une voie de
fuite que pfas-gnn n'avait pas. Autrement dit : l'écart de 0,16-0,18 pt entre le notebook
04 (0,95) et le plancher honnête pfas-gnn (0,60-0,65) est presque entièrement de
l'artefact spatial.

---

## 3. SHAP — interprétation mécaniste : DISCUTABLE

**`n_geotracker_within_50km` (1er du SHAP XGBoost) n'est pas un signal de source.** À
50 km, ce n'est plus la proximité d'un panache (qui agit à ≲ 1-2 km) : c'est un proxy
d'urbanisation / densité industrielle régionale. Le contraste est parlant —
`n_geotracker_within_1km` est DERNIER du SHAP. Le modèle ne récompense pas la proximité
réelle d'une source (l'échelle mécaniste), il récompense « être dans une région dense ».
C'est un driver géographique, pas hydrogéochimique.

**lat/lon (2e, 3e) confirment le plafond géographique.** Le modèle apprend
majoritairement « où » (gradient Nord-Sud / Est-Ouest de prévalence) plutôt que
« pourquoi » (vulnérabilité de la nappe, transport depuis une source). C'est un plafond
mécaniste : ces variables ne portent aucun mécanisme transposable hors Californie.

**`HGT_emb_0` domine la fusion (48,4 % de l'importance aux embeddings).** Vu la
construction du graphe, le seul canal qui mélange l'information entre samples est
geo_cluster. La dimension HGT dominante encode donc très probablement une **coordonnée
de position dans l'espace des clusters géographiques** — c'est-à-dire un résumé spatial
de la cible par pavé. HGT_emb_0 est, mécaniquement, un super-proxy géographique appris.
Cela renforce le diagnostic de fuite : l'apport « graphe » est de l'autocorrélation
spatiale recomposée.

**`xgb_p1` domine le méta-classifieur à 59 %.** Verdict net : la valeur ajoutée de HGT
sur XGBoost est faible. Le stacking s'appuie majoritairement sur XGBoost ; le gain de
0,956 vs 0,952 (4 millièmes d'AUC) est dans le bruit. La « structure relationnelle » ne
fait pas le travail — le gradient boosting sur features (dont lat/lon et within_50km) le
fait.

---

## 4. Résultats et positionnement : PROBLÉMATIQUE (présentation actuelle)

**Présenter honnêtement les deux chiffres.** Dans un mémoire de Master, AUC 0,95-0,96 ne
doit JAMAIS être le chiffre titre. La présentation correcte est : « AUC 0,95 sous split
aléatoire, qui chute à ~0,74-0,78 estimé sous validation spatiale par blocs ; l'écart
mesure l'inflation par autocorrélation spatiale, quantifiée à ~0,20 pt sur ce dataset ».
Le chiffre spatial est le résultat scientifique ; le chiffre aléatoire est le diagnostic
de fuite.

**Comparaison avec Dong et al. (2024) (macro-AUC 0,966, split aléatoire).** Le notebook
04 est comparable à Dong UNIQUEMENT sur le terrain du split aléatoire — donc sur un
terrain biaisé pour les deux. La comparaison honnête exige de placer ce modèle en split
SPATIAL, où ni Dong ni le notebook 04 ne sont actuellement évalués. Annoncer « on
égale/approche Dong » à 0,95 est trompeur : c'est comparer deux chiffres gonflés.

**Mode prédictif ENV_ONLY (0,736) et PROXIMITY_ONLY (0,802).** Ces chiffres sont AUSSI en
split aléatoire. En split spatial, attendre une nouvelle chute : ENV_ONLY probablement
~0,60-0,65 (proche du mur RF honnête de pfas-gnn), PROXIMITY_ONLY ~0,65-0,70 (la
proximité régionale within_50km contient elle-même de la fuite géographique). Le fait que
PROXIMITY_ONLY (0,802) > ENV_ONLY (0,736) confirme d'ailleurs que c'est la GÉOGRAPHIE,
pas la chimie environnementale, qui porte le score.

**« Combine structure relationnelle et puissance du gradient boosting ».** DISCUTABLE et
non soutenu : les SHAP montrent (a) que le méta-classifieur s'appuie à 59 % sur XGBoost,
(b) que l'embedding HGT dominant encode de la position géographique, (c) que les arêtes
relationnelles « vraies » (env/facility/water) sont des identités sans contenu. Ce que le
modèle combine réellement, c'est « gradient boosting sur features géographiques » +
« lissage spatial de la cible ». La formulation correcte serait : « le gain du graphe est
marginal et attribuable à de l'autocorrélation spatiale, pas à un mécanisme relationnel ».

---

## 5. Pistes d'amélioration concrètes

### 5.1 Ce qui manque le plus (priorisé)
1. **Validation spatiale par blocs (priorité absolue).** Sans elle, aucun chiffre n'est
   interprétable. C'est le correctif n°1.
2. **Direction d'écoulement / gradient hydraulique.** Le seul ajout mécaniste réel
   (cf. mon rapport pfas-gnn §5.1) : transforme `dist_geotracker_km` isotrope en proxy de
   panache amont/aval, et permettrait une arête orientée `flows_to` vraiment relationnelle.
3. **Profondeur de nappe / épaisseur vadose / lithologie du chemin.** Vulnérabilité
   verticale réelle, absente ici.
4. **Typologie de sources PFAS élargie** (AFFF/DoD, STEP/biosolides, décharges) : les
   geotracker actuels ratent les sources PFAS dominantes.
5. **Downgrader `within_50km` et lat/lon** : urbanité/géographie, pas mécanisme.

### 5.2 Rendre le notebook comparable à pfas-gnn
- Remplacer `train_test_split` (l.335-338) par la CV spatiale k=8 KMeans géographiques
  (les blocs de pfas-gnn). Rapporter AUC aléatoire ET spatiale côte à côte + le Δ.
- Couper les arêtes inter-blocs pour geo_cluster (un cluster ne doit pas relier train et
  test du même bloc) — sinon la fuite persiste même sous split spatial.
- Refitter KMeans graphe à l'intérieur de chaque pli train (déjà fait sur train, l.556 —
  bon réflexe à conserver, à étendre aux plis).

### 5.3 Si lat/lon doivent rester (pour geo_cluster)
- Les SORTIR de la baseline et les traiter comme **ablation explicite** : modèle
  « sans géo » (mécaniste) vs « avec géo » (oracle géographique). Le delta mesure
  exactement la part de score qui n'est que de la position. Ne jamais présenter le modèle
  « avec lat/lon » comme le résultat prédictif principal.

### 5.4 Rendre les arêtes identité informatives
- Les arêtes env/facility/water actuelles (identité 1:1, l.567) sont à SUPPRIMER : elles
  n'apportent rien qu'un MLP ne ferait. Pour qu'un type de nœud soit relationnel, il faut
  une entité PARTAGÉE par plusieurs samples : p. ex. un vrai nœud `facility`
  (un site geotracker = un nœud, relié à tous les puits dans son rayon de panache capé) ou
  un nœud `aquifer/subbasin` (entité hydrogéologique partagée). Alors le message-passing
  agrège des puits qui partagent réellement une source ou un compartiment — ce qui serait
  mécaniste. Tant que chaque sample a son propre nœud env/facility/water unique, le graphe
  hétérogène est cosmétique.

### Note de parcimonie
Le notebook est techniquement soigné (SMOTE train-only, KMeans fit-on-train, Optuna), mais
la rigueur d'implémentation masque une faiblesse de conception : sous split aléatoire avec
lat/lon en features et un graphe dont la seule relation active est un pavé géographique, le
score élevé est principalement de l'autocorrélation spatiale recomposée. Le résultat
scientifique honnête n'est pas « AUC 0,95 », c'est « le signal mécaniste est faible
(~0,74-0,78 estimé en spatial), le reste est de la géographie ». C'est exactement la
conclusion déjà établie côté pfas-gnn, et ce notebook la confirme par contraste.
