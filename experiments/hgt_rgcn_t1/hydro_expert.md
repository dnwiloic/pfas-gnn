# Critique mécaniste des arêtes du graphe multi-relationnel HGT / R-GCN (T1)

Auteur : hydro-expert (hydrogéochimie PFAS). Date : 2026-06-22.
Périmètre : plausibilité physique des deux relations du graphe homogène puits-puits
utilisé pour HGT / R-GCN sur T1 (California PFAS, eaux souterraines).
Sources confrontées : `src/graph.py`, `experiments/hgt_rgcn_t1/eval_validation.md`,
`experiments/profilage/HYDRO_CRITIQUE.md`, `experiments/hgt_rgcn_t1/REPORT.md`,
`src/config.py`.

> Avertissement de lecture sur les chiffres : le `REPORT.md` du dossier
> `hgt_rgcn_t1/` correspond à un run `smoke=True` (sous-échantillon). Les AUC exactes
> (0.6464, etc.) y sont indicatives, PAS le résultat de production. Ma critique porte
> sur le DESIGN des arêtes et sur l'ordre de grandeur attendu, qui restent valides
> quel que soit le run final.

---

## Synthèse des verdicts

| Élément | Verdict mécaniste |
|---|---|
| Arête `near` (k-NN spatial, cap 1,5 km) | **PLAUSIBLE** comme proxy de panache/source partagée, à condition stricte du cap. C'est la borne haute défendable. |
| Arête `same_subbasin_knn` (intra-sous-bassin, cap 2 km) | **DISCUTABLE** : le prior aquifère est correct dans le principe (ne pas franchir une divide hydrogéologique), mais à 2 km la contrainte sous-bassin est presque toujours déjà satisfaite → l'arête est en grande partie REDONDANTE avec `near`. Apport marginal attendu faible. |
| Distinction `near` vs `same_subbasin_knn` | **DISCUTABLE** : les deux relations encodent essentiellement la même proximité ; la différence n'est informative que pour les rares paires 1,5-2 km à cheval sur une frontière de sous-bassin. |
| Encodage source→puits comme feature de nœud uniquement | **NON JUSTIFIÉ comme suffisant** : c'est une approximation isotrope d'un phénomène anisotrope (advection le long du gradient). Pas faux, mais incomplet — c'est la vraie limitation, pas l'absence de nœud source. |
| Métaphore graphe puits-puits | **DISCUTABLE pour le transport** : un graphe non orienté de voisinage capture la co-appartenance à un panache, pas la direction du transport. C'est la bonne métaphore pour l'autocorrélation, pas pour l'advection. |

---

## 1. Arête `near` — k-NN spatial, cap dur 1,5 km

**Verdict : PLAUSIBLE (sous condition du cap dur).**

### Ce que le message-passing agrège réellement à cette échelle
Le fait dirigeant établi par le profilage (`HYDRO_CRITIQUE.md` §4) est sans ambiguïté :
autocorrélation T1 réelle mais de **courte portée** — Moran's I = 0,43 ; concordance
77 % entre puits 0-1 km, qui retombe au niveau de base (~54 %) **dès la tranche 1-5 km**.
Mécaniquement, l'échelle où deux puits partagent vraiment un signal commun est celle
d'**un panache ou d'une source partagée**, soit ≲ 1-2 km.

Donc à 1 km, dans un même aquifère, deux puits qui captent des PFAS partagent
plausiblement :
- le même panache issu d'une source ponctuelle (AFFF, chromage, raffinerie), ou
- la même empreinte de source diffuse locale (lixiviat, épandage, recharge urbaine).

C'est un signal hydrogéologique défendable : l'arête approxime une **connectivité de
panache**, et le message-passing y agrège un contexte de contamination réellement
partagé.

### À 1,5 km : la limite haute, pas le réglage optimal
À 1,5 km, le mélange devient hétérogène. On agrège encore du panache commun pour les
gros sites AFFF (panaches kilométriques en aval), mais on commence à agréger des puits
**sans lien hydrologique direct** simplement co-localisés dans la même tache urbaine.
Comme la concordance est déjà retombée à la base sur 1-5 km, le contenu informatif
au-delà de ~1 km est faible et le risque de **proxy géographique pur** monte.

Le cap dur est donc le garde-fou central : sans lui, un k-NN dans une zone dense (Los
Angeles, vallée centrale) relierait des puits sur plusieurs km = réencodage de la carte
= fuite spatiale sous CV aléatoire. Le code applique bien ce cap haversine (vrai km) et
la coupe inter-blocs (`cut_cross_block`, assertion 0 résiduel par relation, C-SPAT.5).
C'est correct.

### 1,5 km est-il pertinent pour la Californie centrale ?
- **Plutôt la borne haute du raisonnable, pas trop court.** Au vu de la portée
  d'autocorrélation mesurée (chute dès 1 km), j'aurais préféré le défaut à **1,0 km**
  pour `near`, avec 1,5 km comme ablation. À 1,5 km on est déjà dans la zone où le
  signal est dilué.
- La Californie centrale (vallée de San Joaquin / Sacramento) est dominée par des
  aquifères alluviaux à forte densité de puits ; à cette densité, un cap de 1 km laisse
  largement assez de voisins pour un k=8. Le cap n'est donc pas limitant en densité.
- **Recommandation concrète** : tester `cap_km ∈ {0.5, 1.0, 1.5}` en ablation et
  regarder où l'AUC spatiale décroche. La portée mesurée prédit un plateau ≤ 1 km.

**Important — il n'existe pas, à ma connaissance et dans ce dataset, de mesure directe
de longueur de panache PFAS en aquifère californien.** Le cap n'est PAS calibré sur des
panaches observés ; il est calibré sur la **portée d'autocorrélation de la cible**
(Moran/concordance). C'est méthodologiquement honnête (on borne au range mesuré) mais il
faut le dire : 1,5 km est un choix anti-fuite, pas une longueur de panache validée.

---

## 2. Arête `same_subbasin_knn` — intra-sous-bassin SGMA, cap dur 2 km

**Verdict : DISCUTABLE.**

### Le sous-bassin SGMA est-il une unité hydrogéologique pertinente ?
Oui, en principe. Les sous-bassins DWR/SGMA sont délimités sur des critères
hydrogéologiques (limites de bassins alluviaux, barrières structurales) — pas purement
administratifs comme un county. L'idée « ne pas connecter deux puits 1 km de part et
d'autre d'une divide » est un **vrai prior aquifère** et c'est la bonne intuition. Le
code la matérialise correctement : k-NN spatial RESTREINT au même sous-bassin, et il
REFUSE explicitement la clique de sous-bassin complète (81,6 % des paires > 5 km,
Cramér V(block, subbasin)=0,981 = fuite spatiale déguisée). Ce refus est exact et
important.

### Le problème : redondance avec `near` à ces distances
La faiblesse est dans l'échelle. À ≤ 2 km, deux puits sont déjà presque toujours dans
le même sous-bassin (les sous-bassins de la vallée centrale font des dizaines de km).
La contrainte « même sous-bassin » ne mord donc que sur les rares paires situées **près
d'une frontière de sous-bassin** dans la fenêtre 0-2 km. Pour l'écrasante majorité des
arêtes, `same_subbasin_knn` reproduit `near` (avec un cap juste un peu plus large, 2 km
vs 1,5 km).

Conséquence : les deux relations encodent **largement la même chose** — proximité
spatiale capée. La valeur ajoutée mécaniste de la seconde relation se réduit à : (a) un
cap 0,5 km plus large, et (b) le veto frontière de sous-bassin sur une minorité de
paires. C'est réel mais marginal. Cela explique très bien pourquoi HGT/R-GCN
(multi-relationnel) ne battent pas GraphSAGE/GCN (mono-relationnel) : la seconde
relation n'apporte presque pas d'information neuve.

### Transport PFAS et frontières de sous-bassin
Les sources présentes (Chrome Plater, Airport, Refinery, Bulk Terminal) génèrent des
panaches qui suivent l'**écoulement local**, lequel reste en général dans un même
sous-bassin (les divides sont souvent aussi des lignes de partage des eaux
souterraines). Donc « PFAS reste dans le sous-bassin » est globalement vrai à l'échelle
du panache (≲ km). Mais c'est précisément pour cela que la contrainte est peu
discriminante à ≤ 2 km : à cette échelle on n'a presque jamais traversé une divide de
toute façon.

### Verdict opérationnel
- Mécaniquement DÉFENDABLE (pas une fuite), mais **attendre un gain marginal proche du
  bruit**. C'est cohérent avec les résultats (Δ vs mono-relation < σ inter-plis).
- Les **1 475 puits (~13 %) sans sous-bassin** restent isolés sous cette relation. Ce
  n'est pas un bug mais cela veut dire qu'ils ne reçoivent du message-passing QUE via
  `near` — la seconde relation est inopérante pour eux. À surveiller : si ces puits sont
  concentrés dans certaines régions (aquifères de socle, zones de montagne hors bassins
  alluviaux), leur instabilité dégradera les blocs spatiaux correspondants (cf. §4).

---

## 3. Signal source→puits : bien encodé ?

**Verdict : NON JUSTIFIÉ comme suffisant — mais le manque n'est PAS le nœud source.**

### Ce qui est correct
L'audit méthodologique a raison de refuser les nœuds « source » fabriqués par clique de
`nearest_geotracker_type` : ce serait 4 cliques géantes sans contenu spatial honnête
(C-NODE.1). En l'absence de coordonnées d'installation dans les données, encoder la
proximité-source comme **feature de nœud** (`dist_geotracker_km`,
`nearest_geotracker_type`, `n_geotracker_within_{1,3}km`) est le bon choix par défaut.
Pour la simple question « ce puits est-il près d'une source connue ? », la feature
agrégée suffit.

### Ce qui manque vraiment (la vraie limitation)
Le transport PFAS est **advectif et anisotrope** : le panache suit le **gradient
hydraulique**. La distance euclidienne (ou haversine) à la source n'est un bon proxy
**que si elle est alignée avec l'écoulement** (`HYDRO_CRITIQUE.md` §1.2). Or :
- `dist_geotracker_km` est isotrope : un puits 800 m **en amont** d'une source a la même
  feature qu'un puits 800 m **en aval**, alors que seul l'aval est sur le panache. C'est
  une perte d'information de premier ordre.
- Aucune variable de **direction d'écoulement / gradient de potentiométrie** n'est
  présente. Le modèle ne peut donc pas distinguer amont/aval.
- Aucune variable de **lithologie/perméabilité du chemin entre source et puits**, ni
  d'**épaisseur de zone non saturée (vadose)**. La vulnérabilité dépend fortement de la
  recharge verticale et du temps de transit dans la vadose.

Le seul vrai signal de vulnérabilité mécaniste présent est `well_depth_ft` (profond →
plus protégé, corr ≈ −0,35 d'après le profilage), plus la texture/ksat du sol.

### Conclusion §3
L'absence de nœud source n'est PAS la limitation importante (la feature agrégée la
remplace correctement). La limitation importante est l'**absence de direction de
transport** : le signal est encodé de façon isotrope alors que la physique est
directionnelle. C'est probablement un plafond dur sur ce que TOUTE topologie non
orientée peut apprendre ici.

---

## 4. Interprétation des résultats

### Pourquoi le graphe n'apporte presque rien (AUC spatiale ~0,62-0,65, ≈ mur RF 0,60) ?
Du point de vue hydrogéochimique, ce résultat n'est **PAS étonnant** — il est même
attendu :

1. **Le signal spatial exploitable est de très courte portée et déjà presque capté par
   les features.** L'autocorrélation retombe à la base dès 1-5 km. Les features de
   contexte (densité geotracker dans 1-3 km, cocontaminants, profondeur) encodent déjà
   l'essentiel du voisinage utile. Le graphe ne fait qu'ajouter un lissage local
   redondant.
2. **Les arêtes encodent la co-localisation, pas le mécanisme directionnel.** Sans
   direction d'écoulement, le message-passing ne peut pas reconstruire la structure de
   panache amont/aval. Il agrège un voisinage moyen → il régularise, il ne révèle pas un
   mécanisme nouveau.
3. **Les deux relations sont quasi-redondantes** (§2). Un encodeur multi-relationnel
   sophistiqué (HGT/R-GCN) sur deux relations qui disent la même chose ne peut pas faire
   mieux qu'un encodeur mono-relationnel — c'est ce qu'on observe.

Donc : le faible apport du graphe est une **conclusion physique cohérente**, pas un
échec d'implémentation. Le bon message est « la connectivité spatiale capée n'ajoute
pas de signal mécaniste au-delà des features de voisinage déjà présentes ».

### Plis instables (AUC ≈ 0,46-0,51 sur certains blocs)
Mécaniquement très plausible, et attendu :
- Le signal PFAS est **fortement hétérogène spatialement** : clusters urbains/industriels
  vs vastes zones rurales à faible prévalence et faible densité de sources. Un bloc
  spatial tombant sur une région à régime hydrogéologique différent (aquifères profonds
  confinés, zones de socle/montagne hors bassins alluviaux, où vivent une partie des
  1 475 puits sans sous-bassin) présente une relation features→cible différente de celle
  apprise sur les blocs urbains de la vallée.
- Un modèle entraîné sur des blocs « panache urbain » et testé sur un bloc « aquifère
  rural profond » peut tomber sous 0,5 (anti-corrélation locale) : la même feature
  (p. ex. faible profondeur) n'y a pas le même sens.
- C'est exactement la signature d'un signal dont la **stationnarité spatiale est
  fausse**. L'instabilité inter-blocs (σ ≈ 0,06-0,07) est donc ATTENDUE et constitue un
  résultat honnête, pas un défaut à masquer. C'est la raison d'être de la CV par blocs.

**Recommandation d'interprétation** : cartographier les AUC par bloc sur la carte et
croiser avec (a) la densité de puits, (b) la prévalence T1 locale, (c) la part de puits
sans sous-bassin SGMA. On s'attend à ce que les blocs faibles soient ruraux/profonds ou
riches en puits non assignés. Si c'est le cas, l'instabilité est mécaniste et explicable.

---

## 5. Recommandations hydrogéochimiques

### 5.1 Features/arêtes manquantes, par priorité mécaniste
1. **Direction d'écoulement souterrain / gradient de potentiométrie (priorité 1).**
   C'est le chaînon manquant central. Avec une surface piézométrique (DWR publie des
   niveaux de nappe), on peut :
   - dériver une feature **amont/aval relative à la source la plus proche** (produit
     scalaire entre vecteur source→puits et gradient local) → transforme
     `dist_geotracker_km` isotrope en proxy de panache directionnel ;
   - construire une **arête orientée amont→aval** entre puits proches alignés avec le
     gradient. CE serait la première arête vraiment mécaniste (transport advectif), bien
     plus défendable que `same_subbasin_knn`.
2. **Épaisseur de zone non saturée (depth-to-water) et lithologie du chemin (priorité
   2).** Proxy direct de vulnérabilité verticale et de temps de transit. `well_depth_ft`
   est un substitut partiel ; depth-to-water serait plus propre.
3. **Typologie de sources élargie (priorité 2).** Les 4 types geotracker **ratent les
   sources PFAS dominantes** : AFFF militaire/DoD, sites d'entraînement feu, STEP/
   biosolides, décharges, fluorochimie (`HYDRO_CRITIQUE.md` §2). `nearest_geotracker_type`
   est un proxy partiel et biaisé. Enrichir avec couches DoD/AFFF, EPA ECHO, décharges
   CalRecycle améliorerait le signal source bien plus qu'un raffinement de topologie.
4. **Garder `within_1km`/`within_3km` ; downgrader `within_50km`** (urbanité, pas
   panache).

### 5.2 Le cap 1,5 km est-il justifié pour la Californie ?
- Justifié comme **borne anti-fuite** (range d'autocorrélation), PAS comme longueur de
  panache mesurée — aucune donnée de longueur de panache PFAS californien n'est
  mobilisée ici. Le dire explicitement.
- Je recommande de **réduire le défaut `near` à 1,0 km** et de traiter 1,5 km comme
  ablation, la concordance retombant à la base dès 1 km. Tester {0,5 ; 1,0 ; 1,5}.
- Pour les gros sites AFFF (panaches pluri-km), un cap court sous-représente l'aval
  lointain — mais ces sites sont justement ceux mal couverts par geotracker, donc
  l'arête ne les capterait pas correctement de toute façon. Le bon levier est la
  feature source directionnelle (5.1.1), pas un cap plus large.

### 5.3 La métaphore graphe puits-puits est-elle la bonne ?
- Pour capturer l'**autocorrélation locale** : oui, le k-NN capé non orienté est
  adéquat (et déjà à son plafond utile — d'où le faible gain).
- Pour capturer le **transport** : non. Le transport est directionnel ; un graphe non
  orienté ne peut pas l'exprimer. La métaphore plus juste serait un **graphe orienté
  amont→aval le long du gradient hydraulique** (réseau d'écoulement), pas un graphe de
  voisinage symétrique. C'est la seule évolution qui justifierait réellement un encodeur
  relationnel : alors `flows_to` (orienté, gradient) et `near` (symétrique, voisinage)
  seraient deux relations mécaniquement DISTINCTES — contrairement au couple actuel
  `near` / `same_subbasin_knn` qui dit deux fois la même chose.
- Une approche grille de flux / réseau de drainage est plus lourde et risque de
  réintroduire de la carte ; je ne la recommande pas avant d'avoir testé l'arête
  orientée par gradient, qui est l'incrément minimal mécaniste.

### 5.4 Recommandations pour l'expérience suivante
1. Ne PAS investir davantage dans HGT/R-GCN multi-relationnel **tant que les deux
   relations restent quasi-redondantes** : le plafond est structurel.
2. Prochaine arête à tester : **`flows_to` orientée par le gradient hydraulique** (si
   une couche piézométrique DWR est intégrable). C'est le seul ajout avec un mécanisme
   nouveau (advection), pas une variante de proximité.
3. Avant cela, valider que le gain attendu existe : ablation `near` seul vs
   `near + same_subbasin_knn` ; si Δ < σ inter-plis (≈ 0,06), conclure officiellement
   que la seconde relation est inutile et la retirer (parcimonie).
4. Lancer un run de PRODUCTION (`smoke=False`) avant toute conclusion chiffrée : les
   AUC du REPORT actuel sont issues d'un smoke-test et ne doivent pas être citées comme
   résultat.
5. Croiser AUC-par-bloc avec géographie (§4) pour documenter mécaniquement
   l'instabilité plutôt que de la subir.

---

## Note de parcimonie finale
Le design actuel est **honnête et anti-fuite** (caps durs, coupe inter-blocs par
relation assertée à 0, lat/lon hors features, source en feature non en nœud). Le
problème n'est pas la rigueur, c'est que la **physique exploitable par une topologie de
voisinage non orientée est faible et déjà captée par les features**. Le faible apport du
graphe est un RÉSULTAT mécaniste valide à revendiquer comme tel, pas un échec à corriger
par un encodeur plus gros. La seule voie d'amélioration mécaniste réelle passe par la
**direction d'écoulement** ; sans elle, ajouter des relations de proximité revient à
réencoder la même information.
