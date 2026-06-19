# HYDRO_CRITIQUE — Critique mécaniste de l'espace de features & du graphe (PFAS / CA)

> Étape 2. Agent `hydro-domain-expert`. Critique de domaine de l'espace de features
> proposé dans `experiments/profilage/REPORT.md` (familles A→J), des cibles T1/T2 et de
> la construction du graphe GNN. **Je n'établis pas de faits nouveaux sur les données** :
> je m'appuie sur les chiffres du profilage et j'apporte la plausibilité physique
> (sources, transport, sorption, précurseurs). Tout ce qui est « statistiquement bon
> mais physiquement douteux » est signalé comme tel.
>
> Principe : une importance forte d'une variable sans mécanisme de transport/source
> crédible doit être traitée comme **suspecte** (confondeur d'échantillonnage / artefact
> de design), pas comme un driver.

---

## 0. Rappel des processus PFAS qui guident toute la critique

1. **Source ponctuelle dominante, pas diffuse.** En eaux souterraines, les dépassements
   PFAS sont quasi toujours pilotés par une **source identifiable** (AFFF, industrie
   fluorochimique, décharge/lixiviat, biosolides/épandage de STEP, papeterie, chromage).
   Le signal utile est donc « **suis-je sur le panache d'une source ?** », pas une
   propriété de terrain générique.
2. **Transport advectif le long de l'écoulement**, peu retardé pour les chaînes courtes.
   Le panache suit le gradient hydraulique ; la distance euclidienne à la source n'est
   un bon proxy **que si elle est alignée avec l'écoulement**.
3. **Sorption dépendante de la longueur de chaîne et du groupe fonctionnel.** Les
   sulfonates (PFSA, p. ex. PFOS, PFHxS) sorbent plus que les carboxylates (PFCA) à
   nombre de C égal ; pour une même famille, **plus la chaîne est longue, plus la
   rétention est forte**. Conséquence en aquifère : les **chaînes courtes (PFBA, PFPeA,
   PFBS, PFHxA)** migrent loin et vite (front de panache), les **chaînes longues (PFOS,
   PFOA, PFNA, PFDA)** restent près de la source. La **matière organique (foc)** et la
   teneur en argile augmentent la rétention.
4. **Précurseurs.** Une part des PFAS détectés provient de la **transformation de
   précurseurs** (FOSA/FOSE → FOSAA → PFOS ; fluorotélomères → PFCA). Cela explique des
   co-occurrences (FTS, FOSAA) liées à un type de source (AFFF) et non à la chimie de
   l'aquifère.
5. **Censure analytique.** Le profilage montre que les non-détects sont stockés à la
   limite de rapport et que les cocontaminants ont le même artefact (p50 = LD
   constante). Toute feature dérivée d'un panel analytique doit être lue à travers ce
   prisme : *la présence d'une mesure est elle-même informative du design*.

---

## 1. Verdict synthétique par famille de features

Légende : **GARDER** (mécanisme crédible) · **ÉLAGUER** (bruit probable) ·
**REFORMULER** (utile mais à transformer) · **MÉFIANCE-CONFONDEUR** (peut prédire pour
de mauvaises raisons : design d'échantillonnage, co-analyse labo, accessibilité).

| Famille | Verdict | Justification mécaniste courte |
|---|---|---|
| **A. lat/lon** | **REFORMULER** (porter par le graphe, pas en feature de nœud) | Pas un mécanisme ; pur proxy de localisation. En feature brute → le modèle apprend « telle zone = pollué » (mémorisation de carte) et la CV aléatoire surévalue. À convertir en arêtes k-NN (cf. §4), et tester *avec/sans* lat-lon en nœud. |
| **B. admin/hydrogéo** | **PARTIEL** : `dwr_basin`/`sgma_subbasin` GARDER comme proxys d'aquifère ; `county`/`regional_board`/`sgma_region_office` MÉFIANCE-CONFONDEUR | Le **bassin/sous-bassin** approxime une unité hydrogéologique (connectivité, lithologie, profondeur de nappe) → mécaniquement défendable. Les frontières **administratives** (comté, regional board) ne sont PAS des unités physiques : elles encodent surtout *qui échantillonne et qui dépollue* (LA county sur-représenté). Risque de confondeur de programme. |
| **C. puits — `gm_well_category`** | **GARDER** (avec prudence) | Municipal vs monitoring ≈ deux régimes : un puits **monitoring** est souvent foré *parce qu'on suspecte une source* → corrélé à la contamination par **design d'enquête**, pas par physique. Utile mais à auditer SHAP comme semi-confondeur. |
| **C. puits — `well_depth_ft`** | **GARDER, REFORMULER** | **Le seul vrai signal mécaniste fort du tableau** : profondeur ↑ → vulnérabilité ↓ (corr −0,35 cohérente : aquifères profonds/captifs moins atteints par une contamination de surface récente comme les PFAS). MAIS 94,5 % manquant → l'indicateur `well_depth_ft_missing` risque de devenir un **proxy de catégorie de puits / de programme** plutôt que de profondeur. Imputer par bassin, garder l'indicateur, et **surveiller que SHAP attribue au *manque* et non à la profondeur** (signal d'alerte). |
| **D. sources géotracker** | **GARDER mais INCOMPLET (lacune majeure de typologie)** | Mécanisme idéal (proximité à une source). Cf. §2 : les 4 types présents (Chrome Plater, Bulk Terminal, Airport, Refinery) **ratent les sources PFAS dominantes** (AFFF/militaire, STEP/biosolides, décharges, fluorochimie). `nearest_geotracker_type` est donc un proxy **partiel et biaisé** des sources. Rayons : cf. §4. |
| **E. cocontaminants (44)** | **MÉFIANCE-CONFONDEUR forte ; garder un sous-ensemble restreint, jamais en bloc** | Cf. §3 détaillé. La majorité (VOC chlorés, BTEX, fréons, pesticides) n'a **pas** de co-source PFAS et agit surtout comme **proxy « cet échantillon a été passé en multi-panel / sur un site cleanup »**. Quelques-uns ont une co-occurrence mécaniste plausible (TDS, nitrate, Mn/Fe redox) — comme *contexte hydrogéochimique*, pas comme co-polluant. |
| **F. sol SSURGO** | **GARDER un noyau, ÉLAGUER les redondances** | Mécanisme réel mais **de second ordre** ici (sorption/recharge). Garder `soil_om_pct` (foc → rétention), `soil_ksat`/texture (vitesse d'infiltration → vulnérabilité), `soil_awc`. **Élaguer** la granulométrie ultra-détaillée (8 sous-classes de sable/silt, gradation) : colinéaire, >95 % manquant pour certaines (`soil_silt_*`, `soil_gradation_*`), et le sol de *surface* SSURGO décrit mal la **zone non saturée profonde** qui contrôle réellement le transport. Attendu : effet faible. |
| **G. climat/hydro GLDAS** | **REFORMULER → garder seulement la recharge ; ÉLAGUER la météo instantanée** | Mécanisme : **recharge** (pluie nette = lessivage vers la nappe) influence la dilution/mobilisation. Mais les valeurs sont **mensuelles à la date de prélèvement** : la nappe intègre des **années à décennies** de recharge → une humidité de sol du mois M est physiquement déconnectée de la concentration PFAS. Garder un proxy de **recharge climatique moyenne** (aridité) ; ÉLAGUER `soil_moi_*` instantané, `temp_c`, `et`, `runoff` ponctuels (bruit / proxy de saison). `snowpack` 90 % zéro → ÉLAGUER. |
| **H. air AQS** | **ÉLAGUER (quasi tout)** | Le **dépôt atmosphérique** de PFAS existe (panaches d'usines fluorochimiques, aérosolisation marine) mais : (i) il alimente surtout les **eaux de surface / sols**, pas un signal détectable en **nappe** à l'échelle d'un capteur AQS urbain ; (ii) PM2.5/NO2/SO2/CO/ozone sont des **marqueurs d'urbanisation/trafic**, donc un **confondeur d'urbanité** (les zones urbaines ont plus de sources ET plus de PFAS). Si une variable AQS ressort en SHAP, c'est presque sûrement de l'**urbanité confondue**, pas du dépôt. Garder au plus 0–1 variable comme contrôle d'urbanité, explicitement étiquetée comme telle. |
| **I. temporel** | **GARDER comme contrôle, MÉFIANCE-CONFONDEUR** | Pas un mécanisme de contamination (les PFAS sont persistants, pas de tendance saisonnière forte en nappe profonde). L'année capture surtout la **dérive d'échantillonnage** (35 lignes en 2016 → 10 025 en 2025) et l'évolution des **seuils de rapport analytiques**. À inclure pour *contrôler* la dérive, pas comme prédicteur causal. Saison : effet attendu négligeable en nappe. |
| **J. provenance `gm_dataset_name`** | **NE PAS utiliser comme feature ; garder en variable de contrôle/audit** | C'est le **confondeur de design par excellence** : `WB_CLEANUP` cible des sites déjà connus comme pollués → prévalence artificiellement haute. Laisser le modèle s'en servir = apprendre « cet échantillon vient du programme de dépollution » = fuite de conception. À utiliser uniquement pour stratifier/auditer. |

**Synthèse hiérarchie mécaniste (du plus au moins défendable)** :
`well_depth_ft` > sources (D, si complétées) > bassin/sous-bassin (B) >
foc/ksat/texture sol (F, noyau) > recharge moyenne (G, reformulée) >>
cocontaminants redox/TDS (E, sous-ensemble) >> tout le reste (AQS, météo instantanée,
admin, temporel) qui relève du **contrôle de confondeurs**, pas de la prédiction causale.

---

## 2. Sources PFAS manquantes (lacune la plus grave de l'espace de features)

La famille D ne contient que **4 types géotracker** : Chrome Plater, Bulk Terminal,
Airport, Refinery. C'est une couche « sites pétroliers / industriels classiques ». Du
point de vue PFAS, **les sources les plus émettrices sont absentes ou mal couvertes** :

1. **AFFF — bases militaires, aéroports civils ET sites d'entraînement incendie** *(la
   source n°1 des panaches PFAS en eaux souterraines)*. « Airport » est présent mais
   probablement limité aux aéroports civils géotracker ; il **manque les bases
   militaires (DoD), les sites d'entraînement feu, les casernes de pompiers, les
   raffineries/terminaux avec zones AFFF**. Signature attendue : **PFOS + PFHxS +
   6:2 FTS + 8:2 FTS + FOSAA** élevés. C'est exactement le cluster que T2 voit
   (FTS_6_2~FTS_8_2 0,88 ; FOSAA). → **Ajouter une couche AFFF/DoD si disponible**
   (bases militaires, aéroports avec activité incendie, fire training areas).
2. **Stations d'épuration (STEP) et épandage de biosolides / irrigation par eaux
   recyclées.** Source diffuse majeure en Californie (réutilisation agricole). Signature :
   PFCA chaînes courtes (PFBA, PFPeA, PFHxA) + précurseurs. → **Ajouter localisation
   des STEP, des champs d'épandage de biosolides, des zones d'irrigation par eaux
   recyclées.**
3. **Décharges / sites d'enfouissement (lixiviats).** Source classique, signature large
   (PFBA, PFHxA, PFOA, 5:3 acide). → **Ajouter les landfills (CalRecycle / geotracker
   SWIS).**
4. **Industrie fluorochimique / fabrication & usage** (placage métallique au-delà du
   chromage, textiles/imperméabilisants, papeterie, semi-conducteurs / déchets
   électroniques, photolithographie). La Silicon Valley (Santa Clara très représentée)
   plaide pour une source **semi-conducteurs/électronique** spécifiquement PFAS. →
   **Ajouter sites fab/électronique et papeteries.**
5. **Aéroports : distinguer civil vs activité AFFF.** Un aéroport sans entraînement feu
   n'est pas une source PFAS ; un avec l'est fortement. Le type brut « Airport » mélange
   les deux.

**Conséquence méthodologique** : tant que ces couches manquent, `nearest_geotracker_type`
et les comptes géotracker sont un proxy de source **partiel et systématiquement biaisé
vers les sources non-PFAS**. Le modèle compensera en surchargeant des **confondeurs
d'urbanité/industrialité** (E, H, admin). À documenter comme **limite majeure** ; si les
couches AFFF/STEP/décharge sont accessibles (geotracker SWIS, EPA ECHO, DoD AFFF
inventory, SSO/biosolids CA), leur ajout est la **priorité n°1** pour rendre le modèle
mécaniquement honnête. À défaut, garder D mais ne jamais le présenter comme « le modèle
identifie les sources ».

---

## 3. Cocontaminants (E) : co-transport mécaniste OU proxy de panel analytique ?

**Tranche nette : très majoritairement des proxys de design d'échantillonnage, PAS des
co-polluants mécaniquement co-transportés.** Trois arguments factuels tirés du profilage :

- **Artefact de censure identique aux PFAS** : la plupart des `cocontam_*` ont un p50 =
  une valeur constante (limite de rapport) — ex. `cocontam_dbcp` p50=0,25, `cocontam_naph`
  p50=0,2, fréons à 1e-5… Ce sont massivement des **non-détects encodés à la LD**. Donc
  ce que porte surtout la variable, c'est « **l'analyte a été inclus dans le panel et
  mesuré ici** », ce qui est un **marqueur de programme/labo**, fortement co-déterminé
  avec le fait qu'on a aussi cherché les PFAS sur le même échantillon → **corrélation
  non causale avec la cible**.
- **Manquance structurée par programme** : `cocontam_no3n` 94,8 % manquant, métaux
  (As/Mn/Fe/TDS) ~31 % manquant, VOC ~1 % manquant. Ces motifs de manque **encodent le
  type de site/programme** (un site cleanup VOC vs un puits municipal DDW), pas la
  géochimie.
- **Mécanisme de co-transport réel ≠ co-occurrence de source.** Les PFAS ne sont pas
  chimiquement liés aux solvants chlorés ; quand ils co-occurrent, c'est parce qu'un
  **même site industriel** a relâché les deux (co-localisation de source), pas par
  co-transport. C'est donc encore un proxy de **« site industriel pollué »**.

**Classement mécaniste des cocontaminants** :

- **MÉFIANCE-CONFONDEUR (proxys de panel/site, retirer ou traiter en contrôle)** : tous
  les VOC chlorés (TCE/PCE/TCA/DCE/DCA/VC/CTCl/chlorobenzènes), BTEX (benzène, toluène,
  xylènes, éthylbenzène, styrène), **fréons CFC `fc11`/`fc12`/`fc113`** (attention :
  « fluoro » dans le nom mais ce sont des **CFC, PAS des PFAS** et **pas co-sourcés** —
  ne jamais les laisser passer pour un signal PFAS), pesticides/fumigants
  (DBCP, EDB, TCP-1,2,3), MTBE/TBA. Aucun mécanisme de co-transport PFAS. Les colonnes
  dupliquées (`tmb124==dce12c==btbzt`) et `xylenes` (99,8 % vide) à **supprimer** (déjà
  noté par l'analyste).
  - *Nuance MTBE / fréons / VC* : ce sont des composés **mobiles et persistants** comme
    les PFAS courts ; ils peuvent co-tracer un même **régime hydraulique vulnérable**
    (nappe peu profonde, oxydante, urbaine). Valeur possible comme **proxy de
    vulnérabilité hydrogéologique**, mais c'est exactement le rôle qu'on veut faire jouer
    à `well_depth` / texture — donc redondant et plus propre via ces dernières.
- **CONTEXTE GÉOCHIMIQUE plausible (garder un petit sous-ensemble, étiqueté
  « contexte », pas « co-polluant »)** :
  - `cocontam_tds` (salinité) et `cocontam_so4` : marqueurs d'un aquifère **âgé / à
    faible renouvellement vs jeune / récemment rechargé**. Une nappe jeune et oxydante
    est plus susceptible de porter des PFAS de surface récents → lien **indirect mais
    physique** via l'âge de l'eau. Acceptable comme proxy de vulnérabilité.
  - `cocontam_no3n` (nitrate) : **le meilleur candidat mécaniste de la famille**. Le
    nitrate est un **traceur reconnu de contamination de surface descendante** (recharge
    agricole/urbaine, eaux usées) ; un puits nitraté est un puits **connecté à la
    surface et vulnérable**, donc plus susceptible de PFAS. Co-occurrence via
    **vulnérabilité partagée**, pas co-source. Garder, mais 94,8 % manquant en limite
    fortement la portée → l'indicateur de manque redeviendra un proxy de programme.
  - `cocontam_mn`, `cocontam_fe`, `cocontam_as` (redox) : indicateurs des **conditions
    redox** de l'aquifère. Pertinence indirecte : conditions réductrices (Mn/Fe/As
    élevés) ↔ aquifère **confiné/profond/ancien**, donc *moins* vulnérable aux PFAS de
    surface → effet **négatif** attendu. Garder comme proxy redox/profondeur, signe
    attendu **opposé** à nitrate.

**Recommandation E** : ne **jamais** injecter les 44 en bloc. Retenir un **noyau
contexte hydrogéochimique** {nitrate, TDS, SO4, Mn, Fe, As} avec indicateurs de manque,
**audit SHAP obligatoire**, et **retirer tout le bloc VOC/BTEX/fréons/pesticides** du jeu
de features prédictif (les conserver seulement pour des analyses de robustesse / détection
de confondeur). Tester un modèle **sans aucun cocontaminant** comme référence : si l'AUC
chute fortement *grâce aux VOC*, c'est le signe que le modèle s'appuie sur le design
analytique, pas sur la physique.

---

## 4. Construction du graphe GNN : arêtes justifiées vs fuite spatiale

Le profilage établit une autocorrélation T1 réelle mais de **courte portée** : Moran's I
= 0,43, concordance 0–1 km = 77 % mais retombe à la base (~54 %) **dès 1–5 km**. C'est le
fait dirigeant pour le graphe.

**Ce que cela implique mécaniquement** : la structure spatiale exploitable correspond à
l'échelle d'un **panache / d'une source partagée**, soit **≲ 1–2 km**. Au-delà,
« proximité spatiale » ne fait que **réencoder la carte** (mêmes zones urbaines/bassins)
et devient de la **fuite spatiale** sous CV aléatoire.

**Recommandations d'arêtes** :

1. **k-NN spatial à courte portée — OUI, mais borné par une distance physique.**
   - Utiliser un k petit (k≈5–8) **ET un seuil de distance dur** (p. ex. couper toute
     arête > ~1–2 km), pour que les arêtes approximent une **connectivité de panache**,
     pas une co-appartenance régionale. Un k-NN sans plafond de distance dans une zone
     dense (LA) relierait des puits à quelques centaines de mètres (bon) mais dans une
     zone clairsemée relierait des puits à 50 km (fuite/non-sens).
   - **Anisotropie** : idéalement pondérer/orienter selon le **gradient hydraulique**
     (un puits en aval d'un voisin contaminé est plus à risque que l'amont). Si une carte
     piézométrique ou un MNT (proxy de sens d'écoulement) est disponible, créer des
     **arêtes dirigées amont→aval**. À défaut, l'arête non orientée reste acceptable à
     courte portée mais perd le mécanisme directionnel.
2. **Arêtes « même sous-bassin/aquifère » (`dwr_basin`/`sgma_subbasin`) — OUI avec
   parcimonie.** Mécaniquement défendable (connectivité hydraulique intra-aquifère),
   mais un sous-bassin peut compter des centaines de puits → un graphe « clique par
   bassin » crée des arêtes massives qui **réencodent l'appartenance de bloc** et
   **entrent en collision directe avec la CV spatiale par blocs** (fuite entre
   train/test si les blocs CV ne respectent pas les bassins). **Règle** : si on relie par
   bassin, le **schéma de blocs CV doit être au moins aussi grossier** (LeaveOneRegionOut
   ou blocs ⊇ bassins), sinon les arêtes traversent la frontière train/test. Préférer
   des **sous-cliques limitées aux k plus proches *du même bassin*** plutôt qu'une clique
   complète.
3. **Arêtes « même source / même panache » — OUI, le plus mécaniste, à construire.**
   Relier les puits partageant la **même source géotracker la plus proche** (ou situés
   dans le même rayon d'une source) approxime un **panache commun** → c'est exactement le
   mécanisme PFAS. C'est plus défendable qu'une arête purement géométrique. **À
   privilégier** dès que les couches sources sont enrichies (§2).
4. **Arêtes temporelles (même puits, événements successifs) — OUI, naturelles.** Les
   réplicats temporels d'un même `gm_well_id` doivent être reliés (ou agrégés), et **rester
   du même côté du split** (déjà acté : grouper par `gm_well_id`).
5. **À PROSCRIRE** : arêtes par **county / regional_board / dataset_name** (réencodent le
   programme administratif = confondeur), et tout k-NN spatial **sans plafond de
   distance** (fuite spatiale longue portée). Une topologie qui améliore l'AUC en CV
   aléatoire mais s'effondre en CV spatiale par blocs est le **symptôme d'arêtes qui
   réencodent la carte** — à rapporter explicitement (écart CV aléatoire vs spatiale,
   comme exigé au §3.2 de CLAUDE.md).

**Rayons géotracker 1/3/10/50 km** : pour un **panache d'aquifère**, l'échelle pertinente
est **≲ 1–3 km** (les panaches PFAS dépassent rarement quelques km, sauf gros sites AFFF
qui peuvent atteindre plusieurs km en aval). Donc `within_1km` et `within_3km` sont les
plus mécanistes ; **`within_50km` n'est pas un proxy de panache** mais un proxy de
**densité industrielle régionale / urbanité** (confondeur). Garder 1 et 3 km comme
signaux de source ; traiter 10 km comme marginal et **50 km comme confondeur d'urbanité**
(à auditer, candidat à l'élagage). `dist_geotracker_km` (p50 ~3,9 km) reste utile en
log1p comme distance à la source la plus proche, mais sa valeur dépend entièrement de la
qualité de la typologie source (§2).

---

## 5. Validité des cibles (T1 / T2)

### 5.1 Seuils du Hazard Index T1a — CORRECTS

La formule retenue **HI = PFHxS/10 + PFNA/10 + HFPO-DA/10 + PFBS/2000** (ng/L) **est
conforme à l'EPA 2024** (Final PFAS NPDWR, avril 2024). Vérification des Health-Based
Water Concentrations (HBWC) utilisées comme dénominateurs :

- PFHxS : 10 ng/L ✔
- PFNA : 10 ng/L ✔
- HFPO-DA (GenX) : 10 ng/L ✔
- PFBS : 2000 ng/L ✔
- Et MCL individuels **PFOA = 4 ng/L**, **PFOS = 4 ng/L** ✔ (la règle `PFOA>4 OU
  PFOS>4 OU HI≥1` reproduit bien la logique réglementaire : MCL pour PFOA/PFOS, Hazard
  Index pour le mélange des quatre autres).

**Aucune erreur de seuil sur T1a.** Réserve mécaniste, pas réglementaire :

- Le profilage note que `X_ngL` stocke la LD quand non-détecté. **Croiser
  systématiquement avec `*_detected`** avant de déclencher un dépassement : un PFOA non
  détecté avec LD=4–5 ng/L (dilutions) ne doit pas compter comme `PFOA>4`. C'est un point
  **critique** ici car le MCL (4 ng/L) est du même ordre que des limites de rapport
  observées (PFOA p50=1,475 mais max de LD bien plus haut). **Recommandation** : définir
  T1a comme `(PFOA_detected & PFOA>4) OU (PFOS_detected & PFOS>4) OU (HI≥1 sur analytes
  détectés)`. Sinon risque de **faux positifs de cible** sur des non-détects à LD élevée
  — ce qui empoisonnerait l'apprentissage.

### 5.2 Seuil de label individuel T2 à 2,0 ng/L — défendable mais à assumer comme
**seuil analytique, pas sanitaire**

2,0 ng/L n'est **pas** un seuil réglementaire (les MCL PFOA/PFOS sont à 4 ; les HBWC vont
de 10 à 2000). C'est manifestement un **seuil de quantification/reporting** commun. Comme
cible **multilabel de détection significative**, c'est cohérent et pragmatique
(prévalences exploitables), à condition de le **nommer correctement** : T2 prédit « *cet
analyte dépasse ~2 ng/L (≈ quantifié à niveau non-trace)* », **pas** « *cet analyte
dépasse son seuil sanitaire* ». À documenter pour ne pas sur-interpréter. Un même seuil
absolu (2,0) appliqué à tous les analytes est **chimiquement hétérogène** (2 ng/L est
trivial pour PFBS dont la HBWC=2000, mais proche du MCL pour PFOA=4) — acceptable pour une
tâche de détection, mais à ne pas confondre avec un risque sanitaire par analyte.

### 5.3 Seuil somme > 70 ng/L (T1b) — défendable comme repère historique, à étiqueter

70 ng/L correspond à l'**ancien Health Advisory combiné PFOA+PFOS de l'US-EPA (2016)**,
souvent réutilisé comme « somme PFAS ». Ici il est appliqué à **Σ des 31 analytes**, ce
qui est **plus permissif** que l'esprit original (qui visait PFOA+PFOS uniquement) :
sommer 31 analytes gonfle la somme et change la sémantique. **Défendable comme cible
secondaire de « charge PFAS totale élevée »**, mais à étiqueter clairement comme un seuil
**non réglementaire actuel** (l'EPA 2024 a remplacé l'approche somme par MCL+HI). Garder
T1a comme cible primaire (bon choix de l'analyste) ; T1b en secondaire/robustesse.

---

## 6. Co-occurrences PFAS (T2) : cohérence chimique et dépendances exploitables

Les corrélations de labels observées sont **mécaniquement cohérentes** et utilisables :

- **NEtFOSAA ~ NMeFOSAA = 0,98** : ce sont **deux précurseurs FOSAA de la même famille
  sulfonamide (ECF, électrofluoration historique 3M)**, co-fabriqués et co-présents dans
  l'**AFFF ancien**. Quasi-identité attendue. **Exploitable** : un seul facteur latent
  « précurseurs FOSAA / AFFF ECF ».
- **FTS_6_2 ~ FTS_8_2 = 0,88** : **fluorotélomères sulfonates**, marqueurs de l'**AFFF
  fluorotélomère (post-2002)**. Co-occurrence = signature de source AFFF FT. **Exploitable**
  comme facteur « AFFF fluorotélomère ». La distinction FOSAA (ECF) vs FTS (FT) sépare
  *l'âge/le type d'AFFF* — information de source réelle.
- **PFOA ~ PFOS = 0,75** (et PFHxA~PFOA 0,75, PFBA~PFPeA 0,76) : co-détection des PFAS
  « legacy » les plus ubiquistes ; cohérent (sources multiples, persistance, ubiquité).
- **PFTeDA~PFTrDA 0,95, PFDoDA~PFUnDA 0,93, PFDoDA~PFTrDA/PFTeDA ~0,85** : **homologues à
  très longue chaîne** (C11–C14). Leur co-occurrence reflète (i) la **distribution
  d'homologues d'une même source** et (ii) le fait qu'ils **co-sorbent/co-restent près de
  la source** (faible mobilité). Cluster « PFCA longues chaînes ».
- **F53B_major ~ F53B_minor 0,78, ADONA~F53B 0,77** : alternatives au PFOS (chrome
  plating chinois pour F-53B ; ADONA = alternative GenX-like). Cohérent avec une source
  **placage / fluorochimie spécifique**. Rares ici (prévalence < 5 %).

**Dépendances de labels mécaniquement exploitables pour T2** (à encoder via un
**graphe de labels / classifier chains / structure hiérarchique**) :
1. **Facteur « AFFF »** regroupant {PFOS, PFHxS, 6:2 FTS, 8:2 FTS, NEtFOSAA, NMeFOSAA} —
   co-activés par une source mousse anti-incendie.
2. **Facteur « chaînes courtes mobiles / front de panache »** {PFBA, PFPeA, PFBS, PFHxA,
   PFHpA} — co-activés en aval / sources diffuses (décharge, STEP).
3. **Gradient de longueur de chaîne** : ordonner les labels par nombre de C permet
   d'exploiter le fait que *si une chaîne longue est présente, les courtes le sont en
   général aussi* (la source émet tout le spectre, et les courtes migrent plus loin) —
   relation d'**implication partielle** exploitable par un modèle de structure de labels.
4. **Séparation PFSA vs PFCA** : à nombre de C égal, le sulfonate sorbe plus → présence
   relative PFOS vs PFOA informe sur la **distance à la source** (près = enrichi en
   sulfonates/chaînes longues ; loin = enrichi en carboxylates courts). Un GNN qui
   apprend cette **anisotropie de composition le long du panache** ferait de la physique
   réelle.

---

## 7. Signaux d'alerte pour l'interprétation SHAP (à surveiller en aval)

**Directions ATTENDUES (rassurantes, mécanistes)** :

- `well_depth_ft` ↑ → P(dépassement) ↓ (aquifère profond/captif protégé). Inverse =
  alerte.
- `dist_geotracker_km` ↑ → P ↓ ; `n_geotracker_within_1km`/`3km` ↑ → P ↑ (proximité de
  source). Monotonie attendue.
- `soil_om_pct` ↑ (foc) → rétention → **chaînes longues** plus probables localement, mais
  effet net faible/ambigu en nappe (sorption retarde sans empêcher) ; effet de **second
  ordre**.
- `soil_ksat` ↑ / sol sableux → infiltration rapide → vulnérabilité ↑ → P ↑.
- nitrate ↑ → P ↑ (vulnérabilité de surface partagée) ; Mn/Fe/As ↑ → P ↓ (redox
  réducteur ≈ profond/confiné).
- Composition par chaîne (T2) : near-source enrichi en PFOS/chaînes longues, far-field en
  chaînes courtes — cohérent avec sorption longueur-dépendante.

**SIGNAUX D'ALERTE (importance forte = suspecte de fuite/artefact)** :

1. **Un cocontaminant VOC/BTEX/fréon (TCE, PCE, MTBE, benzène, FC11/12/113) en top
   importance** → quasi-certainement un **proxy de panel analytique / site cleanup**, pas
   un mécanisme. Re-tester sans le bloc VOC ; si l'AUC s'effondre, le modèle trichait.
2. **`gm_dataset_name` (surtout WB_CLEANUP) avec forte importance** → fuite de design
   (le programme conditionne la prévalence). Inacceptable comme driver.
3. **`county`/`regional_board`/`sgma_region_office` en haut du classement** → mémorisation
   de carte administrative ; à confronter à la chute en CV spatiale par blocs.
4. **`well_depth_ft_missing` (l'indicateur) plus important que `well_depth_ft` (la
   valeur)** → le modèle exploite *qui renseigne la profondeur* (proxy de programme/type
   de puits), pas la profondeur physique. Alerte de confondeur.
5. **Variable AQS (PM2.5/NO2/CO) importante** → confondeur d'urbanité déguisé en « dépôt
   atmosphérique ». À rejeter comme explication mécaniste.
6. **`n_geotracker_within_50km` >> `within_1km/3km`** → le modèle s'appuie sur la
   **densité régionale** (urbanité) et non sur la **proximité de panache** ; downgrade
   mécaniste, garder seulement les petits rayons.
7. **Variable météo instantanée (humidité de sol du mois, temp, runoff) importante** →
   incohérent avec l'inertie pluri-décennale d'une nappe → proxy de **saison/dérive
   d'échantillonnage**, pas de recharge réelle.
8. **Tout signe inversé par rapport à la section « attendu »** (p. ex. profondeur ↑ → P ↑,
   distance source ↑ → P ↑) = drapeau rouge à investiguer (confondeur, censure, ou
   inversion de variable).

**Test transversal recommandé** (relève de l'éval, mais je le pose ici comme garde-fou
mécaniste) : comparer SHAP / importance **en CV aléatoire vs CV spatiale par blocs**.
Toute variable dont l'importance **s'effondre en CV spatiale** prédisait via la structure
de carte, pas via un mécanisme → à traiter comme artefact.

---

## 8. Limites de domaine à porter au rapport final

- **Sources PFAS sous-représentées** (pas d'AFFF/militaire, STEP/biosolides, décharges,
  fluorochimie/électronique) : le modèle ne pourra pas « identifier la source » au sens
  physique ; il approchera la contamination via des proxys d'urbanité/industrialité. À
  énoncer comme limite n°1.
- **Sol SSURGO de surface ≠ zone non saturée profonde** : la rétention réelle se joue sur
  toute la colonne jusqu'à la nappe, mal décrite ici.
- **Pas de sens d'écoulement / piézométrie** : le transport advectif directionnel n'est
  pas représenté ; la distance euclidienne sur/sous-estime selon amont/aval.
- **Censure analytique** omniprésente (PFAS et cocontaminants) : risque de faux positifs
  de cible (MCL 4 ng/L ≈ LD) et de features qui encodent le panel plus que la chimie.
- **Dérive temporelle d'échantillonnage** : volume et probablement prévalence varient avec
  l'année → l'année est un confondeur, pas un mécanisme.

---

## RÉSUMÉ POUR LE FIL PRINCIPAL

**Verdict par famille de features :**
- **A (lat/lon)** : REFORMULER → porter par le graphe k-NN, tester avec/sans en nœud.
- **B (admin/hydrogéo)** : `dwr_basin`/`sgma_subbasin` GARDER (proxy aquifère) ;
  `county`/`regional_board`/`region_office` MÉFIANCE-CONFONDEUR (frontières
  administratives = programme, pas physique).
- **C (puits)** : `well_depth_ft` GARDER — **seul signal mécaniste fort** (profond → moins
  vulnérable, corr −0,35), mais surveiller que SHAP n'attribue pas au *manque* ;
  `gm_well_category` GARDER avec prudence (monitoring = biais d'enquête).
- **D (sources géotracker)** : GARDER mais **INCOMPLET** — typologie biaisée vers sources
  non-PFAS ; rayons 1–3 km mécanistes, 50 km = confondeur d'urbanité.
- **E (cocontaminants)** : **MÉFIANCE-CONFONDEUR forte** — majoritairement proxys de
  panel analytique/site cleanup (VOC, BTEX, **fréons CFC ≠ PFAS**, pesticides) ; ne garder
  qu'un noyau *contexte* {nitrate, TDS, SO4, Mn, Fe, As}, jamais le bloc VOC ; tester un
  modèle sans cocontaminants.
- **F (sol)** : GARDER noyau {OM%, ksat, texture, AWC}, ÉLAGUER sous-classes
  granulo/gradation (colinéaires, très manquantes, surface ≠ profondeur). Effet 2e ordre.
- **G (climat/hydro)** : REFORMULER → garder un proxy de **recharge moyenne/aridité**,
  ÉLAGUER météo instantanée (`soil_moi_*`, temp, runoff, snowpack) = proxy de saison.
- **H (air AQS)** : ÉLAGUER quasi tout — dépôt atmosphérique non détectable en nappe à
  cette échelle ; PM/NO2/CO = confondeur d'urbanité.
- **I (temporel)** : contrôle de dérive, pas prédicteur causal.
- **J (provenance)** : NE PAS utiliser comme feature (confondeur de design WB_CLEANUP) ;
  audit/stratification seulement.

**Top 3 sources PFAS manquantes à ajouter (priorité décroissante) :**
1. **AFFF / bases militaires & DoD / fire-training areas / aéroports avec activité
   incendie** (source n°1 ; signature PFOS+PFHxS+6:2FTS+8:2FTS+FOSAA, exactement le
   cluster T2).
2. **STEP / épandage de biosolides / irrigation par eaux recyclées** (diffus, CA ;
   signature PFCA courtes).
3. **Décharges/landfills (lixiviats)** et **fluorochimie/électronique-semi-conducteurs
   (Silicon Valley) / papeteries**.

**Corrections de seuils de cible :**
- **T1a HI : seuils EPA 2024 CORRECTS** (PFHxS/10, PFNA/10, HFPO-DA/10, PFBS/2000 ; MCL
  PFOA/PFOS=4). Aucune erreur. **Mais** ajouter le **garde-fou de détection** : ne
  déclencher un dépassement que si l'analyte est `detected` (le MCL 4 ng/L est de l'ordre
  des limites de rapport → risque de faux positifs de cible sur non-détects à LD élevée).
- **T2 seuil 2,0 ng/L** : OK mais à étiqueter « seuil analytique/quantification », **non
  sanitaire** (hétérogène selon analyte).
- **T1b Σ>70 ng/L** : OK en secondaire, mais c'est l'ancien Health Advisory PFOA+PFOS
  (2016) appliqué à la somme des 31 → plus permissif, **non réglementaire actuel** ; à
  étiqueter comme repère historique, garder T1a en primaire.

**Graphe GNN :** k-NN spatial **OUI mais plafonné en distance (~1–2 km)** sinon fuite
spatiale (autocorrélation retombe à la base dès 1–5 km) ; privilégier **arêtes
« même source/panache »** et **amont→aval** si écoulement disponible ; arêtes
« même sous-bassin » OK seulement si la CV par blocs est ⊇ bassins ; **proscrire** les
arêtes par county/board/dataset et tout k-NN sans plafond. Rapporter l'écart d'importance
CV aléatoire vs spatiale comme test d'artefact.
