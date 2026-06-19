# REPORT — Profilage du jeu de données PFAS / eaux souterraines (Californie)

> Étape 1 du projet (mode prédictif strict, cf. `CLAUDE.md`). Partir de zéro, aucune
> hypothèse préalable : tout fait ci-dessous est établi par analyse.
> Reproductible : `python3 experiments/profilage/profile.py` (graine 42) régénère
> `experiments/profilage/profile_metrics.json`, source de tous les chiffres.

- **Données** : `data/CA-PFAS-ASGWS.parquet` — **46 338 lignes × 201 colonnes**.
- **Auteur** : agent `data-analyst` (rapporté au fil principal, persisté ici).

---

## 0. Résumé exécutif

1. **Granularité** : 1 ligne = **un puits × un événement de prélèvement**. 11 333
   puits uniques, 46 338 événements ; clé `(gm_well_id, collection_date)` unique,
   **0 doublon**. 4 349 puits ont plusieurs prélèvements (max 139). **⇒ tout split
   doit grouper par `gm_well_id`** (sinon fuite par pseudo-réplicats temporels).
2. **Le jeu contient déjà les cibles et labels pré-calculés** (artefacts dérivés des
   mesures PFAS, PAS du contexte) : ils sont en blocklist.
3. **Fuite propre, bien isolée** : les 16 colonnes les plus corrélées à T1 sont
   toutes des `*_detected`. Aucune fuite cachée dans le contexte, aucune sentinelle.
4. **Autocorrélation spatiale forte** : Moran's I (T1, niveau puits, kNN k=8) =
   **0,426** ; concordance T1 = **76,7 % à 0-1 km vs 54 % de base**. **⇒ CV spatiale
   par blocs obligatoire.**
5. **T1 primaire (EPA 2024) prévalence 45,6 %** (quasi-équilibré) ; **T1 secondaire
   (Σ>70 ng/L) prévalence 24,8 %**.
6. **T2 : 15 labels curés** retenus ; co-occurrences fortes exploitables.

---

## 1. Structure & types

- **46 338 × 201**, ~71 Mo en mémoire.
- **Granularité (par comptages d'unicité)** : `gm_well_id` = 11 333 valeurs ;
  `(gm_well_id, collection_date)` = 46 338 = nb de lignes, **0 doublon**. Donc une
  ligne = **un événement de prélèvement d'un puits**. 4 349 puits multi-prélèvements
  (max 139 prélèvements pour un même puits).
- **Couverture temporelle** : 2016 → 2026, fortement concentrée 2019-2025 (35 lignes
  en 2016 → 10 025 en 2025). **Dérive d'échantillonnage** dans le temps (volume ET
  potentiellement prévalence) — à surveiller pour toute feature temporelle.
- **À nettoyer** :
  - Constantes : `label_PFEESA`, `label_PFMBA`, `label_PFMPA` (toujours 0),
    `pfas_class_assignment` (JSON constant).
  - Quasi-constante / quasi-vide : `cocontam_xylenes` (99,8 % manquant).
  - Colonnes **dupliquées** : `cocontam_tmb124 == cocontam_dce12c == cocontam_btbzt`
    (à dédoublonner).

## 2. Distributions & valeurs manquantes

- **Aucune sentinelle** de type -999 détectée.
- **Encodage des non-détects** : les colonnes `*_ngL` stockent la **limite de
  rapport** quand l'analyte n'est pas détecté (ex. PFOA non détecté : médiane
  1,0 ng/L, max 227 ng/L par dilution). **⇒ `*_detected` est le drapeau de censure
  faisant autorité** ; une concentration élevée peut être une limite de rapport, pas
  une détection. **Point critique à croiser systématiquement avec `*_detected`.**
- Manquants notables côté contexte : `well_depth_ft` **94,5 % manquant** ;
  `cocontam_xylenes` 99,8 %. Géotracker (distances/comptes) **0 % manquant**.

## 3. Candidats-cibles & analytes PFAS

- **31 analytes PFAS** détectés sous forme `*_ngL` (concentrations), chacun doublé
  d'un `*_detected` (flag) et d'un `label_*` (dépassement). Analytes : PFOS, PFHxS,
  PFOA, PFBS, PFHxA, FTS_6_2, PFHpA, PFBA, FTS_8_2, PFPeA, PFNA, PFPeS, NEtFOSAA,
  NMeFOSAA, PFDA, … (+ 3 analytes constants à 0 : PFEESA, PFMBA, PFMPA).
- **Sémantique établie PAR PREUVE** (pas par le nom seul) :
  - `sum_pfas_ngL` = Σ des 31 `*_ngL` → **corr = 1,000000, écart max = 0**.
  - `target_sum_gt70` = (`sum_pfas_ngL` > 70) → **concordance 100 %**.
  - `label_X` = (`X_ngL` > 2,0 ng/L) → **vérifié sur les 31 analytes, 100 %**.

## 4. Fuite de cible — blocklist justifiée (96 colonnes à EXCLURE des features)

| Groupe | n | Colonnes | Preuve de fuite |
|---|---|---|---|
| Concentrations | 31 | tous les `*_ngL` | mesure PFAS directe (= la cible) |
| Détection | 31 | tous les `*_detected` | indicateur de détection (corr ≤ 0,70 avec T1) |
| Labels | 31 | tous les `label_*` | = `ngL > 2,0` (dérivation exacte vérifiée, 100 %) |
| Sommes/cibles | 2 | `sum_pfas_ngL`, `target_sum_gt70` | dérivées exactes (corr 1,0) |
| Artefact | 1 | `pfas_class_assignment` | JSON constant dérivé du profil PFAS |

- **Radar de fuite** : les **16 colonnes les plus corrélées à T1 sont toutes des
  `*_detected`** (corr 0,15-0,70). Premier vrai contexte au classement :
  `well_depth_ft` (corr −0,35, mais 94,5 % manquant), puis
  `n_geotracker_within_1km` (0,18). **Aucune fuite cachée dans le contexte.**
- `gm_well_id` **n'est pas une feature** mais la **clé de groupe** des splits.
- **Zone de vigilance (non-fuite mais à auditer SHAP)** : les 44 `cocontam_*`
  (proxys possibles de « échantillon analysé en labo ») et `gm_dataset_name`
  (confondeur de design : p. ex. WB_CLEANUP cible des sites déjà pollués).

## 5. Définitions de cible proposées

### T1 — binaire (dépassement réglementaire)

- **T1a (primaire, EPA 2024)** : `PFOA>4 OU PFOS>4 OU HazardIndex≥1`, avec
  **HI = PFHxS/10 + PFNA/10 + HFPO_DA/10 + PFBS/2000** (concentrations-seuils en
  ng/L). **Hypothèse de seuils à VALIDER par l'expert hydro.**
  → **Prévalence 45,6 % (21 154 positifs), déséquilibre 1,2** (quasi-équilibré).
- **T1b (secondaire)** : `sum_pfas_ngL > 70` → **prévalence 24,8 % (11 490 pos)**.
- Accord T1a/T1b = 77,5 %.
- **Procédure** : calculer la cible depuis les `*_ngL`, **puis retirer ces colonnes**
  (blocklist). Croiser avec `*_detected` pour ne pas déclencher un dépassement sur
  une limite de rapport élevée sans détection (cf. §2).

### T2 — multilabel (seuils RÉGLEMENTAIRES par composé — voir [T2_TARGETS.md](T2_TARGETS.md))

> ⚠️ **Définition T2 mise à jour (2026-06-19).** La définition uniforme initiale
> (`label_X = X_ngL > 2,0`) est **remplacée** par un schéma **hybride** : seuil
> **réglementaire EPA 2024** pour les composés réglementés (PFOA 4, PFOS 4, PFHxS 10,
> PFNA 10, HFPO‑DA 10 ng/L), **seuil analytique 2,0 ng/L en repli** sinon, **+ garde-fou
> de détection (eval C1)**. Spécification complète : **[T2_TARGETS.md](T2_TARGETS.md)**.

- **Découverte critique** : les colonnes `label_*` fournies (`ngL > 2,0` sans garde-fou)
  sont **gonflées par la censure** (limites de rapport > 2 ng/L sur non-détects comptées
  comme dépassements) — p. ex. FTS_6_2 0,279 → **0,038** après garde-fou (détection 3,9 %).
  **⇒ recalculer T2 depuis `*_ngL` + `*_detected`, ne jamais utiliser `label_*` brut.**
- **Cœur T2 = 9 labels exploitables (prév. ≥ 5 %)** : PFOS 0,393 · PFBS 0,373 ·
  PFHxA 0,368 · PFOA 0,340 · PFHpA 0,258 · PFBA 0,229 · PFPeA 0,227 · PFHxS 0,146 ·
  PFPeS 0,088. **+ PFNA** optionnel (réglementé, rare 0,025). HFPO‑DA exclu (~0 %).
- **Co-occurrences fortes** : PFBA~PFPeA 0,80 · PFHxA~PFOA 0,74 · PFOA~PFOS 0,71 …
- **2,56 labels positifs/ligne** ; **47,8 % de lignes entièrement négatives**.

## 6. Structure spatiale & autocorrélation

- **Colonnes de localisation** : `latitude`, `longitude`, + hiérarchie admin
  (`county`, `regional_board`, `dwr_region`, `dwr_basin`, `sgma_*`).
- **Autocorrélation de T1 (niveau puits)** :
  - **Moran's I (kNN k=8) = 0,426** (vs −0,0003 sous H0 de permutation).
  - Concordance T1 entre paires : **76,7 % à 0-1 km vs 54 % de base** ; retombe à la
    base vers 5-20 km (portée ~quelques km).
- **⇒ Un split aléatoire gonflera les scores. CV spatiale par blocs obligatoire**,
  rapportée systématiquement à côté du split aléatoire, avec l'écart.

### Blocs spatiaux proposés

- **Schéma principal** : **KMeans k=8 sur (lat, lon) au niveau puits** (blocs de
  476 à 3 163 puits ; prévalence T1 par bloc 0,21-0,59).
- **Variante conservatrice** : KMeans k=5.
- **Alternative interprétable** : **LeaveOneRegionOut** par `regional_board` (9) ou
  `dwr_region` (10).
- **Toujours grouper par `gm_well_id` d'abord** (les prélèvements d'un même puits
  restent du même côté du split).

## 7. Espace de features candidat (NON fuitant, ~105 colonnes, par famille)

- **A. Localisation pure** ⚠️ tester avec/sans, à porter de préférence par le graphe
  k-NN plutôt qu'en feature de nœud : `latitude`, `longitude`.
- **B. Admin / hydrogéo (catégoriel)** : `county` (58), `regional_board` (9),
  `dwr_region` (10), `dwr_basin` (239), `sgma_subbasin_name` (237), `sgma_basin_name`,
  `sgma_region_office`. → one-hot faible cardinalité ; encodage haute cardinalité
  **out-of-fold uniquement**.
- **C. Puits** : `gm_well_category` (5, one-hot) ; `well_depth_ft` (94,5 % manquant →
  imputation par bassin + indicateur `well_depth_ft_missing`).
- **D. Sources (géotracker, 0 % manquant)** : `dist_geotracker_km`,
  `nearest_geotracker_type` (4 : Chrome Plater / Bulk Terminal / Airport / Refinery),
  `n_geotracker_within_{1,3,10,50}km`. → log1p.
- **E. Cocontaminants (44, admissibles, AUDITER SHAP)** : nitrate, TCE/PCE, MTBE, As,
  TDS, Mn, Fe… → log1p + indicateur de manque ; dédoublonner le cluster identique ;
  retirer `cocontam_xylenes`.
- **F. Sol SSURGO (26)** : texture, `soil_om_pct`, `soil_ph`, `soil_ksat_um_s`,
  `soil_awc_cm_cm`… → standardiser + imputer.
- **G. Climat / hydro GLDAS (~12)** : rainfall, ET, runoff, humidités de sol, temp,
  snowpack.
- **H. Air AQS (8)** : PM2.5/PM10/NO2/SO2/vent/humidité/ozone/CO — pertinence faible,
  candidats à l'élagage.
- **I. Temporel** : dériver année + saison de `collection_date` (⚠️ dérive de
  prévalence dans le temps).
- **J. Provenance** : `gm_dataset_name` comme variable de **contrôle** (auditer, ne
  pas laisser le modèle s'appuyer sur le design d'échantillonnage).

**Transformations transverses** : imputation **toujours intra-fold** ; indicateurs de
manque (>20 %) ; log1p (distances / comptes / cocontaminants) ; standardisation ;
encodage catégoriel **out-of-fold**.

---

## 8. Limites & questions ouvertes

1. **Seuils du Hazard Index T1a** = hypothèse réglementaire à valider (concentrations-
   seuils PFHxS/PFNA/HFPO-DA/PFBS).
2. **Censure** : risque qu'un `X_ngL > seuil` se déclenche sur une limite de rapport
   élevée sans détection réelle — croiser avec `X_detected`.
3. **Cocontaminants** : causes physiques OU simples proxys de « échantillon analysé » ?
4. **Dérive temporelle** de l'échantillonnage (volume et prévalence) → biais possible.
5. `well_depth_ft` très prédictif (−0,35) mais 94,5 % manquant → valeur conditionnée à
   l'imputation.

## 9. Recommandations pour l'étape suivante

- Faire **critiquer cet espace de features par `hydro-domain-expert`** (plausibilité
  mécaniste) — en cours.
- Faire **valider le protocole de CV spatiale + blocklist par `eval-methodologist`**
  avant tout run.
- Geler la **blocklist (96 col.)**, la **clé de groupe `gm_well_id`** et le **schéma
  de blocs KMeans k=8** comme contrat partagé pour les agents de modélisation.

---

## 10. Références réglementaires

Seuils de cible T1a/T2 fondés sur la réglementation fédérale (références détaillées et
comparaison des cadres dans [T2_TARGETS.md](T2_TARGETS.md) §7) :

- **US EPA — PFAS NPDWR, règle finale 10 avril 2024** (MCL : PFOA 4, PFOS 4, PFHxS 10,
  PFNA 10, HFPO‑DA 10 ng/L ; Hazard Index = 1) — Federal Register doc. 2024‑07773 :
  <https://www.federalregister.gov/documents/2024/04/26/2024-07773/pfas-national-primary-drinking-water-regulation>
  · page programme : <https://www.epa.gov/sdwa/and-polyfluoroalkyl-substances-pfas>
- **US EPA — reconsidération 2025‑2026** (maintien PFOA/PFOS ; rescision proposée de
  PFHxS/PFNA/HFPO‑DA/Hazard Index) :
  <https://www.epa.gov/newsreleases/epa-announces-it-will-keep-maximum-contaminant-levels-pfoa-pfos>
- **California State Water Board — Notification/Response Levels PFAS** (jurisdiction du
  jeu de données) :
  <https://www.waterboards.ca.gov/drinking_water/certlic/drinkingwater/pfas.html>

> URL consultées le 2026‑06‑19.

---

### Artefacts

- `experiments/profilage/profile.py` — script déterministe (graine 42).
- `experiments/profilage/profile_metrics.json` — tous les chiffres ci-dessus.
