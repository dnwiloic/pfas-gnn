# T2 — Définition des cibles multilabel (seuils réglementaires par composé)

> Décision utilisateur (2026-06-19) : **schéma HYBRIDE** — seuil **réglementaire**
> pour les composés réglementés, **seuil analytique 2,0 ng/L en repli** pour les
> autres ; cadre réglementaire = **US EPA 2024 (MCL fédéraux)**.
> Reproductible : `python3 experiments/profilage/t2_hybrid_final.py` (graine 42) →
> `experiments/profilage/t2_hybrid_metrics.json`.
> Remplace la définition T2 « uniforme 2,0 ng/L » du profilage initial
> (cf. [REPORT.md](REPORT.md) §5).

## 1. Schéma de seuils

`label_X = 1  ⟺  (X_ngL > seuil_X)  ET  X_detected`  (garde-fou de détection, eval **C1**).

| Composé | Réglementé | Seuil (ng/L) | Source |
|---|---|---|---|
| PFOA | oui | **4** | EPA MCL 2024 |
| PFOS | oui | **4** | EPA MCL 2024 |
| PFHxS | oui | **10** | EPA MCL 2024 |
| PFNA | oui | **10** | EPA MCL 2024 |
| HFPO‑DA (GenX) | oui | **10** | EPA MCL 2024 |
| **tous les autres** | non | **2,0** | seuil analytique de quantification (repli) |

Note : PFBS et PFHxA n'ont **pas** de MCL individuel EPA (PFBS = Hazard Index
uniquement) → ils retombent sur le repli analytique 2,0 ng/L. Caveat réglementaire :
EPA a proposé (mai 2026) de **rescinder** PFHxS/PFNA/HFPO‑DA ; on conserve les MCL
2024 comme référence (cohérence avec la cible T1a) — à ré-évaluer si la règle finale
change.

## 2. Le garde-fou de détection est décisif (pas seulement pour T1)

Les colonnes `label_*` **fournies dans le jeu** valent `X_ngL > 2,0` **sans** garde-fou
de détection → elles sont **gonflées par la censure** (non-détects portant une limite
de rapport > 2 ng/L comptés comme dépassements). Effet mesuré :

| Composé | détection % | `label_*` fourni (sans garde) | détecté-dépassement @2,0 |
|---|---|---|---|
| FTS_6_2 | 3,9 % | 0,279 | **0,038** |
| FTS_8_2 | 1,7 % | 0,244 | **0,017** |
| NEtFOSAA | 0,6 % | 0,095 | **0,005** |
| NMeFOSAA | 0,2 % | 0,092 | **0,002** |
| PFBA | 23,9 % | 0,274 | 0,229 |
| PFHpA | 27,9 % | 0,279 | 0,258 |

⇒ **Recalculer T2 à partir des `*_ngL` + `*_detected`, ne jamais utiliser les
`label_*` bruts** (ils sont en blocklist au même titre que les mesures).

## 3. Jeu de labels retenu (schéma hybride EPA + garde-fou)

**15 labels** survivent au filtre de mesurabilité (mesuré ≥ 50 %), dont **9 exploitables**
(prévalence ≥ 5 %) :

| Label | Seuil | Source | Prévalence | mesuré % | exploitable |
|---|---|---|---|---|---|
| PFOS | 4 | EPA MCL | **0,393** | 99,8 % | ✅ |
| PFBS | 2,0 | analytique | **0,373** | 95,1 % | ✅ |
| PFHxA | 2,0 | analytique | **0,368** | 95,8 % | ✅ |
| PFOA | 4 | EPA MCL | **0,340** | 99,8 % | ✅ |
| PFHpA | 2,0 | analytique | **0,258** | 97,3 % | ✅ |
| PFBA | 2,0 | analytique | **0,229** | 55,7 % | ✅ |
| PFPeA | 2,0 | analytique | **0,227** | 55,5 % | ✅ |
| PFHxS | 10 | EPA MCL | **0,146** | 95,2 % | ✅ |
| PFPeS | 2,0 | analytique | **0,088** | 56,0 % | ✅ |
| PFNA | 10 | EPA MCL | 0,025 | 97,4 % | ⚠️ rare réglementé |
| PFDA | 2,0 | analytique | 0,045 | 96,1 % | ✗ <5 % |
| FTS_6_2 | 2,0 | analytique | 0,038 | 55,8 % | ✗ censure |
| FTS_8_2 | 2,0 | analytique | 0,017 | 55,8 % | ✗ censure |
| NEtFOSAA | 2,0 | analytique | 0,005 | 58,0 % | ✗ censure |
| NMeFOSAA | 2,0 | analytique | 0,002 | 57,9 % | ✗ censure |

- **HFPO‑DA** : détecté à 0,1 % (quasi absent des nappes CA) → **abandonné** (constant).
- **PFNA** : réglementé mais rare (2,5 %) sous MCL 10 ; **conservé en label optionnel**,
  à traiter en label déséquilibré (rééchantillonnage par label / pondération), décision
  finale laissée au `multilabel-specialist`.

**Ensemble T2 recommandé :**
- **Cœur (9 labels, prév. ≥ 5 %)** : PFOS, PFBS, PFHxA, PFOA, PFHpA, PFBA, PFPeA,
  PFHxS, PFPeS.
- **+ PFNA** en option (réglementé, rare).

## 4. Caractéristiques de la cible multilabel

- **2,56 labels positifs/ligne** en moyenne (cœur 9 + PFNA) ; **47,8 % de lignes
  entièrement négatives** (vs 25 % sous la définition gonflée — la censure créait de
  faux positifs).
- **Co-occurrences fortes exploitables** (chaînes/sources communes), p. ex. :
  PFBA~PFPeA +0,80 · PFHxA~PFOA +0,74 · PFHpA~PFOA +0,74 · PFBS~PFHxA +0,72 ·
  PFOA~PFOS +0,71 · PFBS~PFOA +0,65 · PFHxS~PFPeS +0,64. Structure de dépendance de
  labels à exploiter (chaînes de classifieurs, etc.).
- Les anciennes co-occurrences « précurseurs » (NEtFOSAA~NMeFOSAA 0,98 ;
  FTS_6_2~FTS_8_2 0,88) **disparaissent du cœur** : elles reposaient sur des labels
  désormais jugés non fiables (censure) — cohérent, on ne s'appuie plus dessus.

## 5. Lien avec T1 et conséquences

- Les 5 composés réglementés de T2 (PFOA, PFOS, PFHxS, PFNA, HFPO‑DA) sont exactement
  les briques de **T1a** (EPA 2024 : PFOA>4 ∨ PFOS>4 ∨ HI≥1). T2 « réglementé » est
  donc la **décomposition par-composé** du dépassement T1a, complétée par les labels
  analytiques (PFBS, PFHxA, PFHpA, PFBA, PFPeA, PFPeS) qui élargissent au-delà du
  périmètre strictement réglementaire.
- À interpréter comme tel dans la rédaction : T2 mêlant seuils réglementaires (santé)
  et seuils analytiques (quantification) — sémantique hétérogène **assumée et documentée**.

## 6. Contrat figé pour les agents aval

1. Construire T2 depuis `*_ngL` + `*_detected` selon le tableau §1 (jamais depuis
   `label_*` bruts).
2. Cœur = 9 labels ; PFNA optionnel (déséquilibré) ; HFPO‑DA exclu.
3. Garde-fou de détection (C1) obligatoire sur chaque label.
4. Mêmes splits groupés (`gm_well_id`) + CV spatiale par blocs que T1 (cf.
   [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md)).

## 7. Références réglementaires

**US EPA — PFAS National Primary Drinking Water Regulation (NPDWR), règle finale du
10 avril 2024.** MCL individuels (ng/L) : PFOA 4 · PFOS 4 · PFHxS 10 · PFNA 10 ·
HFPO‑DA (GenX) 10 ; Hazard Index = 1 pour les mélanges de PFNA, GenX, PFHxS, PFBS.
- Federal Register, 89 FR (26 avril 2024), doc. 2024‑07773 :
  <https://www.federalregister.gov/documents/2024/04/26/2024-07773/pfas-national-primary-drinking-water-regulation>
- Page programme EPA PFAS / SDWA :
  <https://www.epa.gov/sdwa/and-polyfluoroalkyl-substances-pfas>

**US EPA — reconsidération 2025‑2026.** Mai 2025 : intention de conserver les MCL
PFOA/PFOS (4 ng/L) et de **rescinder** PFHxS, PFNA, HFPO‑DA et le Hazard Index ;
règles proposées le 18 mai 2026 (maintien PFOA/PFOS, extension de délai possible à 2031).
- <https://www.epa.gov/newsreleases/epa-announces-it-will-keep-maximum-contaminant-levels-pfoa-pfos>
- Règle d'extension de conformité proposée :
  <https://www.epa.gov/sdwa/proposed-pfoa-and-pfos-compliance-extension-rule>

**California State Water Resources Control Board — Notification Levels (NL) /
Response Levels (RL), révisés oct. 2025** (jurisdiction du jeu de données ; non
retenus comme seuils T2 mais documentés pour le choix de cadre, cf.
`t2_regulatory.py`). Valeurs (ng/L) : PFOA NL 4 / RL 10 · PFOS NL 4 / RL 40 ·
PFHxS NL 3 / RL 10 · PFHxA NL 1000 / RL 10000 · PFBS NL 500 / RL 5000.
- Page PFAS — systèmes d'eau potable :
  <https://www.waterboards.ca.gov/drinking_water/certlic/drinkingwater/pfas.html>
- Chronologie PFAS Californie :
  <https://www.waterboards.ca.gov/pfas/ca_pfas_timeline.html>
- Avis d'émission de NL révisée (PFOS, 28 oct. 2025) :
  <https://waterboards.ca.gov/drinking_water/certlic/drinkingwater/documents/pfas/PFOS-revised-nl-issuance-20251028.pdf>

> Toutes les URL ont été consultées le 2026‑06‑19. Les valeurs réglementaires sont des
> faits externes susceptibles d'évoluer (cf. reconsidération fédérale en cours) ; le
> contrat T2 fige les MCL EPA 2024 comme référence et reste révisable si la règle
> finale change.

### Artefacts
- `experiments/profilage/t2_hybrid_final.py` — construction + curation (graine 42).
- `experiments/profilage/t2_hybrid_metrics.json` — prévalences, co-occurrences.
- `experiments/profilage/t2_regulatory.py` / `t2_regulatory_metrics.json` — comparaison
  des cadres (EPA / CA NL / CA RL) ayant motivé le choix.
