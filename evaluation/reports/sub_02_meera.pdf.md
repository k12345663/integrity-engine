# Integrity report: `sub_02_meera.pdf`

**Overall risk: HIGH**  
Modality: pdf | Status: ok

| Signal | Score |
|---|---|
| Internal peer similarity | 45.2% |
| External (web/reference) match | 0.0% |
| AI-content likelihood (advisory) | 1.3% |

## Summary for reviewers

45.2% of the original text overlaps with other submissions: sub_01_aarav.pdf (45.2%). Strongest evidence: p.2 <-> sub_01_aarav.pdf p.2 (verbatim, cos 0.95). 147 words of shared template/prompt text were detected and excluded from scoring. AI-content likelihood 1%; strongest signal: log_perplexity; advisory only - corroborate before acting. Overall risk: HIGH.

## Peer matches (same modality)

### 1. Peer match: `sub_01_aarav.pdf` - verbatim

Chunk cosine **0.95** | sentence evidence **1.00** (sentence cosine) | word overlap (Jaccard) 1.00 | pdf -> pdf

**This submission** - page 2, ¶8

> **Pune Municipal Corporation has already capped dockless fleets in the old city wards, and a stricter parking rule could force us to build designated parking bays at our own cost. We would reduce this exposure by co-funding thirty marked parking hubs with the corporation and sharing anonymised trip data with its traffic cell, which turns the regulator into a partner rather than an opponent. The second risk is battery degradation. Swap stations cycle batteries harder than overnight depot charging, and if pack life falls below eight hundred cycles the station economics break down. A leasing contract with the battery supplier, priced per kilowatt-hour delivered rather than per pack, moves this risk onto the party that understands the chemistry best.**

**Matched source** - `sub_01_aarav.pdf`, page 2, ¶10

> **Pune Municipal Corporation has already capped dockless fleets in the old city wards, and a stricter parking rule could force us to build designated parking bays at our own cost. We would reduce this exposure by co-funding thirty marked parking hubs with the corporation and sharing anonymised trip data with its traffic cell, which turns the regulator into a partner rather than an opponent. The second risk is battery degradation. Swap stations cycle batteries harder than overnight depot charging, and if pack life falls below eight hundred cycles the station economics break down. A leasing contract with the battery supplier, priced per kilowatt-hour delivered rather than per pack, moves this risk onto the party that understands the chemistry best.**

<details><summary>Sentence-level alignment</summary>

| # | This submission | Source | score (method) |
|---|---|---|---|
| 1 | Pune Municipal Corporation has already capped dockless fleets in the old city wards, and a stricter parking rule could force us to build designated parking bays at our own cost. | Pune Municipal Corporation has already capped dockless fleets in the old city wards, and a stricter parking rule could force us to build designated parking bays at our own cost. | 1.00 (cosine) |
| 2 | We would reduce this exposure by co-funding thirty marked parking hubs with the corporation and sharing anonymised trip data with its traffic cell, which turns the regulator into a partner rather than an opponent. | We would reduce this exposure by co-funding thirty marked parking hubs with the corporation and sharing anonymised trip data with its traffic cell, which turns the regulator into a partner rather than an opponent. | 1.00 (cosine) |
| 3 | The second risk is battery degradation. | The second risk is battery degradation. | 1.00 (cosine) |
| 4 | Swap stations cycle batteries harder than overnight depot charging, and if pack life falls below eight hundred cycles the station economics break down. | Swap stations cycle batteries harder than overnight depot charging, and if pack life falls below eight hundred cycles the station economics break down. | 1.00 (cosine) |
| 5 | A leasing contract with the battery supplier, priced per kilowatt-hour delivered rather than per pack, moves this risk onto the party that understands the chemistry best. | A leasing contract with the battery supplier, priced per kilowatt-hour delivered rather than per pack, moves this risk onto the party that understands the chemistry best. | 1.00 (cosine) |

</details>

## Shared template text (excluded from scores)

These passages appear in many submissions (case prompt, instructions, citations) and are not counted.

- (138 words, in 5 documents, page 1, ¶2) Case Prompt - Inter-University Strategy Challenge 2026, Round 2. VoltRide Mobility operates 1,800 dockless electric scooters in Pune and completed 410,000 rides in the last financial year. The founders must decide whether to expand into three tier-2 cities (Nagpur, Indore and Coimbatore) within e...
- (9 words, in 5 documents, page 2, ¶12) VoltRide Mobility Case Exhibits 1-4, Inter-University Strategy Challenge 2026.

## AI-content analysis

Method: logistic regression calibrated on HC3 + SAGE (GPT-4.1 / Claude 3.7 / Gemini 2.5) (CV AUROC 0.91).

| Signal | Contribution (log-odds; + = more AI-like) |
|---|---|
| log_perplexity | -3.11 |
| burstiness | -0.41 |
| marker_rate | -1.19 |
| classifier | -0.14 |

Raw signals: GPT-2 perplexity 104.9, burstiness 0.59, assistant-phrase rate 0.00/100 words, classifier 0.0002.

## Scoring basis

- Words scored: 263 (template words excluded: 147, reference-list words excluded: 12)
- Document metadata: {"size_bytes": 4556, "pages": 2, "header_footer_blocks_removed": 4, "extraction_seconds": 0.0}
