# Integrity report: `sub_01_aarav.pdf`

**Overall risk: HIGH**  
Modality: pdf | Status: ok

| Signal | Score |
|---|---|
| Internal peer similarity | 44.3% |
| External (web/reference) match | 0.0% |
| AI-content likelihood (advisory) | 0.1% |

## Summary for reviewers

44.3% of the original text overlaps with other submissions: sub_02_meera.pdf (26.6%), pitch_10_sanjay.mp4 (17.7%). Strongest evidence: p.2 <-> sub_02_meera.pdf p.2 (verbatim, cos 0.95); p.2 <-> pitch_10_sanjay.mp4 00:00:04 - 00:00:24 (near-verbatim, cos 0.82). 147 words of shared template/prompt text were detected and excluded from scoring. AI-content likelihood 0%; strongest signal: burstiness; advisory only - corroborate before acting. Overall risk: HIGH.

## Peer matches (same modality)

### 1. Peer match: `sub_02_meera.pdf` - verbatim

Chunk cosine **0.95** | sentence evidence **1.00** (sentence cosine) | word overlap (Jaccard) 1.00 | pdf -> pdf

**This submission** - page 2, ¶10

> **Pune Municipal Corporation has already capped dockless fleets in the old city wards, and a stricter parking rule could force us to build designated parking bays at our own cost. We would reduce this exposure by co-funding thirty marked parking hubs with the corporation and sharing anonymised trip data with its traffic cell, which turns the regulator into a partner rather than an opponent. The second risk is battery degradation. Swap stations cycle batteries harder than overnight depot charging, and if pack life falls below eight hundred cycles the station economics break down. A leasing contract with the battery supplier, priced per kilowatt-hour delivered rather than per pack, moves this risk onto the party that understands the chemistry best.**

**Matched source** - `sub_02_meera.pdf`, page 2, ¶8

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

## Cross-modal matches (video/audio <-> document)

### 2. Cross-modal match: `pitch_10_sanjay.mp4` - near-verbatim (light edits)

Chunk cosine **0.82** | sentence evidence **0.95** (sentence cosine) | word overlap (Jaccard) 0.69 | pdf -> media

**This submission** - page 2, ¶8

> **Our recommendation is to stay in Pune for the next** eighteen **months and build a dense** battery-swapping **network before chasing new cities. Right now each scooter spends almost** forty **minutes a day** off **the road waiting for a van to collect and recharge it,** and that idle time is the biggest reason the fleet earns so little. Twenty **swap stations near the busiest pickup zones would lift** utilisation **from 5.1 to 6.4 rides per scooter per day.**

**Matched source** - `pitch_10_sanjay.mp4`, video 00:00:04 - 00:00:24

> **Our recommendation is to stay in Pune for the next** 18 **months and build a dense** battery swapping **network before chasing new cities. Right now each scooter spends almost** 40 **minutes a day** of **the road waiting for a van to collect and recharge it.** 20 **swap stations near the busiest pickup zones would lift** utilization **from 5.1 to 6.4 rides per scooter per day.**

<details><summary>Sentence-level alignment</summary>

| # | This submission | Source | score (method) |
|---|---|---|---|
| 1 | Our recommendation is to stay in Pune for the next eighteen months and build a dense battery-swapping network before chasing new cities. | Our recommendation is to stay in Pune for the next 18 months and build a dense battery swapping network before chasing new cities. | 0.99 (cosine) |
| 2 | Right now each scooter spends almost forty minutes a day off the road waiting for a van to collect and recharge it, and that idle time is the biggest reason the fleet earns so little. | Right now each scooter spends almost 40 minutes a day of the road waiting for a van to collect and recharge it. | 0.88 (cosine) |
| 3 | Twenty swap stations near the busiest pickup zones would lift utilisation from 5.1 to 6.4 rides per scooter per day. | 20 swap stations near the busiest pickup zones would lift utilization from 5.1 to 6.4 rides per scooter per day. | 0.99 (cosine) |

</details>

## Shared template text (excluded from scores)

These passages appear in many submissions (case prompt, instructions, citations) and are not counted.

- (138 words, in 5 documents, page 1, ¶2) Case Prompt - Inter-University Strategy Challenge 2026, Round 2. VoltRide Mobility operates 1,800 dockless electric scooters in Pune and completed 410,000 rides in the last financial year. The founders must decide whether to expand into three tier-2 cities (Nagpur, Indore and Coimbatore) within e...
- (9 words, in 5 documents, page 2, ¶14) VoltRide Mobility Case Exhibits 1-4, Inter-University Strategy Challenge 2026.

## AI-content analysis

Method: logistic regression calibrated on HC3 + SAGE (GPT-4.1 / Claude 3.7 / Gemini 2.5) (CV AUROC 0.91).

| Signal | Contribution (log-odds; + = more AI-like) |
|---|---|
| log_perplexity | -2.67 |
| burstiness | -3.41 |
| marker_rate | -1.19 |
| classifier | -0.14 |

Raw signals: GPT-2 perplexity 85.5, burstiness 1.52, assistant-phrase rate 0.00/100 words, classifier 0.0003.

## Scoring basis

- Words scored: 447 (template words excluded: 147, reference-list words excluded: 12)
- Document metadata: {"size_bytes": 5176, "pages": 2, "header_footer_blocks_removed": 4, "extraction_seconds": 0.48}
