# Integrity report: `sub_06_ananya.pdf`

**Overall risk: HIGH**  
Modality: pdf | Status: ok

| Signal | Score |
|---|---|
| Internal peer similarity | 0.0% |
| External (web/reference) match | 50.8% |
| AI-content likelihood (advisory) | 0.5% |

## Summary for reviewers

No meaningful overlap with other submissions. 50.8% matches external reference material: Scooter-sharing system (50.8%). Strongest evidence: p.1 <-> Scooter-sharing system ¶1 (verbatim, cos 0.61). AI-content likelihood 0%; strongest signal: burstiness; advisory only - corroborate before acting. Overall risk: HIGH.

## External reference matches

### 1. External match: `Scooter-sharing system` - verbatim

Chunk cosine **0.61** | sentence evidence **1.00** (sentence cosine) | word overlap (Jaccard) 1.00 | pdf -> reference

**This submission** - page 1, ¶4

> **A scooter-sharing system or kicksharing system is a shared transport service in which e-scooters are made available to use for short-term rentals. E-scooters are typically "dockless," meaning that they do not have a fixed home location and are dropped off and picked up from certain locations in the service area. To rent a dockless e-scooter, users download a smartphone application. The application shows users a map of nearby e-scooters and enables them to unlock them. Scooters are equipped with built-in GPS chips and cellular connectivity which allows them to report their location in real time during a trip.**

**Matched source** - `Scooter-sharing system`, https://en.wikipedia.org/wiki/Scooter-sharing_system (¶1)

> **A scooter-sharing system or kicksharing system is a shared transport service in which e-scooters are made available to use for short-term rentals. E-scooters are typically "dockless," meaning that they do not have a fixed home location and are dropped off and picked up from certain locations in the service area. To rent a dockless e-scooter, users download a smartphone application. The application shows users a map of nearby e-scooters and enables them to unlock them. Scooters are equipped with built-in GPS chips and cellular connectivity which allows them to report their location in real time during a trip.**

<details><summary>Sentence-level alignment</summary>

| # | This submission | Source | score (method) |
|---|---|---|---|
| 1 | A scooter-sharing system or kicksharing system is a shared transport service in which e-scooters are made available to use for short-term rentals. | A scooter-sharing system or kicksharing system is a shared transport service in which e-scooters are made available to use for short-term rentals. | 1.00 (cosine) |
| 2 | E-scooters are typically "dockless," meaning that they do not have a fixed home location and are dropped off and picked up from certain locations in the service area. | E-scooters are typically "dockless," meaning that they do not have a fixed home location and are dropped off and picked up from certain locations in the service area. | 1.00 (cosine) |
| 3 | To rent a dockless e-scooter, users download a smartphone application. | To rent a dockless e-scooter, users download a smartphone application. | 1.00 (cosine) |
| 4 | The application shows users a map of nearby e-scooters and enables them to unlock them. | The application shows users a map of nearby e-scooters and enables them to unlock them. | 1.00 (cosine) |
| 5 | Scooters are equipped with built-in GPS chips and cellular connectivity which allows them to report their location in real time during a trip. | Scooters are equipped with built-in GPS chips and cellular connectivity which allows them to report their location in real time during a trip. | 1.00 (cosine) |

</details>

## AI-content analysis

Method: logistic regression calibrated on HC3 + SAGE (GPT-4.1 / Claude 3.7 / Gemini 2.5) (CV AUROC 0.91).

| Signal | Contribution (log-odds; + = more AI-like) |
|---|---|
| log_perplexity | -1.11 |
| burstiness | -3.41 |
| marker_rate | -1.19 |
| classifier | -0.14 |

Raw signals: GPT-2 perplexity 41.0, burstiness 1.39, assistant-phrase rate 0.00/100 words, classifier 0.0002.

## Scoring basis

- Words scored: 193 (template words excluded: 0, reference-list words excluded: 0)
- Document metadata: {"size_bytes": 2583, "pages": 1, "header_footer_blocks_removed": 0, "extraction_seconds": 0.0}
