# Integrity report: `sub_04_diya.docx`

**Overall risk: HIGH**  
Modality: docx | Status: ok

| Signal | Score |
|---|---|
| Internal peer similarity | 71.1% |
| External (web/reference) match | 0.0% |
| AI-content likelihood (advisory) | 0.1% |

## Summary for reviewers

71.1% of the original text overlaps with other submissions: sub_03_kabir.docx (71.1%). Strongest evidence: p.1 <-> sub_03_kabir.docx p.1 (paraphrase, cos 0.80). 147 words of shared template/prompt text were detected and excluded from scoring. AI-content likelihood 0%; strongest signal: burstiness; advisory only - corroborate before acting. Overall risk: HIGH.

## Peer matches (same modality)

### 1. Peer match: `sub_03_kabir.docx` - paraphrase (semantic match, low word overlap)

Chunk cosine **0.80** | sentence evidence **0.64** (cross-encoder verified paraphrase) | word overlap (Jaccard) 0.23 | docx -> docx

**This submission** - page 1-2, ¶4

> There are three reasons Indore is a good test bed. Its central area has almost no slopes, which stretches battery range on every charge. About 120,000 college students live within a short ride **of Vijay Nagar,** producing lots of brief journeys. Finally, no rival sharing company operates in the city today, so the first users will build their routines around VoltRide. The trial would be operated by a local franchisee that buys the vehicles, while VoltRide provides the platform, branding and servicing rules. This approach keeps assets off VoltRide's books and shifts depreciation exposure to the franchisee, who by our estimates recovers the investment within about fourteen months if each scooter does five trips daily. After half a year, a clear threshold decides the next step. Should average trips per vehicle pass 4.5 while damage remains below 3%, the company recruits franchisees for **Nagpur and Coimbatore;** otherwise it brings the scooters home to Pune, having spent under ten million rupees.

**Matched source** - `sub_03_kabir.docx`, page 1-2, ¶4

> Indore suits a pilot for three reasons. The core of the city is flat, so batteries last longer per charge. Around 1.2 lakh students study within six kilometres **of Vijay Nagar,** which gives a dense base of short trips. And no shared scooter operator is active there yet, so early riders will form habits around VoltRide's app. We would run the pilot through a franchise partner who owns the scooters while VoltRide supplies the software, the brand and the maintenance standards. Our model shows payback for the partner in fourteen months at five rides per scooter per day. The decision rule after six months is simple. If rides per scooter exceed 4.5 and the damage rate stays under three percent, VoltRide signs franchise partners in **Nagpur and Coimbatore.**

<details><summary>Sentence-level alignment</summary>

| # | This submission | Source | score (method) |
|---|---|---|---|
| 1 | There are three reasons Indore is a good test bed. | Indore suits a pilot for three reasons. | 0.60 (cross-encoder; sentence cos 0.54) |
| 2 | Its central area has almost no slopes, which stretches battery range on every charge. | The core of the city is flat, so batteries last longer per charge. | 0.60 (cross-encoder; sentence cos 0.53) |
| 3 | About 120,000 college students live within a short ride of Vijay Nagar, producing lots of brief journeys. | Around 1.2 lakh students study within six kilometres of Vijay Nagar, which gives a dense base of short trips. | 0.60 (cross-encoder; sentence cos 0.75) |
| 4 | Finally, no rival sharing company operates in the city today, so the first users will build their routines around VoltRide. | And no shared scooter operator is active there yet, so early riders will form habits around VoltRide's app. | 0.60 (cross-encoder; sentence cos 0.63) |
| 5 | The trial would be operated by a local franchisee that buys the vehicles, while VoltRide provides the platform, branding and servicing rules. | We would run the pilot through a franchise partner who owns the scooters while VoltRide supplies the software, the brand and the maintenance standards. | 0.74 (cross-encoder; sentence cos 0.59) |
| 6 | This approach keeps assets off VoltRide's books and shifts depreciation exposure to the franchisee, who by our estimates recovers the investment within about fourteen months if each scooter does five trips daily. | Our model shows payback for the partner in fourteen months at five rides per scooter per day. | 0.74 (cross-encoder; sentence cos 0.60) |
| 7 | After half a year, a clear threshold decides the next step. | The decision rule after six months is simple. | 0.62 (cross-encoder; sentence cos 0.56) |
| 8 | Should average trips per vehicle pass 4.5 while damage remains below 3%, the company recruits franchisees for Nagpur and Coimbatore; otherwise it brings the scooters home to Pune, having spent under ten million rupees. | If rides per scooter exceed 4.5 and the damage rate stays under three percent, VoltRide signs franchise partners in Nagpur and Coimbatore. | 0.62 (cross-encoder; sentence cos 0.57) |

</details>

## Shared template text (excluded from scores)

These passages appear in many submissions (case prompt, instructions, citations) and are not counted.

- (138 words, in 5 documents, page 1, ¶2) Case Prompt - Inter-University Strategy Challenge 2026, Round 2. VoltRide Mobility operates 1,800 dockless electric scooters in Pune and completed 410,000 rides in the last financial year. The founders must decide whether to expand into three tier-2 cities (Nagpur, Indore and Coimbatore) within e...
- (9 words, in 5 documents, page 2, ¶8) VoltRide Mobility Case Exhibits 1-4, Inter-University Strategy Challenge 2026.

## AI-content analysis

Method: logistic regression calibrated on HC3 + SAGE (GPT-4.1 / Claude 3.7 / Gemini 2.5) (CV AUROC 0.91).

| Signal | Contribution (log-odds; + = more AI-like) |
|---|---|
| log_perplexity | -2.95 |
| burstiness | -3.41 |
| marker_rate | -1.19 |
| classifier | -0.14 |

Raw signals: GPT-2 perplexity 97.3, burstiness 1.13, assistant-phrase rate 0.00/100 words, classifier 0.0004.

## Scoring basis

- Words scored: 228 (template words excluded: 147, reference-list words excluded: 12)
- Document metadata: {"size_bytes": 39062, "page_numbers": "explicit_breaks", "extraction_seconds": 0.0}
