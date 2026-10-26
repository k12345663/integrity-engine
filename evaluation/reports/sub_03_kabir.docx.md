# Integrity report: `sub_03_kabir.docx`

**Overall risk: HIGH**  
Modality: docx | Status: ok

| Signal | Score |
|---|---|
| Internal peer similarity | 89.4% |
| External (web/reference) match | 0.0% |
| AI-content likelihood (advisory) | 0.1% |

## Summary for reviewers

89.4% of the original text overlaps with other submissions: sub_04_diya.docx (89.4%). Strongest evidence: p.1 <-> sub_04_diya.docx p.1 (paraphrase, cos 0.80). 147 words of shared template/prompt text were detected and excluded from scoring. AI-content likelihood 0%; strongest signal: burstiness; advisory only - corroborate before acting. Overall risk: HIGH.

## Peer matches (same modality)

### 1. Peer match: `sub_04_diya.docx` - paraphrase (semantic match, low word overlap)

Chunk cosine **0.80** | sentence evidence **0.73** (cross-encoder verified paraphrase) | word overlap (Jaccard) 0.24 | docx -> docx

**This submission** - page 1-2, ¶3

> Kabir's team argues for a single-city pilot in Indore rather than an immediate three city rollout. The logic is about learning cheaply. A pilot of 300 scooters costs a fraction of a full launch and tells VoltRide whether its Pune playbook transfers to a smaller market before serious capital is committed. Indore suits a pilot for three reasons. The core of the city is flat, so batteries last longer per charge. Around 1.2 lakh students study within six kilometres **of Vijay Nagar,** which gives a dense base of short trips. And no shared scooter operator is active there yet, so early riders will form habits around VoltRide's app. We would run the pilot through a franchise partner who owns the scooters while VoltRide supplies the software, the brand and the maintenance standards. This keeps the balance sheet light, and the partner carries the depreciation risk. Our model shows payback for the partner in fourteen months at five rides per scooter per day. The decision rule after six months is simple. If rides per scooter exceed 4.5 and the damage rate stays under three percent, VoltRide signs franchise partners in **Nagpur and Coimbatore.**

**Matched source** - `sub_04_diya.docx`, page 1-2, ¶1

> Team Diya - Test One City First The aim is inexpensive learning: trialling roughly three hundred e-scooters needs only a small share of the money a complete rollout would, and it reveals whether the operating model built in Pune works in a smaller city before large investments are made. There are three reasons Indore is a good test bed. Its central area has almost no slopes, which stretches battery range on every charge. About 120,000 college students live within a short ride **of Vijay Nagar,** producing lots of brief journeys. Finally, no rival sharing company operates in the city today, so the first users will build their routines around VoltRide. The trial would be operated by a local franchisee that buys the vehicles, while VoltRide provides the platform, branding and servicing rules. This approach keeps assets off VoltRide's books and shifts depreciation exposure to the franchisee, who by our estimates recovers the investment within about fourteen months if each scooter does five trips daily. After half a year, a clear threshold decides the next step. Should average trips per vehicle pass 4.5 while damage remains below 3%, the company recruits franchisees for **Nagpur and Coimbatore;** otherwise it brings the scooters home to Pune, having spent under ten million rupees.

<details><summary>Sentence-level alignment</summary>

| # | This submission | Source | score (method) |
|---|---|---|---|
| 1 | Kabir's team argues for a single-city pilot in Indore rather than an immediate three city rollout. | Team Diya - Test One City First | 0.80 (cross-encoder; sentence cos 0.42) |
| 2 | The logic is about learning cheaply. | The aim is inexpensive learning: trialling roughly three hundred e-scooters needs only a small share of the money a complete rollout would, and it reveals whether the operating model built in Pune works in a smaller city before large investments are made. | 0.80 (cross-encoder; sentence cos 0.33) |
| 3 | A pilot of 300 scooters costs a fraction of a full launch and tells VoltRide whether its Pune playbook transfers to a smaller market before serious capital is committed. | The aim is inexpensive learning: trialling roughly three hundred e-scooters needs only a small share of the money a complete rollout would, and it reveals whether the operating model built in Pune works in a smaller city before large investments are made. | 0.80 (cross-encoder; sentence cos 0.64) |
| 4 | Indore suits a pilot for three reasons. | There are three reasons Indore is a good test bed. | 0.80 (cross-encoder; sentence cos 0.54) |
| 5 | The core of the city is flat, so batteries last longer per charge. | Its central area has almost no slopes, which stretches battery range on every charge. | 0.67 (cross-encoder; sentence cos 0.53) |
| 6 | Around 1.2 lakh students study within six kilometres of Vijay Nagar, which gives a dense base of short trips. | About 120,000 college students live within a short ride of Vijay Nagar, producing lots of brief journeys. | 0.67 (cross-encoder; sentence cos 0.75) |
| 7 | And no shared scooter operator is active there yet, so early riders will form habits around VoltRide's app. | Finally, no rival sharing company operates in the city today, so the first users will build their routines around VoltRide. | 0.67 (cross-encoder; sentence cos 0.63) |
| 8 | We would run the pilot through a franchise partner who owns the scooters while VoltRide supplies the software, the brand and the maintenance standards. | The trial would be operated by a local franchisee that buys the vehicles, while VoltRide provides the platform, branding and servicing rules. | 0.75 (cross-encoder; sentence cos 0.59) |
| 9 | This keeps the balance sheet light, and the partner carries the depreciation risk. | This approach keeps assets off VoltRide's books and shifts depreciation exposure to the franchisee, who by our estimates recovers the investment within about fourteen months if each scooter does five trips daily. | 0.75 (cross-encoder; sentence cos 0.40) |
| 10 | Our model shows payback for the partner in fourteen months at five rides per scooter per day. | This approach keeps assets off VoltRide's books and shifts depreciation exposure to the franchisee, who by our estimates recovers the investment within about fourteen months if each scooter does five trips daily. | 0.75 (cross-encoder; sentence cos 0.60) |
| 11 | The decision rule after six months is simple. | After half a year, a clear threshold decides the next step. | 0.62 (cross-encoder; sentence cos 0.56) |
| 12 | If rides per scooter exceed 4.5 and the damage rate stays under three percent, VoltRide signs franchise partners in Nagpur and Coimbatore. | Should average trips per vehicle pass 4.5 while damage remains below 3%, the company recruits franchisees for Nagpur and Coimbatore; otherwise it brings the scooters home to Pune, having spent under ten million rupees. | 0.62 (cross-encoder; sentence cos 0.57) |

</details>

## Shared template text (excluded from scores)

These passages appear in many submissions (case prompt, instructions, citations) and are not counted.

- (138 words, in 5 documents, page 1, ¶2) Case Prompt - Inter-University Strategy Challenge 2026, Round 2. VoltRide Mobility operates 1,800 dockless electric scooters in Pune and completed 410,000 rides in the last financial year. The founders must decide whether to expand into three tier-2 cities (Nagpur, Indore and Coimbatore) within e...
- (9 words, in 5 documents, page 2, ¶8) VoltRide Mobility Case Exhibits 1-4, Inter-University Strategy Challenge 2026.

## AI-content analysis

Method: logistic regression calibrated on HC3 + SAGE (GPT-4.1 / Claude 3.7 / Gemini 2.5) (CV AUROC 0.91).

| Signal | Contribution (log-odds; + = more AI-like) |
|---|---|
| log_perplexity | -2.85 |
| burstiness | -3.41 |
| marker_rate | -1.19 |
| classifier | -0.14 |

Raw signals: GPT-2 perplexity 92.8, burstiness 1.26, assistant-phrase rate 0.00/100 words, classifier 0.0003.

## Scoring basis

- Words scored: 217 (template words excluded: 147, reference-list words excluded: 12)
- Document metadata: {"size_bytes": 38989, "page_numbers": "explicit_breaks", "extraction_seconds": 0.0}
