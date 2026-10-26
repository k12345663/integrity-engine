# Integrity report: `sub_15_tanvi.pdf`

**Overall risk: MEDIUM**  
Modality: pdf | Status: ok

| Signal | Score |
|---|---|
| Internal peer similarity | 0.0% |
| External (web/reference) match | 0.0% |
| AI-content likelihood (advisory) | 86.1% |

## Summary for reviewers

No meaningful overlap with other submissions. AI-content likelihood 86%; strongest signal: marker_rate; advisory only - corroborate before acting. Overall risk: MEDIUM.

## AI-content analysis

Method: logistic regression calibrated on HC3 + SAGE (GPT-4.1 / Claude 3.7 / Gemini 2.5) (CV AUROC 0.91).

| Signal | Contribution (log-odds; + = more AI-like) |
|---|---|
| log_perplexity | -1.40 |
| burstiness | -3.41 |
| marker_rate | +6.23 |
| classifier | -0.14 |

Raw signals: GPT-2 perplexity 47.0, burstiness 1.26, assistant-phrase rate 7.92/100 words, classifier 0.0005.

## Scoring basis

- Words scored: 210 (template words excluded: 0, reference-list words excluded: 0)
- Document metadata: {"size_bytes": 2901, "pages": 1, "header_footer_blocks_removed": 0, "extraction_seconds": 0.0}
