# Integrity report: `sub_08_zara.pdf`

**Overall risk: LOW**  
Modality: pdf | Status: ok

| Signal | Score |
|---|---|
| Internal peer similarity | 0.0% |
| External (web/reference) match | 0.0% |
| AI-content likelihood (advisory) | 0.6% |

## Summary for reviewers

No meaningful overlap with other submissions. AI-content likelihood 1%; strongest signal: log_perplexity; advisory only - corroborate before acting. Overall risk: LOW.

## AI-content analysis

Method: logistic regression calibrated on HC3 + SAGE (GPT-4.1 / Claude 3.7 / Gemini 2.5) (CV AUROC 0.91).

| Signal | Contribution (log-odds; + = more AI-like) |
|---|---|
| log_perplexity | -2.86 |
| burstiness | -1.47 |
| marker_rate | -1.19 |
| classifier | -0.14 |

Raw signals: GPT-2 perplexity 93.5, burstiness 0.76, assistant-phrase rate 0.00/100 words, classifier 0.0003.

## Scoring basis

- Words scored: 191 (template words excluded: 0, reference-list words excluded: 0)
- Document metadata: {"size_bytes": 3513, "pages": 2, "header_footer_blocks_removed": 4, "extraction_seconds": 0.0}
