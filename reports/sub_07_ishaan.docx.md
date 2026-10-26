# Integrity report: `sub_07_ishaan.docx`

**Overall risk: MEDIUM**  
Modality: docx | Status: ok

| Signal | Score |
|---|---|
| Internal peer similarity | 0.0% |
| External (web/reference) match | 0.0% |
| AI-content likelihood (advisory) | 99.7% |

## Summary for reviewers

No meaningful overlap with other submissions. AI-content likelihood 100%; strongest signal: marker_rate; advisory only - corroborate before acting. Overall risk: MEDIUM.

## AI-content analysis

Method: logistic regression calibrated on HC3 + SAGE (GPT-4.1 / Claude 3.7 / Gemini 2.5) (CV AUROC 0.91).

| Signal | Contribution (log-odds; + = more AI-like) |
|---|---|
| log_perplexity | -0.04 |
| burstiness | -0.80 |
| marker_rate | +6.23 |
| classifier | -0.14 |

Raw signals: GPT-2 perplexity 24.9, burstiness 0.65, assistant-phrase rate 9.54/100 words, classifier 0.0004.

## Scoring basis

- Words scored: 245 (template words excluded: 0, reference-list words excluded: 0)
- Document metadata: {"size_bytes": 38692, "page_numbers": "explicit_breaks", "extraction_seconds": 0.01}
