# Integrity report: `sub_09_vikram.docx`

**Overall risk: LOW**  
Modality: docx | Status: ok

| Signal | Score |
|---|---|
| Internal peer similarity | 0.0% |
| External (web/reference) match | 0.0% |
| AI-content likelihood (advisory) | 0.0% |

## Summary for reviewers

No meaningful overlap with other submissions. AI-content likelihood 0%; strongest signal: log_perplexity; advisory only - corroborate before acting. Overall risk: LOW.

## AI-content analysis

Method: logistic regression calibrated on HC3 + SAGE (GPT-4.1 / Claude 3.7 / Gemini 2.5) (CV AUROC 0.91).

| Signal | Contribution (log-odds; + = more AI-like) |
|---|---|
| log_perplexity | -3.45 |
| burstiness | -3.41 |
| marker_rate | -1.19 |
| classifier | -0.14 |

Raw signals: GPT-2 perplexity 123.1, burstiness 1.32, assistant-phrase rate 0.00/100 words, classifier 0.0002.

## Scoring basis

- Words scored: 165 (template words excluded: 0, reference-list words excluded: 0)
- Document metadata: {"size_bytes": 38277, "page_numbers": "explicit_breaks", "extraction_seconds": 0.0}
