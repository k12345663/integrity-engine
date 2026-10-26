# Integrity report: `pitch_10_sanjay.mp4`

**Overall risk: HIGH**  
Modality: media | Status: ok

| Signal | Score |
|---|---|
| Internal peer similarity | 81.2% |
| External (web/reference) match | 0.0% |
| AI-content likelihood (advisory) | n/a |

## Summary for reviewers

81.2% of the original text overlaps with other submissions: sub_01_aarav.pdf (81.2%). Strongest evidence: 00:00:04 - 00:00:24 <-> sub_01_aarav.pdf p.2 (verbatim, cos 0.82). AI-content likelihood not estimated: Insufficient original text for a reliable estimate (94 < 120 words). Overall risk: HIGH.

## Cross-modal matches (video/audio <-> document)

### 1. Cross-modal match: `sub_01_aarav.pdf` - verbatim

Chunk cosine **0.82** | sentence evidence **0.95** (sentence cosine + cross-encoder verified paraphrase) | word overlap (Jaccard) 0.76 | media -> pdf

**This submission** - video 00:00:04 - 00:00:24

> **Our recommendation is to stay in Pune for the next** 18 **months and build a dense** battery swapping **network before chasing new cities. Right now each scooter spends almost** 40 **minutes a day** of **the road waiting for a van to collect and recharge it. And that idle time is the biggest reason the fleet** unsolittle. 20 **swap stations near the busiest pickup zones would lift** utilization **from 5.1 to 6.4 rides per scooter per day.**

**Matched source** - `sub_01_aarav.pdf`, page 2, ¶8

> **Our recommendation is to stay in Pune for the next** eighteen **months and build a dense** battery-swapping **network before chasing new cities. Right now each scooter spends almost** forty **minutes a day** off **the road waiting for a van to collect and recharge it, and that idle time is the biggest reason the fleet** earns so little. Twenty **swap stations near the busiest pickup zones would lift** utilisation **from 5.1 to 6.4 rides per scooter per day.**

<details><summary>Sentence-level alignment</summary>

| # | This submission | Source | score (method) |
|---|---|---|---|
| 1 | Our recommendation is to stay in Pune for the next 18 months and build a dense battery swapping network before chasing new cities. | Our recommendation is to stay in Pune for the next eighteen months and build a dense battery-swapping network before chasing new cities. | 0.99 (cosine) |
| 2 | Right now each scooter spends almost 40 minutes a day of the road waiting for a van to collect and recharge it. | Right now each scooter spends almost forty minutes a day off the road waiting for a van to collect and recharge it, and that idle time is the biggest reason the fleet earns so little. | 0.88 (cosine) |
| 3 | And that idle time is the biggest reason the fleet unsolittle. | Right now each scooter spends almost forty minutes a day off the road waiting for a van to collect and recharge it, and that idle time is the biggest reason the fleet earns so little. | 0.95 (cross-encoder; sentence cos 0.50) |
| 4 | 20 swap stations near the busiest pickup zones would lift utilization from 5.1 to 6.4 rides per scooter per day. | Twenty swap stations near the busiest pickup zones would lift utilisation from 5.1 to 6.4 rides per scooter per day. | 0.99 (cosine) |

</details>

## AI-content analysis

Not estimated: Insufficient original text for a reliable estimate (94 < 120 words)

## Scoring basis

- Words scored: 96 (template words excluded: 0, reference-list words excluded: 0)
- Document metadata: {"size_bytes": 349205, "duration_seconds": 28.13, "audio_rms": 0.10404, "asr_model": "whisper-base.en", "asr_segments_dropped": 0, "extraction_seconds": 1.55}
