# EVALUATION

All numbers below are reproduced by `python evaluate.py`, which writes [`evaluation/results.md`](evaluation/results.md) and `evaluation/results.json`. AI-calibration numbers come from `python scripts/calibrate_ai_detector.py` (stored in [`config/ai_calibration.json`](config/ai_calibration.json)). The model benchmark comes from `python scripts/benchmark_models.py`. The public paraphrase benchmark comes from `python scripts/benchmark_public_paraphrase.py` ([`evaluation/public_paraphrase.md`](evaluation/public_paraphrase.md)).

> **Read this first: how much these numbers mean.** The corpus has 15 files and 4 planted relationships. I built it myself, and I designed the thresholds while looking at it. Perfect pair-level scores on such a corpus show that every scenario is *handled*, not that the system is accurate in the wild.
>
> I therefore also report three things that are harder to game:
> - a **sentence-level** metric, scored against the exact planted passages;
> - **ablations** that remove one component at a time;
> - an **AI detector calibrated and cross-validated on 577 external, public samples** that were never used for design;
> - an **out-of-sample paraphrase benchmark on 9,725 public sentence pairs** (MRPC, PAWS), run with the engine's fixed thresholds (§5.4).
>
> Thresholds were fixed from the model benchmark in §6 *before* the sweeps in §5 were run; I did not move them to the sweep optimum.
>
> **Version note.** v2 added two matching changes, both designed in response to failures observed on this corpus in v1 (§7.3, §7.4): leave-one-out attribution, and mixed 2- and 3-sentence windows. Their gain (sentence F1 0.89 → 0.97) is therefore *in-sample*. The v1 numbers are kept in the ablation table for comparison.

---

## 1. Setup

- **Corpus:** `sample_submissions/` (15 files: 6 PDF, 6 DOCX, 3 MP4) with ground truth in `dataset_manifest.json`.
  - Relationships: 1 exact copy, 1 paraphrase, 1 cross-modal (MP4 reads a PDF paragraph), 1 external copy (Wikipedia).
  - Also included: 2 AI-generated documents, 3 independent originals, 1 heavily formatted PDF, 3 degraded files, and a 150-word case prompt shared by 5 files.
  - Every submission answers the *same* case and mentions the same cities, prices and fleet sizes, which makes for hard negatives.
- **Hardware:** Apple M5 laptop, CPU only.
- **Default config:** `config/default.yaml`.
- **Runtime:** a full run over 15 files takes **33 s** with warm model caches. Whisper `base.en` transcribes the 28 s clip in 1.6 s. The cross-encoder scored 799 window pairs in the whole run (387 in v1; leave-one-out attribution roughly doubles the calls).
- **Clean installs:** fresh environments built from `requirements.txt` on Python 3.9.25 and 3.12.15 both pass all 17 tests and produce identical reports.

## 2. Headline results

| Area | Metric | Result |
|---|---|---|
| Ingestion robustness | files processed / correctly flagged | **15 / 15**; 3 / 3 degraded files flagged with a reason; batch never crashed |
| Peer + cross-modal detection | pair-level confusion (66 pairs) | **TP 3, FP 0, FN 0, TN 63** |
| Evidence quality | sentence-level P / R / F1 | **1.00 / 0.93 / 0.97** (TP 42, FP 0, FN 3); v1: 0.95 / 0.84 / 0.89 |
| Localisation | evidence points to the planted page / timestamp | 2 / 2 page-checkable relationships on **page 2**; video span **00:00:04–00:00:24** of a 28 s clip (the read-out paragraph, between intro and outro) |
| External similarity | doc-level (threshold 10%) | **TP 1, FP 0, FN 0, TN 11**; cites "Scooter-sharing system" + URL |
| Template isolation | template-only pairs flagged | **0 / 8** (8 / 8 without the filter) |
| AI indicator | this corpus, 2 AI vs 8 human | AUROC 1.00, P = R = 1.00 at 50% (**see §4: this is mostly stylometry**) |
| AI indicator | external HC3 + SAGE, 5-fold CV | **AUROC 0.91**, accuracy 0.84 |
| Risk bands | agreement with expected band (15 files) | **15 / 15** |
| ASR | WER of the read-out clip (Indian-English TTS) | **9.1%**; British-English clip **12.3%** |
| Public paraphrase data (out-of-sample) | MRPC test, 1,725 pairs, fixed thresholds | **F1 0.81, precision 0.79**; cross-encoder AUROC 0.84 vs cosine 0.74 / Jaccard 0.76 |
| Public paraphrase data (adversarial) | PAWS test, 8,000 word-swap pairs | **fails** (accuracy 0.44): it cannot tell "A to B" from "B to A" (§7.9) |

### 2.1 Confusion matrices

**Peer / cross-modal pairs** (unordered document pairs among the 12 analysable files):

| | Predicted match | Predicted no match |
|---|---|---|
| **Actual match** | 3 | 0 |
| **Actual no match** | 0 | 63 |

**Sentence level** (each scoring sentence of ≥ 6 words, in every analysable document):

| | Flagged | Not flagged |
|---|---|---|
| **Planted copy / paraphrase** | 42 | 3 |
| **Original text** | 0 | (all remaining sentences) |

| Document | Relationship | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| pitch_10_sanjay.mp4 | cross-modal (target) | 4 | 0 | 0 | 1.00 | 1.00 |
| sub_01_aarav.pdf | copy + cross-modal (source) | 8 | 0 | 0 | 1.00 | 1.00 |
| sub_02_meera.pdf | exact copy (target) | 5 | 0 | 0 | 1.00 | 1.00 |
| sub_03_kabir.docx | paraphrase (source) | 12 | 0 | 1 | 1.00 | 0.92 |
| sub_04_diya.docx | paraphrase (target) | 8 | 0 | 2 | 1.00 | 0.80 |
| sub_06_ananya.pdf | external copy | 5 | 0 | 0 | 1.00 | 1.00 |

**AI indicator** (threshold 50%, 10 documents with ≥ 120 words; the two short video transcripts are "not estimated"):

| | Predicted AI | Predicted human |
|---|---|---|
| **AI-generated** | 2 (99.7%, 86.1%) | 0 |
| **Human-style** | 0 | 8 (all ≤ 1.3%) |

## 3. Scenario walk-through

| Scenario | What happened | Pass |
|---|---|---|
| **1. Cross-modal duplication** | Whisper transcribed `pitch_10`. All 4 transcript sentences of the read-out paragraph align to `sub_01` **page 2, ¶8**, at sentence evidence 0.88–0.99, even though ASR wrote "18" for "eighteen" and "unsolittle" for "earns so little". The video is 81% covered; its intro and outro are correctly *not* matched. The `sub_01` report lists the match under `multimodal_matches` with `matched_timestamp`. | ✅ |
| **2. Paraphrase** | `sub_03` ↔ `sub_04`: chunk cosine 0.80, cross-encoder 0.60–0.80, word Jaccard **0.23**, Levenshtein ratio **0.51**. 89% / 71% coverage, labelled "paraphrase (semantic match, low word overlap)". A classic 5-gram shingle baseline misses this pair entirely (§5.2). | ✅ |
| **3. Corrupted / empty** | `empty_13.pdf` ("File is empty (0 bytes)"), `corrupt_14.docx` ("DOCX is corrupted or not an Office Open XML file") and `silent_12.mp4` ("Video has no audio track") are each logged as WARNING and get an UNASSESSED report. The other 12 files are processed. `tests/test_edge_cases.py` adds image-only PDF, password PDF, muted audio track, truncated MP4, unsupported type, upper-case extension, nested folders and identical duplicates. | ✅ |
| **4. Common template** | 45 sentences (9 per file: the 8-sentence, 150-word prompt + 1 shared citation; DF = 5 ≥ threshold 3) are excluded from scoring in the five files that share them. `sub_05` stays **LOW (0%)** despite sharing 147 words with four other files. Disabling the filter flags all **8** template-only pairs. | ✅ |
| **5. Batch stress test** | One command processes all 15 files and writes 30 reports, `summary.md` (sorted by risk, with flagged files listed) and an N×N similarity matrix (`similarity_matrix.csv`). | ✅ |

## 4. AI-generation indicator: calibration and honest limits

**Calibration on external data** (`scripts/calibrate_ai_detector.py`): 577 texts of ≥ 120 words, 50/50 human/AI. 30% come from HC3 (2022 ChatGPT, finance / CS / open QA). 70% come from SAGE (human IvyPanda essays and Medium articles vs GPT-4.1, Claude 3.7 Sonnet and Gemini 2.5 Flash).

| Feature | AUROC (all) | HC3 only | SAGE only (modern LLMs) |
|---|---|---|---|
| GPT-2 log-perplexity (low → AI) | 0.70 | **1.00** | 0.62 |
| Burstiness (low → AI) | 0.78 | 0.63 | **0.84** |
| Assistant-phrase rate | 0.73 | 0.52 | **0.83** |
| RoBERTa OpenAI detector | 0.67 | 0.98 | **0.55 (chance)** |
| **Logistic combination (5-fold CV)** | **0.91** | | |

The table reports the SAGE burstiness value as 1 − 0.158 (the score inverted so higher means more AI-like).

**Cross-source transfer** (fit on one corpus, test on the other):

| Trained on → tested on | AUROC |
|---|---|
| HC3 → SAGE | **0.57** |
| SAGE → HC3 | **0.21** (inverted) |

What this shows:

1. The **perplexity** and **supervised-detector** signals that work perfectly on 2022 ChatGPT text barely work on 2025 frontier models. This is the main reason detector scores must not be treated as proof.
2. **My first calibration (HC3 only) scored every document in our corpus as human (0%)**, including the two AI texts. Adding SAGE fixed this. It is a concrete example of detector ageing and domain shift.
3. On our corpus, AUROC 1.0 is driven by the **assistant-phrase rate**. The two AI documents use 7.5 and 9.5 marker phrases per 100 words, the human-style documents use 0. Perplexity alone would *not* separate them: `sub_15` (AI) has perplexity 47, and `sub_06` (human, encyclopedic) has 44.

**Robustness probe.** I deleted the marker phrases from `sub_07` and re-scored it:

| Variant | AI % | Marker rate | GPT-2 perplexity |
|---|---|---|---|
| original | 99.7 | 9.5 / 100 words | 24.9 |
| markers removed | **0.6** | 0 | 46.1 |

A student who lightly edits ChatGPT output defeats this indicator, as they would defeat most current detectors. This is why the engine treats AI likelihood as advisory: on its own it can raise a submission to MEDIUM at most.

## 5. Ablations, baselines and threshold sweeps

### 5.1 Ablations

Each row removes one component, with everything else held at default.

| Variant | Pair TP/FP/FN | Pair F1 | Sentence P | Sentence R | Sentence F1 | Template pairs flagged |
|---|---|---|---|---|---|---|
| **Full system (v2)** | 3/0/0 | **1.00** | **1.00** | 0.93 | **0.97** | 0 |
| No boilerplate filter | 3/**8**/0 | 0.43 | 0.49 | 0.87 | 0.63 | **8** |
| No cross-encoder (bi-encoder only) | 2/**1**/**1** | 0.67 | 0.73 | **0.49** | 0.59 | 1 |
| No anchor rule | 3/**1**/0 | 0.86 | **0.67** | **0.98** | 0.79 | 0 |
| No leave-one-out attribution | 3/0/0 | 1.00 | 0.93 | 0.93 | 0.93 | 0 |
| 3-sentence windows only | 3/0/0 | 1.00 | 0.97 | **0.80** | 0.88 | 0 |
| v1 (3-sentence windows, no leave-one-out) | 3/0/0 | 1.00 | 0.95 | 0.84 | 0.89 | 0 |
| Single-sentence evidence allowed | 3/0/0 | 1.00 | 1.00 | 0.93 | 0.97 | 0 |

Every component earns its place:

- **The template filter** prevents 8 false accusations.
- **The cross-encoder** is the only thing that finds the paraphrase. Bi-encoder-only (with the cosine fallback) misses the pair, halves sentence recall and lets one template pair through.
- **The anchor rule** removes the AI-vs-AI false positive (§7.2). It costs some recall (0.98 → 0.93), because some paraphrased windows contain no shared number or name.
- **Leave-one-out attribution** removes the two spill-over false positives (precision 0.93 → 1.00, §7.3).
- **2-sentence windows** recover paraphrased sentences at paragraph edges (recall 0.80 → 0.93, §7.4). That is the trade-off I chose, since a false accusation is costlier than a missed paraphrase that a reviewer can still find via the similarity matrix.
- **The ≥ 2-sentence passage rule** has little effect here, but it guards against single generic sentences in larger corpora.

### 5.2 Lexical baseline vs semantic engine (Scenario 2)

A classic word 5-gram shingle detector (pair flagged if ≥ 3 shared shingles) gets **P 1.00 / R 0.67**: it finds the verbatim copy and the read-out transcript but not the paraphrase.

| Relationship | Chunk cosine | Sentence evidence | Word Jaccard | Levenshtein ratio |
|---|---|---|---|---|
| Exact copy | 0.95 | 1.00 | 1.00 | 1.00 |
| Cross-modal (ASR transcript) | 0.82 | 0.95 | 0.76 | 0.95 |
| **Paraphrase** | 0.80 | 0.64 (cross-encoder) | **0.23** | **0.51** |

The brief's Scenario 2 expects cosine > 0.80 alongside low lexical overlap. Two observations:

- MiniLM's *chunk* cosine for a heavy paraphrase is 0.78–0.80, and it is not reliably above the 0.77 that two *unrelated* answers to the same case reach (§6). A pure "cosine > 0.80" rule would therefore be either too loose or too strict on this corpus.
- That is why the decision is made by the cross-encoder plus the anchor check rather than by a raw cosine threshold.

### 5.3 Threshold sweeps

**Direct-path sentence cosine:**

| Threshold | Pair TP/FP/FN | Sentence P / R / F1 |
|---|---|---|
| 0.70 | 3/**1**/0 (AI-vs-AI) | 0.91 / 0.93 / 0.92 |
| 0.75–0.90 | 3/0/0 | 1.00 / 0.93 / 0.97 |

Results are stable from 0.75 to 0.90; the default of 0.80 sits mid-plateau.

**Cross-encoder threshold:**

| Threshold | Sentence P / R / F1 |
|---|---|
| 0.50 | 1.00 / 0.93 / 0.97 |
| 0.55 | 1.00 / 0.93 / 0.97 |
| **0.60 (default)** | 1.00 / 0.93 / 0.97 |
| 0.65 | 1.00 / 0.76 / 0.86 |
| 0.70 | 1.00 / 0.69 / 0.82 |

Results are flat from 0.50 to 0.60, and recall falls off above that. The default of 0.60 was chosen beforehand from the benchmark, where it sits just above the highest same-topic negative window score (0.589), so it is the most conservative point on the plateau. (In v1, 0.50 scored marginally better; I did not move the threshold to chase that.)

### 5.4 Out-of-sample: public paraphrase benchmarks

The sample corpus is too small to trust, so `scripts/benchmark_public_paraphrase.py` applies the engine's sentence-pair decision rule to two public, human-labelled datasets. The rule is direct cosine ≥ 0.80, **or** cross-encoder ≥ 0.60 with cosine ≥ 0.60 and the anchor rule, all at the **fixed production thresholds** with no tuning. Baselines get thresholds tuned for F1 on each dataset's validation split, which favours them.

**MRPC** (1,725 news sentence pairs, 66% paraphrases):

| Method | Precision | Recall | F1 | Accuracy | AUROC |
|---|---|---|---|---|---|
| Word Jaccard (tuned, ≥ 0.30) | 0.71 | 0.96 | 0.82 | 0.71 | 0.76 |
| Levenshtein ratio (tuned, ≥ 0.58) | 0.72 | 0.92 | 0.81 | 0.71 | 0.75 |
| MiniLM cosine (tuned, ≥ 0.56) | 0.70 | 0.97 | 0.81 | 0.70 | 0.74 |
| Cross-encoder score | | | | | **0.84** |
| Engine: direct path only (fixed) | **0.80** | 0.65 | 0.72 | 0.66 | |
| Engine: direct + cross-encoder (fixed) | 0.76 | 0.92 | **0.84** | **0.76** | |
| **Engine: + anchor rule (default, fixed)** | 0.79 | 0.84 | 0.81 | 0.74 | |

- The cross-encoder is the strongest single signal on real paraphrases (AUROC 0.84 vs 0.74–0.76). This confirms, on data I did not build, the design choice made in §6.
- At fixed thresholds the engine is **more precise** than every tuned baseline (0.79 vs 0.70–0.72). For an integrity tool, where a false accusation is the costly error, that is the right direction.
- The anchor rule costs a little F1 here (0.84 → 0.81). MRPC pairs are single sentences from the same news story, often without numbers or rare names, so the rule rejects some true paraphrases. The trade is −3 points of F1 on MRPC against a precision gain from 0.67 to 1.00 on our corpus, where it removes the AI-vs-AI false positive. That is a judgement call, and the flag is configurable (`matching.require_anchor`).

**PAWS** (8,000 adversarial pairs, 44% paraphrases): all methods fail, including mine. The engine flags 99.9% of pairs (accuracy 0.44); cosine AUROC is 0.63 and cross-encoder AUROC is 0.53. See §7.9.

## 6. Trade-off analysis: embedding model, verifier and index vs compute

**Model benchmark** (`scripts/benchmark_models.py`): 4 paraphrased paragraph pairs vs 615 paragraph pairs from *unrelated* submissions on the same case. A usable scorer needs a positive margin (min positive − max negative).

| Scorer | Size | Paraphrase min | Same-topic negative max | AI-vs-AI max | Margin |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (bi-encoder) | 90 MB | 0.663 | 0.846 | 0.806 | −0.182 |
| paraphrase-MiniLM-L6-v2 | 90 MB | 0.651 | 0.804 | 0.782 | −0.154 |
| all-mpnet-base-v2 | 420 MB | 0.769 | 0.829 | 0.821 | −0.060 |
| bge-small-en-v1.5 | 130 MB | 0.840 | 0.863 | 0.854 | −0.023 |
| gte-small | 130 MB | 0.936 | 0.939 | 0.942 | −0.004 |
| **stsb-distilroberta-base (cross-encoder)** | 330 MB | 0.651 | **0.534** | 0.688 | **+0.117** |

Window-size study for the cross-encoder. This was an exploratory analysis during development: each document's windows were compared to their nearest window in every other document, and the cross-encoder scored the best-matching pair.

| Window | Max same-topic negative | Median paraphrase score | Paraphrase windows ≥ 0.60 |
|---|---|---|---|
| 1 sentence | 0.651 | 0.582 | 36% |
| 2 sentences | 0.667 | 0.636 | 65% |
| **3 sentences** | **0.589** | 0.633 | **71%** |

Decisions that follow:

- **Bi-encoder: keep MiniLM-L6 for retrieval.** No bi-encoder separates paraphrase from same-topic text. Larger ones (mpnet is 4.7× the size and correspondingly slower) narrow the gap but never close it. Retrieval only needs *recall* at a loose threshold, so the cheapest model wins. It also embeds ~1.8k sentences/s on a laptop CPU (measured, batch 64), which matters at 1M sentences (README §8).
- **Verifier: cross-encoder on 2- and 3-sentence windows,** run only on bi-encoder-shortlisted window pairs. Here that is 799 pairs for the whole corpus (including leave-one-out re-scoring), instead of every window pair. 3-sentence windows have the cleanest separation (table above), and adding 2-sentence windows recovers paragraph-edge recall without new false positives (§5.1). That is the classic retrieve-then-rerank trade: roughly 100× the per-pair cost, paid on well under 1% of pairs.
- **Index: FAISS `IndexFlatIP`.** Exact search over ~200 chunks costs microseconds, and exactness removes one source of error from the evaluation. HNSW only pays off past ~1M vectors and adds ~256 bytes of graph links per vector at M = 32 (~17% over 384-d float32). IVF-PQ saves ~30× memory at a recall cost. The switch is a config flag (`matching.index_type: hnsw`), and the scaling plan is in README §8.3.
- **ASR: Whisper `base.en`** gives 9–12% WER on accented TTS speech at ~18× real time on CPU (a 28 s clip in 1.6 s). `tiny` is faster but less accurate; `small.en` should reduce proper-noun errors like "Obit/Orbit" and "Coimbart or" at ~3× the compute. I did not measure either here; `small.en` is the production choice on GPU (README §8.2).
- **AI LM: GPT-2 small (124M).** Larger observers (e.g. Binoculars with Falcon-7B) are much stronger zero-shot detectors but need a GPU with ~16 GB of memory. That is out of scope for "commodity hardware", and the calibration shows perplexity is a weak signal on modern LLM text anyway.

## 7. Failure analysis

### 7.1 AI indicator depends on assistant phrasing and is easily evaded (false-negative risk)
- **Observed.** Deleting marker phrases drops `sub_07` from 99.7% to 0.6% (§4). On modern-LLM data, perplexity has AUROC 0.62 and the RoBERTa detector 0.55. An HC3-only calibration labelled both AI documents as human.
- **Why.** Frontier models write with higher perplexity than 2022 ChatGPT, and supervised detectors do not transfer across generators (HC3→SAGE AUROC 0.57).
- **Resolution.**
  - Keep AI likelihood advisory (done).
  - Fine-tune a detector on multi-generator, multi-domain data (RAID, SAGE, DACTYL) and recalibrate per challenge.
  - Use stronger zero-shot observers (Binoculars) on GPU.
  - Above all, rely on process evidence: drafts, version history, oral defence.

### 7.2 Two independent AI-written answers look like a paraphrase pair (ambiguous false positive)
- **Observed.** Before the anchor rule, `sub_07` and `sub_15` (no ground-truth relation) matched at 29–32% coverage via cross-encoder windows scored ~0.70. One example: "To mitigate these risks, VoltRide should foster strong relationships with municipal authorities…" vs "In summary, by harnessing operational excellence and strategic foresight…". With the direct threshold at 0.70, they also match by plain cosine (§5.3).
- **Why.** LLM answers to the same prompt converge on the same generic argument structure and vocabulary.
- **Ambiguity.** Arguably this *is* a signal: two students submitting near-identical LLM output. But it is not peer copying, and calling it plagiarism would be unfair.
- **Resolution (implemented).** Verified paraphrase must share a rare specific detail: a number, or two rare names (`anchors.py`). This removed the false positive (sentence precision 0.61 → 0.95). The cost is that a paraphrase that drops every number and name is only caught when the cross-encoder is ≥ 0.80 confident.
- **Future.** Report "generic-text similarity" as its own category and cluster AI-like submissions rather than pairing them.

### 7.3 Window spill-over: adjacent original sentence pulled into a copied passage (v1: sentence-level FP ×2, fixed in v2)
- **Observed (v1).** In `sub_02`, "Finally the third risk is execution: hiring a city manager who knows Indore." was flagged as matching `sub_01`'s "The third risk is that a well funded rival enters Pune…". It sits right after the two copied risk paragraphs.
- **Why.** The cross-encoder verifies 3-sentence *windows*. A window made of two copied sentences plus one structurally parallel original sentence scores 0.66, and every sentence in it is attributed.
- **First idea, rejected with data.** A sentence-cosine floor cannot work: both false positives had sentence cosine 0.55, in the middle of the true verified pairs' range (0.24–0.75).
- **Resolution (implemented in v2).** Leave-one-out attribution: re-score each verified window without each of its sentences, and attribute only sentences whose removal does *not* raise the cross-encoder score. Both false positives disappeared (sentence precision 0.93 → 1.00, with no recall loss). `sub_01`/`sub_02` coverage fell from 48%/50% to 44%/45%, which is the inflation removed. The cost is ~2× cross-encoder calls.

### 7.4 Paraphrase recall at sentence level (v1: 0.70, v2: 0.80–0.92)
- **Observed (v1).** The opening sentences ("Kabir's team argues for a single-city pilot…" / "In our view VoltRide should test one new market…") and the closing conditional sentence of the paraphrase pair were missed. So was "If not, it pulls the fleet back to Pune and has lost less than one crore rupees" vs "otherwise it brings the scooters home to Pune, having spent under ten million rupees".
- **Why.** Windows at paragraph and document edges mix in unrelated context. Heavy restructuring (two sentences merged into one) lowers both the bi-encoder prefilter and the cross-encoder below 0.60. In some windows the anchor rule rejects otherwise-correct evidence.
- **Resolution (implemented in v2).** Mixed 2- and 3-sentence windows, so edge sentences get windows without unrelated context. Sentence recall on the pair rose from 0.69/0.70 to 0.92/0.80 with no new false positives.
- **Still missed (3 sentences).** "If not, it pulls the fleet back to Pune and has lost less than one crore rupees" (`sub_03`); "In our view VoltRide should test one new market, Indore…" and the long merged sentence "The aim is inexpensive learning: trialling roughly three hundred e-scooters…" (`sub_04`, which compresses two source sentences into one).
- **Further fixes.**
  - A cross-encoder fine-tuned for paraphrase identification (MRPC / QQP) rather than generic STS.
  - Many-to-one alignment, so one target sentence can absorb two source sentences.
  - Let a confirmed passage absorb neighbouring sentences whose cross-encoder score is ≥ 0.5.

### 7.5 ASR errors on proper nouns and numbers
- **Observed.** "Team Orbit" → "Team **Obit**"; "VoltRide" → "Voltride" / "Volt Ride"; "Coimbatore" → "**Coimbart or**"; "earns so little" → "**unsolittle**"; spoken numbers came out as digits ("18" for "eighteen").
- **Impact.** Matching was unaffected (embeddings gave sentence cosine 0.99 for "18 months" vs "eighteen months"). But the Markdown highlighter, which bolds identical 3-word runs, under-highlights number differences, and an AI or lexical analysis of the transcript would be distorted.
- **Resolution.**
  - Whisper `small.en`/`large-v3` (faster-whisper on GPU).
  - An `initial_prompt` with the case vocabulary (company names, cities).
  - Number normalisation before highlighting (`anchors.extract_numbers` already handles digit/word equivalence).
  - Real accented human speech should be tested; TTS is far cleaner than a nervous student in a noisy room.

### 7.6 Template detection can hide a collusion ring (design risk, not observed)
- **Risk.** A sentence counts as template when it appears in ≥ max(3, 25% of) submissions. With 12 analysable files the threshold is 3, so three students sharing the same copied paragraph would be treated as template.
- **Resolution.**
  - Organisers upload the prompt (`--template_dir`); the DF rule then only needs to catch citations and instructions, and its threshold can be raised.
  - Flag "high-DF text that is *not* in the official prompt" for review instead of silently discarding it.
  - At 10k submissions the fractional threshold (25% = 2,500 documents) makes this practically impossible.

### 7.7 Direction of copying is unknown
Both `sub_01` (the source) and `sub_02` (the copier) are rated HIGH. Similarity is symmetric; only provenance can say who copied whom: submission timestamps, draft history or earlier public versions. The report says "overlaps with", never "copied from".

### 7.8 Dataset validity
- **Single author.** All texts were drafted with an AI assistant, including the "human" ones (disclosed in the manifest). So the AI-detection result on this corpus measures style separation, not real-world accuracy.
- **Clean speech.** TTS speech is cleaner than real recordings.
- **Small scale.** 4 relationships cannot estimate rates with any confidence: a 95% interval on 3/3 recall spans roughly 0.29–1.0.
- **Next step.** Hold out a labelled sample of real, adjudicated past cases and evaluate on that before setting production thresholds.

### 7.9 Adversarial word-order paraphrases (PAWS): every method fails
- **Observed.** On PAWS the engine flags 99.9% of pairs. PAWS negatives are built by swapping words ("flights from New York to Florida" vs "flights from Florida to New York"), so both sentences share nearly all their words. MiniLM cosine (AUROC 0.63) and the STS cross-encoder (0.53) are near chance; only character-level Levenshtein does a little better (0.76).
- **Does it matter for plagiarism?** Mostly not. A PAWS negative copies the source's wording, so flagging it as overlap is defensible. But the engine cannot tell "same claim" from "same words, opposite claim", so a reviewer must read the passages, which the report puts side by side.
- **Resolution.** A cross-encoder fine-tuned on PAWS (the PAWS authors report ~90% accuracy for BERT trained on it), used only as a meaning-agreement label in the report, not as a gate.

## 8. What I would do next

1. Real-data hold-out set and threshold recalibration (§7.8).
2. Paraphrase-tuned cross-encoder (MRPC/QQP; PAWS for meaning agreement); distilled version for throughput.
3. Many-to-one sentence alignment for merged paraphrases (§7.4).
4. Fine-tuned multi-generator AI detector plus Binoculars on GPU; keep it advisory.
5. OCR for image-only PDFs and keyframe OCR for slide text in videos.
6. Reviewer feedback loop: decisions become labels for recalibration.
