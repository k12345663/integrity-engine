"""Fast unit tests (no ML models needed)."""

import numpy as np

from integrity_engine.anchors import anchors_agree, extract_numbers
from integrity_engine.boilerplate import df_threshold
from integrity_engine.chunking import chunk_sentences
from integrity_engine.config import BoilerplateConfig, ChunkingConfig, RiskConfig
from integrity_engine.index import VectorIndex
from integrity_engine.models import Sentence
from integrity_engine.normalize import (
    PAGE_NUMBER_RE,
    clean_text,
    find_repeated_margin_lines,
    format_timestamp,
    split_sentences,
)
from integrity_engine.scoring import risk_level


def test_clean_text_dehyphenates_and_normalises():
    assert clean_text("infor-\nmation ﬁnance’s  \n text") == "information finance's text"


def test_sentence_split_respects_abbreviations_and_decimals():
    text = "Margins fell 4.2% in Q3, e.g. due to costs. Dr. Rao disagreed. Revenue rose!"
    assert split_sentences(text) == [
        "Margins fell 4.2% in Q3, e.g. due to costs.",
        "Dr. Rao disagreed.",
        "Revenue rose!",
    ]


def test_header_footer_detection_needs_repetition():
    pages = [["Case Challenge | Team A", "Page 1"], ["Case Challenge | Team A", "Page 2"], ["Page 3"]]
    repeated = find_repeated_margin_lines(pages)
    assert "case challenge | team a" in repeated and "page #" in repeated
    assert find_repeated_margin_lines([["Only page"]]) == set()
    assert PAGE_NUMBER_RE.match("Page 3 of 10") and not PAGE_NUMBER_RE.match("Phase 3 results")


def _sent(i, n_words=20):
    return Sentence(sid=i, text=" ".join(f"w{i}_{k}" for k in range(n_words)) + ".", n_words=n_words)


def test_chunking_overlap_and_tail_merge():
    sents = [_sent(i) for i in range(10)]
    chunks = chunk_sentences(
        "d",
        sents,
        ChunkingConfig(max_tokens=80, min_tokens=30, overlap_sentences=1),
        count_tokens=lambda t: len(t.split()),
    )
    # every sentence is covered, consecutive chunks share exactly one sentence
    covered = {sid for c in chunks for sid in c.sentence_ids}
    assert covered == set(range(10))
    for a, b in zip(chunks, chunks[1:]):
        assert a.sentence_ids[-1] == b.sentence_ids[0]
    assert all(c.n_tokens <= 100 for c in chunks)


def test_index_returns_self_first_so_pipeline_must_exclude_it():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(6, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = VectorIndex(8)
    index.add(vecs, [("a", 0), ("a", 1), ("b", 0), ("b", 1), ("c", 0), ("c", 1)])
    scores, pos = index.search(vecs[:1], 3)
    assert index.keys[pos[0][0]] == ("a", 0) and abs(scores[0][0] - 1.0) < 1e-5


def test_number_normalisation_for_anchors():
    assert extract_numbers("three hundred scooters") == extract_numbers("300 scooters") == {"#300"}
    assert extract_numbers("1.2 lakh students") == extract_numbers("120,000 students")
    assert extract_numbers("one crore") == extract_numbers("ten million")
    assert anchors_agree({"#300"}) and anchors_agree({"vijay", "nagar"}) and not anchors_agree({"india's"})


def test_boilerplate_threshold_scales_with_corpus():
    cfg = BoilerplateConfig()
    assert df_threshold(5, cfg) == 3
    assert df_threshold(12, cfg) == 3
    assert df_threshold(100, cfg) == 25


def test_risk_bands():
    cfg = RiskConfig()
    assert risk_level(45, 0, None, cfg) == "HIGH"
    assert risk_level(12, 0, 10, cfg) == "MEDIUM"
    assert risk_level(0, 0, 95, cfg) == "MEDIUM"  # AI alone never reaches HIGH
    assert risk_level(12, 0, 95, cfg) == "HIGH"
    assert risk_level(2, 3, 20, cfg) == "LOW"


def test_english_ratio_flags_non_english():
    from integrity_engine.normalize import english_ratio

    assert english_ratio("The team argues that the pilot is a cheap way to learn about the market.") > 0.3
    assert english_ratio("La movilidad urbana en Pune está creciendo rápidamente y VoltRide debería expandirse.") < 0.12


def test_timestamp_format():
    assert format_timestamp(24.4) == "00:00:24" and format_timestamp(3725) == "01:02:05"
