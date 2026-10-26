"""Pipeline configuration (defaults + optional YAML override)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ChunkingConfig:
    max_tokens: int = 200  # MiniLM truncates at 256 word pieces; 200 leaves headroom
    min_tokens: int = 40
    overlap_sentences: int = 1  # ~10-20% overlap at typical sentence lengths


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    device: Optional[str] = None  # None -> auto (cuda / mps / cpu)


@dataclass
class MatchingConfig:
    top_k: int = 10  # neighbours retrieved per query chunk
    chunk_candidate_threshold: float = 0.55  # chunk pairs worth aligning
    # Direct path: a sentence pair with bi-encoder cosine >= this is a match
    # (verbatim copies, light edits, ASR transcripts of read-out text).
    sentence_threshold: float = 0.80
    # Verified path (paraphrase): windows of N sentences shortlisted by
    # bi-encoder cosine, then confirmed by a cross-encoder STS model.
    cross_encoder_model: Optional[str] = "cross-encoder/stsb-distilroberta-base"
    window_sizes: List[int] = field(default_factory=lambda: [2, 3])  # sentences per window
    window_prefilter: float = 0.60
    cross_encoder_threshold: float = 0.60
    window_cosine_fallback: float = 0.85  # used only if the cross-encoder is unavailable
    # Verified paraphrase evidence must share a rare specific detail (number or
    # named entity) unless the cross-encoder is very confident.
    require_anchor: bool = True
    anchor_max_doc_fraction: float = 0.10
    cross_encoder_strong: float = 0.80
    # Leave-one-out attribution: a sentence in a verified window is only
    # attributed if dropping it does not raise the window's score by more than
    # this margin (i.e. it is part of the match, not a bystander).
    loo_attribution: bool = True
    loo_margin: float = 0.0
    min_sentence_words: int = 6  # shorter sentences are too generic to match
    # A passage must span >= 2 aligned sentences, unless a single sentence is a
    # near-exact match: one similar sentence between two answers to the same
    # case is weak evidence (e.g. "We recommend expanding to Nagpur first").
    min_passage_sentences: int = 2
    single_sentence_threshold: float = 0.90
    single_sentence_min_words: int = 12
    max_matches_reported: int = 15
    index_type: str = "flat"  # "flat" (exact) | "hnsw" (approximate)


@dataclass
class BoilerplateConfig:
    enabled: bool = True
    near_duplicate_threshold: float = 0.92
    min_doc_frequency: int = 3
    min_doc_fraction: float = 0.25
    template_dir: Optional[str] = None  # organiser-supplied prompt texts


@dataclass
class ExternalConfig:
    reference_dir: str = "reference_corpus"
    online: bool = False  # query the Wikipedia API in addition to the local KB
    online_timeout: float = 8.0
    online_cache_dir: str = ".cache/online_reference"
    online_queries_per_doc: int = 3


@dataclass
class AIDetectionConfig:
    enabled: bool = True
    perplexity_model: str = "gpt2"
    classifier_model: Optional[str] = "openai-community/roberta-base-openai-detector"
    min_words: int = 120
    max_tokens: int = 2048  # per document budget for the LM
    calibration_file: Optional[str] = "config/ai_calibration.json"


@dataclass
class ASRConfig:
    model_name: str = "base.en"
    silence_rms_threshold: float = 0.003
    no_speech_prob_threshold: float = 0.6
    min_words: int = 5


@dataclass
class RiskConfig:
    high_peer: float = 30.0
    high_external: float = 30.0
    medium_peer: float = 10.0
    medium_external: float = 10.0
    ai_high: float = 80.0
    ai_medium: float = 60.0


@dataclass
class Config:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    boilerplate: BoilerplateConfig = field(default_factory=BoilerplateConfig)
    external: ExternalConfig = field(default_factory=ExternalConfig)
    ai: AIDetectionConfig = field(default_factory=AIDetectionConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    supported_extensions: Dict[str, str] = field(
        default_factory=lambda: {
            ".pdf": "pdf",
            ".docx": "docx",
            ".txt": "text",
            ".md": "text",
            ".mp4": "media",
            ".mov": "media",
            ".mkv": "media",
            ".webm": "media",
            ".m4a": "media",
            ".mp3": "media",
            ".wav": "media",
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _merge(obj: Any, overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    """Load defaults, then a YAML file, then ``section.key=value`` overrides."""
    import yaml

    cfg = Config()
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _merge(cfg, data)
    for item in overrides or []:
        dotted, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        *parents, leaf = dotted.split(".")
        target: Any = cfg
        for p in parents:
            target = getattr(target, p)
        if not hasattr(target, leaf):
            raise ValueError(f"Unknown config key: {dotted}")
        setattr(target, leaf, value)
    return cfg


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
