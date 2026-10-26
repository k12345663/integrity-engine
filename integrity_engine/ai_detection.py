"""AI-generated text likelihood from four transparent, reproducible signals.

1. Perplexity (GPT-2). LLM output sits in high-probability regions of a
   language model, so its perplexity under GPT-2 is low.
2. Burstiness. Human writing mixes predictable and surprising sentences; the
   standard deviation of per-sentence log-perplexity is higher for humans.
3. An open classifier (``roberta-base-openai-detector``). It was trained on
   GPT-2 output and is weak on modern LLMs; the calibration decides its weight.
4. Stylometric markers: frequency of phrases over-used by chat assistants
   ("delve", "in today's rapidly evolving", "furthermore", ...).

The signals are combined by a logistic regression fitted on a *public,
external* labelled corpus (HC3: human vs ChatGPT answers) by
``scripts/calibrate_ai_detector.py`` - never on our own evaluation set.
Every sub-score is returned so reviewers can see why a number is high. The
output is an *indicator* for human review, never proof.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import AIDetectionConfig
from .models import Document

log = logging.getLogger(__name__)

# Phrases strongly over-represented in assistant-style prose. Counted per
# 100 words; kept short and generic on purpose.
AI_MARKERS = [
    r"\bdelv(e|es|ing)\b",
    r"\btapestry\b",
    r"\bin today's (rapidly )?(evolving|changing|dynamic)\b",
    r"\bever-evolving\b",
    r"\blandscape\b",
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\bin conclusion\b",
    r"\bit is (important|worth|crucial) to note\b",
    r"\bplays? a (pivotal|crucial|vital) role\b",
    r"\bseamless(ly)?\b",
    r"\bleverag(e|es|ing)\b",
    r"\brobust\b",
    r"\bholistic\b",
    r"\bfoster(s|ing)?\b",
    r"\bnavigat(e|ing) the\b",
    r"\bunderscore(s|d)?\b",
    r"\bmultifaceted\b",
    r"\bparamount\b",
    r"\bstrategic(ally)? (positioned|imperative)\b",
    r"\bsynerg(y|ies|istic)\b",
    r"\bpivotal\b",
    r"\bcomprehensive\b",
    r"\bharness(es|ing)?\b",
    r"\bin summary\b",
    r"\boverall,\b",
]
_MARKER_RE = re.compile("|".join(AI_MARKERS), re.IGNORECASE)
Z_CLIP = 3.0
MIN_SENTENCE_WORDS = 6  # headings / table rows distort per-sentence statistics


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class AIDetector:
    def __init__(self, cfg: AIDetectionConfig):
        self.cfg = cfg
        self._lm = None
        self._clf = None

    # -- models (lazy) -------------------------------------------------
    def _load_lm(self):
        if self._lm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.cfg.perplexity_model)
            model = AutoModelForCausalLM.from_pretrained(self.cfg.perplexity_model).eval()
            self._lm = (tok, model)
        return self._lm

    def _load_clf(self):
        if self._clf is None and self.cfg.classifier_model:
            try:
                from transformers import pipeline

                self._clf = pipeline(
                    "text-classification",
                    model=self.cfg.classifier_model,
                    device="cpu",
                    truncation=True,
                    max_length=512,
                )
            except Exception as exc:
                log.warning("AI classifier unavailable (%s); continuing without it", exc)
                self.cfg.classifier_model = None
        return self._clf

    # -- signals -------------------------------------------------------
    def perplexity_signals(self, sentences: List[str]) -> Dict[str, float]:
        """Token-level NLL under GPT-2, aggregated per document and per sentence."""
        import torch

        tok, model = self._load_lm()
        text = " ".join(sentences)
        enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=self.cfg.max_tokens)
        ids = torch.tensor(enc["input_ids"])
        offsets = enc["offset_mapping"]
        window = model.config.n_positions  # 1024 for GPT-2
        stride = window // 2
        n = len(ids)
        nll = np.full(n, np.nan)  # nll[t] = -log p(token t | tokens < t)
        prev_end = 1
        with torch.no_grad():
            for begin in range(0, n, stride):
                end = min(begin + window, n)
                logits = model(ids[begin:end].unsqueeze(0)).logits[0]
                targets = torch.arange(max(prev_end, begin + 1), end)
                if len(targets):
                    loss = torch.nn.functional.cross_entropy(
                        logits[targets - begin - 1], ids[targets], reduction="none"
                    )
                    nll[targets.numpy()] = loss.numpy()
                prev_end = end
                if end == n:
                    break
        token_nll = nll[~np.isnan(nll)]

        # Map tokens to sentences via character offsets.
        bounds, pos = [], 0
        for s in sentences:
            bounds.append((pos, pos + len(s)))
            pos += len(s) + 1
        per_sentence: List[List[float]] = [[] for _ in sentences]
        si = 0
        for t, (a, _) in enumerate(offsets):
            if t == 0 or np.isnan(nll[t]):
                continue
            while si < len(bounds) - 1 and a >= bounds[si][1]:
                si += 1
            per_sentence[si].append(nll[t])
        sent_means = [float(np.mean(v)) for v in per_sentence if len(v) >= 4]
        return {
            "log_perplexity": float(np.mean(token_nll)) if len(token_nll) else float("nan"),
            "perplexity": float(np.exp(np.mean(token_nll))) if len(token_nll) else float("nan"),
            "burstiness": float(np.std(sent_means)) if len(sent_means) > 2 else float("nan"),
            "tokens_scored": int(len(token_nll)),
        }

    def classifier_probability(self, text: str) -> Optional[float]:
        clf = self._load_clf()
        if clf is None:
            return None
        words_ = text.split()
        windows = [" ".join(words_[i : i + 300]) for i in range(0, len(words_), 300)][:6]
        probs = []
        for out in clf(windows):
            label = out["label"].lower()
            p = out["score"]
            # The OpenAI detector labels machine text "Fake" and human text "Real".
            probs.append(p if label in ("fake", "label_0", "ai", "machine") else 1.0 - p)
        return float(np.mean(probs)) if probs else None

    @staticmethod
    def marker_rate(text: str) -> float:
        n = max(1, len(text.split()))
        return 100.0 * len(_MARKER_RE.findall(text)) / n

    # -- combination ---------------------------------------------------
    FEATURES = ("log_perplexity", "burstiness", "marker_rate", "classifier")

    def calibration(self) -> Optional[Dict[str, object]]:
        """Logistic-regression weights fitted by scripts/calibrate_ai_detector.py."""
        if not hasattr(self, "_calibration"):
            self._calibration = None
            path = Path(self.cfg.calibration_file) if self.cfg.calibration_file else None
            if path and not path.is_absolute():
                path = Path(__file__).resolve().parent.parent / path
            if path and path.exists():
                self._calibration = json.loads(path.read_text())
        return self._calibration

    def combine(self, sig: Dict[str, Optional[float]]) -> Dict[str, object]:
        """Map raw signals to a probability with a calibrated logistic model.

        Each feature is standardised with the calibration set's mean/std,
        clipped to +-3 sd, and contributes ``coef * z`` logits; missing
        features contribute 0 (mean imputation). Contributions are reported so
        the score is explainable.
        """
        cal = self.calibration()
        if cal is None:
            return self._combine_uncalibrated(sig)
        logit = cal["intercept"]
        contrib: Dict[str, float] = {}
        for name, mean, std, coef in zip(cal["features"], cal["mean"], cal["std"], cal["coef"]):
            x = sig.get(name)
            z = 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else (x - mean) / (std or 1.0)
            # Clip to the calibration range: a linear model must not extrapolate
            # to 100% just because a feature is far outside what it was fitted on.
            z = max(-Z_CLIP, min(Z_CLIP, z))
            contrib[name] = round(coef * z, 3)
            logit += coef * z
        return {
            "probability": round(100.0 * _sigmoid(logit), 1),
            "method": f"logistic regression calibrated on {cal['source']} (CV AUROC {cal['cv_auroc']:.2f})",
            "logit_contributions": contrib,
            "intercept": round(cal["intercept"], 3),
        }

    @staticmethod
    def _combine_uncalibrated(sig: Dict[str, Optional[float]]) -> Dict[str, object]:
        """Fallback with fixed, documented weights when no calibration exists."""
        parts: Dict[str, float] = {}
        weights = {"log_perplexity": 0.4, "marker_rate": 0.4, "burstiness": 0.2}
        lp, bu = sig.get("log_perplexity"), sig.get("burstiness")
        if lp is not None and not math.isnan(lp):
            parts["log_perplexity"] = _sigmoid(3.0 * (math.log(40.0) - lp))
        if bu is not None and not math.isnan(bu):
            parts["burstiness"] = _sigmoid(4.0 * (1.0 - bu))
        parts["marker_rate"] = _sigmoid(2.5 * (sig.get("marker_rate", 0.0) - 1.2))
        total = sum(weights[k] for k in parts)
        score = sum(parts[k] * weights[k] for k in parts) / total
        return {
            "probability": round(100.0 * score, 1),
            "method": "fixed weights (uncalibrated)",
            "components": {k: round(100.0 * v, 1) for k, v in parts.items()},
        }

    def signals(self, sents: List[str], name: str = "text") -> Dict[str, Optional[float]]:
        """Raw feature values for a list of sentences (also used by calibration)."""
        sents = [s for s in sents if len(s.split()) >= MIN_SENTENCE_WORDS]
        text = " ".join(sents)
        signals: Dict[str, Optional[float]] = {"marker_rate": self.marker_rate(text)}
        try:
            signals.update(self.perplexity_signals(sents))
        except Exception as exc:
            log.warning("%s: perplexity model failed (%s)", name, exc)
        try:
            signals["classifier"] = self.classifier_probability(text)
        except Exception as exc:
            log.warning("%s: classifier failed (%s)", name, exc)
        return signals

    def analyse(self, doc: Document) -> Dict[str, object]:
        sents = [s.text for s in doc.scoring_sentences()]
        text = " ".join(sents)
        n_words = len(text.split())
        if not self.cfg.enabled:
            return {"probability": None, "reason": "AI detection disabled"}
        if n_words < self.cfg.min_words:
            return {
                "probability": None,
                "reason": f"Insufficient original text for a reliable estimate "
                f"({n_words} < {self.cfg.min_words} words)",
            }
        signals = self.signals(sents, doc.doc_id)
        result = self.combine(signals)
        result["signals"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in signals.items()}
        result["words_analysed"] = n_words
        if doc.modality == "media":
            result["caveat"] = "Spoken transcript: ASR normalisation lowers perplexity; treat with extra caution"
        return result
