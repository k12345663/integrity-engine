"""Risk banding and plain-language explanations."""

from __future__ import annotations

from typing import Dict, List, Optional

from .config import RiskConfig


def risk_level(peer: float, external: float, ai: Optional[float], cfg: RiskConfig) -> str:
    """Combine the three signals into LOW / MEDIUM / HIGH.

    Overlap evidence is verifiable (passages are quoted), so it can drive a
    HIGH rating by itself. AI likelihood is a statistical indicator with a
    known false-positive problem (notably for non-native English writers), so
    on its own it can raise a submission to MEDIUM at most; combined with
    moderate overlap it can escalate to HIGH.
    """
    if peer >= cfg.high_peer or external >= cfg.high_external:
        return "HIGH"
    if ai is not None and ai >= cfg.ai_high and (peer >= cfg.medium_peer or external >= cfg.medium_external):
        return "HIGH"
    if peer >= cfg.medium_peer or external >= cfg.medium_external:
        return "MEDIUM"
    if ai is not None and ai >= cfg.ai_medium:
        return "MEDIUM"
    return "LOW"


def match_type(lexical_jaccard: float) -> str:
    if lexical_jaccard >= 0.7:
        return "verbatim"
    if lexical_jaccard >= 0.4:
        return "near-verbatim (light edits)"
    return "paraphrase (semantic match, low word overlap)"


def build_explanation(
    peer: float,
    external: float,
    ai: Dict[str, object],
    peer_sources: Dict[str, float],
    external_sources: Dict[str, float],
    strongest: List[str],
    boilerplate_words: int,
    risk: str,
) -> str:
    parts: List[str] = []
    if peer_sources:
        top = ", ".join(f"{k} ({v:.1f}%)" for k, v in sorted(peer_sources.items(), key=lambda kv: -kv[1])[:3])
        parts.append(f"{peer:.1f}% of the original text overlaps with other submissions: {top}.")
    else:
        parts.append("No meaningful overlap with other submissions.")
    if external_sources:
        top = ", ".join(f"{k} ({v:.1f}%)" for k, v in sorted(external_sources.items(), key=lambda kv: -kv[1])[:3])
        parts.append(f"{external:.1f}% matches external reference material: {top}.")
    if strongest:
        parts.append("Strongest evidence: " + "; ".join(strongest) + ".")
    if boilerplate_words:
        parts.append(
            f"{boilerplate_words} words of shared template/prompt text were detected and excluded from scoring."
        )
    p = ai.get("probability")
    if p is None:
        parts.append(f"AI-content likelihood not estimated: {ai.get('reason', 'n/a')}.")
    else:
        contrib = ai.get("logit_contributions") or {k: v - 50 for k, v in ai.get("components", {}).items()}
        if contrib:
            pick = max if p >= 50 else min
            driver = f"; strongest signal: {pick(contrib, key=contrib.get)}"
        else:
            driver = ""
        parts.append(f"AI-content likelihood {p:.0f}%{driver}; advisory only - corroborate before acting.")
    parts.append(f"Overall risk: {risk}.")
    return " ".join(parts)
