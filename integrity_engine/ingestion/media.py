"""Audio/video ingestion: ffmpeg audio extraction + Whisper transcription.

ffmpeg comes from the ``imageio-ffmpeg`` wheel, so no system install is
needed. Audio is decoded to 16 kHz mono PCM in memory and handed to Whisper
as a NumPy array.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Optional

import numpy as np

from ..config import ASRConfig
from ..models import Document, TextBlock
from ..normalize import clean_text, words
from .errors import ExtractionError

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_whisper_models = {}


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: str) -> dict:
    """Return duration (s) and whether an audio stream exists, via ``ffmpeg -i``."""
    proc = subprocess.run([_ffmpeg(), "-hide_banner", "-i", path], capture_output=True, text=True)
    err = proc.stderr
    if "Invalid data found" in err or "moov atom not found" in err or "No such file" in err:
        raise ExtractionError("Media file is corrupted or unreadable")
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", err)
    duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None
    return {"duration": duration, "has_audio": bool(re.search(r"Stream #\S+.*Audio:", err))}


def load_audio(path: str) -> np.ndarray:
    cmd = [
        _ffmpeg(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise ExtractionError(f"Audio decoding failed: {proc.stderr.decode(errors='ignore').strip()[:200]}")
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


def _model(name: str):
    if name not in _whisper_models:
        import whisper

        _whisper_models[name] = whisper.load_model(name, device="cpu")
    return _whisper_models[name]


def extract_media(doc: Document, cfg: Optional[ASRConfig] = None) -> None:
    cfg = cfg or ASRConfig()
    info = probe(doc.path)
    doc.meta["duration_seconds"] = round(info["duration"], 2) if info["duration"] else None
    if not info["has_audio"]:
        raise ExtractionError("Video has no audio track; nothing to transcribe")

    audio = load_audio(doc.path)
    if audio.size == 0:
        raise ExtractionError("Audio track is empty")
    rms = float(np.sqrt(np.mean(np.square(audio))))
    doc.meta["audio_rms"] = round(rms, 5)
    if rms < cfg.silence_rms_threshold:
        raise ExtractionError(f"Audio track is silent (RMS {rms:.5f}); no speech to transcribe")

    result = _model(cfg.model_name).transcribe(
        audio, language="en", fp16=False, condition_on_previous_text=False, temperature=0.0
    )
    dropped = 0
    for i, seg in enumerate(result.get("segments", []), start=1):
        text = clean_text(seg.get("text", ""))
        # Whisper hallucinates on noise/silence; such segments have a high
        # no-speech probability or a degenerate (repetitive) compression ratio.
        if (
            not text
            or seg.get("no_speech_prob", 0) > cfg.no_speech_prob_threshold
            or seg.get("compression_ratio", 0) > 2.4
        ):
            dropped += 1
            continue
        doc.blocks.append(TextBlock(text=text, start=float(seg["start"]), end=float(seg["end"]), paragraph=i))

    doc.meta["asr_model"] = f"whisper-{cfg.model_name}"
    doc.meta["asr_segments_dropped"] = dropped
    n_words = sum(len(words(b.text)) for b in doc.blocks)
    if n_words < cfg.min_words:
        raise ExtractionError(f"No intelligible speech detected ({n_words} words transcribed)")
