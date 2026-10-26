#!/usr/bin/env python
"""Pre-download every model so the first pipeline run works offline (~1.4 GB).

python scripts/download_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integrity_engine.config import Config  # noqa: E402


def main() -> None:
    cfg = Config()
    import whisper
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    print("embedding model:", cfg.embedding.model_name)
    SentenceTransformer(cfg.embedding.model_name)
    print("cross-encoder:", cfg.matching.cross_encoder_model)
    CrossEncoder(cfg.matching.cross_encoder_model)
    print("perplexity LM:", cfg.ai.perplexity_model)
    AutoTokenizer.from_pretrained(cfg.ai.perplexity_model)
    AutoModelForCausalLM.from_pretrained(cfg.ai.perplexity_model)
    if cfg.ai.classifier_model:
        print("AI classifier:", cfg.ai.classifier_model)
        AutoTokenizer.from_pretrained(cfg.ai.classifier_model)
        AutoModelForSequenceClassification.from_pretrained(cfg.ai.classifier_model)
    print("whisper:", cfg.asr.model_name)
    whisper.load_model(cfg.asr.model_name, device="cpu")
    print("done")


if __name__ == "__main__":
    main()
