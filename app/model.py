"""A fine-tuned checkpoint: one mmBERT encoder whose outputs are the moderation categories and the sentiment.

The checkpoint is a standard `ModernBertForSequenceClassification` whose `id2label` names every output.
Category outputs are independent sigmoids (a message can insult and swear at once); the `sentiment.*`
outputs are one softmax. Both come from the same forward pass.
"""

from __future__ import annotations

import math
from pathlib import Path

from bench.taxonomy import CATEGORIES

SENTIMENTS = ("positive", "negative", "neutral")
SENTIMENT_PREFIX = "sentiment."
LABELS = CATEGORIES + tuple(f"{SENTIMENT_PREFIX}{s}" for s in SENTIMENTS)
MAX_LENGTH = 256


def output_names(config) -> list[str]:
    """The name of each output, in order: every one of LABELS, once."""
    names = [config.id2label[i] for i in range(len(config.id2label))]
    if sorted(names) != sorted(LABELS):
        raise ValueError(f"no es un checkpoint de este modelo: sus salidas son {names}")
    return names


def raw_outputs(logits: list[float], names: list[str]) -> dict[str, float]:
    raw = {n: 1 / (1 + math.exp(-x)) for n, x in zip(names, logits) if not n.startswith(SENTIMENT_PREFIX)}
    sentiment = [(n, x) for n, x in zip(names, logits) if n.startswith(SENTIMENT_PREFIX)]
    top = max(x for _, x in sentiment)
    total = sum(math.exp(x - top) for _, x in sentiment)
    raw.update({n: math.exp(x - top) / total for n, x in sentiment})
    return raw


def predict_raw(tokenizer, model, device, texts: list[str], max_length: int = MAX_LENGTH) -> list[dict[str, float]]:
    import torch

    names = output_names(model.config)
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits.float().cpu().tolist()
    return [raw_outputs(row, names) for row in logits]


def load(path: Path, device: str | None = None):
    """(tokenizer, model, device) for a local checkpoint directory, in eval mode."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, use_safetensors=True)
    output_names(model.config)
    return tokenizer, model.to(device).eval(), device
