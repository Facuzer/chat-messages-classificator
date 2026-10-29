from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.checkpoints import ActiveCheckpoint, checkpoint_problem
from app.jsonfile import read_json
from app.model import SENTIMENTS
from bench.taxonomy import CATEGORIES, FLAG_CATEGORIES


@dataclass(frozen=True)
class Verdict:
    label: str
    # The probability of `label`.
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class Flag:
    # The highest of the `flag` categories' probabilities, and which category it is.
    score: float
    category: str
    threshold: float


@dataclass(frozen=True)
class Classification:
    flagged: bool
    flag: Flag
    categories: dict[str, float]
    sentiment: Verdict
    model: str
    latency_ms: int


def _sentiment(raw: dict[str, float]) -> Verdict:
    probabilities = {s: raw[f"sentiment.{s}"] for s in SENTIMENTS}
    label = max(SENTIMENTS, key=probabilities.__getitem__)
    return Verdict(label=label, confidence=probabilities[label], probabilities=probabilities)


def to_classification(raw: dict[str, float], threshold: float, model: str, latency_ms: int) -> Classification:
    categories = {c: raw[c] for c in CATEGORIES}
    category = max(FLAG_CATEGORIES, key=categories.__getitem__)
    return Classification(
        # The threshold is the score of a calibration negative, so flagging starts above it.
        flagged=categories[category] > threshold,
        flag=Flag(score=categories[category], category=category, threshold=threshold),
        categories=categories,
        sentiment=_sentiment(raw),
        model=model,
        latency_ms=latency_ms,
    )


class Classifier:
    def __init__(self, predict, threshold: float, model: str, checkpoint: dict | None = None):
        # `predict(texts)` returns the checkpoint's raw outputs (app.model.predict_raw) per text.
        self.predict = predict
        self.threshold = threshold
        self.model = model
        # The active checkpoint as app.checkpoints.ActiveCheckpoint.get returns it.
        self.checkpoint = checkpoint
        # One inference at a time: the endpoint runs in a thread pool and every request shares the model.
        self._lock = threading.Lock()

    def classify(self, text: str) -> Classification:
        with self._lock:
            start = time.perf_counter()
            [raw] = self.predict([text])
            latency_ms = round((time.perf_counter() - start) * 1000)
        return to_classification(raw, self.threshold, self.model, latency_ms)


def threshold_of(checkpoint: Path) -> float:
    """FLAG_THRESHOLD when set, else the threshold the trainer chose for this checkpoint."""
    override = os.environ.get("FLAG_THRESHOLD")
    if override:
        return float(override)
    return float(read_json(Path(checkpoint) / "training_meta.json", {})["threshold"])


def load_classifier(active: ActiveCheckpoint | None = None, loader=None, device: str | None = None) -> Classifier:
    """The active checkpoint, on MODEL_DEVICE (CPU unless set): the size of machine it has to run on."""
    from app.model import load, predict_raw

    checkpoint = (active or ActiveCheckpoint()).get()
    if checkpoint is None:
        raise RuntimeError("No hay un modelo activo. Entrená uno con: uv run python scripts/finetune.py")
    path = checkpoint["path"]
    problem = checkpoint_problem(path)
    if problem:
        raise RuntimeError(
            f"El modelo activo ({path.name}) {problem}. Entrená uno con uv run python scripts/finetune.py "
            "o activá otro con scripts/activate_model.py."
        )
    tokenizer, model, device = (loader or load)(path, device or os.environ.get("MODEL_DEVICE") or "cpu")
    return Classifier(lambda texts: predict_raw(tokenizer, model, device, texts), threshold=threshold_of(path),
                      model=path.name, checkpoint=checkpoint)
