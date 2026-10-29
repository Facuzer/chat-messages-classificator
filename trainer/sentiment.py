"""Sentiment labels for the training texts, drafted once, offline, by an LLM.

No permissively licensed Spanish or Portuguese sentiment corpus exists, so Qwen3-4B (trainer.labeler)
labels every training text with a probability per sentiment. The labels are kept on disk by normalized text and are never
redone: retraining only labels texts that are new.
"""

from __future__ import annotations

import json
from pathlib import Path

from trainer.data import text_key

SENTIMENTS = ("positive", "negative", "neutral")
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "training" / "sentiment_labels.jsonl"


class SentimentLabels:
    """A JSONL file, one `{"key", "probs", "labeler"}` line per text, appended as labels arrive."""

    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)
        self._probs = None

    def _load(self) -> dict[str, dict[str, float]]:
        if self._probs is None:
            self._probs = {}
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a run killed mid-write leaves a partial line
                    self._probs[row["key"]] = row["probs"]
        return self._probs

    def get(self, text: str) -> dict[str, float] | None:
        return self._load().get(text_key(text))

    def add(self, rows: list[tuple[str, dict[str, float]]], labeler: str) -> None:
        probs = self._load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_newline = False
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as f:
                f.seek(-1, 2)
                needs_newline = f.read(1) != b"\n"
        payload = "".join(json.dumps({"key": text_key(text), "probs": p, "labeler": labeler}) + "\n" for text, p in rows)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            if needs_newline:
                f.write("\n")
            f.write(payload)
        probs.update({text_key(text): p for text, p in rows})


def sentiment_labeler(device: str | None = None):
    from trainer.labeler import SentimentLabeler

    return SentimentLabeler(device=device)


def label_missing(texts: list[str], labels: SentimentLabels, labeler, batch_size: int = 16, log=print) -> int:
    """Label every text that has no label yet; returns how many were labeled."""
    todo = {}
    for text in texts:
        if labels.get(text) is None:
            todo.setdefault(text_key(text), text)
    todo = list(todo.values())
    name = f"{labeler.name}@{labeler.version}"
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        raw = labeler.predict(batch, "es")
        labels.add([(text, {s: r[f"sentiment.{s}"] for s in SENTIMENTS}) for text, r in zip(batch, raw)], name)
        done = start + len(batch)
        if done % (batch_size * 50) < batch_size or done == len(todo):
            log(f"  sentimiento: {done}/{len(todo)} textos etiquetados")
    return len(todo)
