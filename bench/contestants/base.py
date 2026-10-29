"""What every contestant provides, and how its scores become the taxonomy's."""

from __future__ import annotations

from typing import Protocol

from bench.taxonomy import FLAG, FLAG_CATEGORIES


class Contestant(Protocol):
    name: str
    # Part of the cache key: change it whenever the same text could get a different raw output.
    version: str
    languages: frozenset[str]
    # Shown in the report: what this contestant is.
    notes: str
    # SPDX id of the model's license, shown in the report.
    license: str

    def predict(self, texts: list[str], lang: str) -> list[dict[str, float]]:
        """Raw outputs under the model's own label names. Loads the model on first use."""

    def to_scores(self, raw: dict[str, float]) -> dict[str, float]:
        """Raw outputs mapped onto taxonomy keys. Pure, so scoring from the cache needs no model."""

    # Optional `unload()`: drop the loaded model. The runner calls it once a contestant is done, so the
    # next contestant fits on a small GPU.


def effective_scores(contestant, raw: dict[str, float]) -> dict[str, float]:
    scores = dict(contestant.to_scores(raw))
    if FLAG not in scores and all(c in scores for c in FLAG_CATEGORIES):
        scores[FLAG] = max(scores[c] for c in FLAG_CATEGORIES)
    return scores


def free_gpu() -> None:
    """Give back the GPU memory of a model that was just dropped, so the next one fits."""
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
