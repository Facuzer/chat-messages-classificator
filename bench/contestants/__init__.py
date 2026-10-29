"""The checkpoints the benchmark compares. Building one loads nothing; each model loads on its first prediction."""

from __future__ import annotations

from bench.contestants.base import Contestant, effective_scores  # noqa: F401
from bench.contestants.finetuned import FineTunedContestant, checkpoint_factories  # noqa: F401
