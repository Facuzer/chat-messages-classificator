"""The moderation taxonomy every benchmark source and checkpoint output is mapped onto.

docs/benchmark.md has the labeling policy behind each category.
"""

from __future__ import annotations

from dataclasses import dataclass

LANGS = ("es", "pt", "en")
CATEGORIES = ("insult", "threat", "identity_hate", "sexual_harassment", "profanity")
# What `flag` (should moderation act?) is built from. Profanity alone is not harmful: the chat policy
# decides what to do with it. Sexual harassment is scored on its own: only the Spanish suite labels it.
FLAG_CATEGORIES = ("insult", "threat", "identity_hate")
# Traits whose attack counts as identity_hate. Class, body and age insults are `insult`.
PROTECTED = ("race_ethnicity", "nationality", "religion", "gender", "sexual_orientation", "gender_identity", "disability")
TARGETS = ("player", "group", "none")
FLAG = "flag"
TARGETS_PLAYER = "targets_player"
SCORED = CATEGORIES + (FLAG, TARGETS_PLAYER)
NOT_FLAGGED = {category: False for category in FLAG_CATEGORIES}


@dataclass(frozen=True)
class Example:
    id: str
    source: str
    lang: str
    text: str
    # True / False per category; None or absent when the source does not say, so it is not scored.
    labels: dict
    # Who a harmful message attacks; None when the source does not say.
    target: str | None = None
    functionality: str | None = None
    author: str | None = None
    # Examples generated from one template share a cluster: bootstrap and splits keep them together.
    cluster: str | None = None

    def __post_init__(self):
        if self.lang not in LANGS:
            raise ValueError(f"{self.id}: unknown language {self.lang!r}")
        unknown = sorted(set(self.labels) - set(CATEGORIES))
        if unknown:
            raise ValueError(f"{self.id}: unknown categories {unknown}")
        bad = {k: v for k, v in self.labels.items() if not (v is None or isinstance(v, bool))}
        if bad:
            raise ValueError(f"{self.id}: labels must be True, False or None, got {bad}")
        if self.target is not None and self.target not in TARGETS:
            raise ValueError(f"{self.id}: unknown target {self.target!r}")
        if not self.text.strip():
            raise ValueError(f"{self.id}: empty text")

    @property
    def cluster_key(self) -> str:
        return self.cluster or self.id


def gold(example: Example, label: str) -> bool | None:
    """The correct answer for one scored label, or None when the example cannot score it."""
    if label == FLAG:
        values = [example.labels.get(c) for c in FLAG_CATEGORIES]
        if any(v is True for v in values):
            return True
        if all(v is False for v in values):
            return False
        return None
    if label == TARGETS_PLAYER:
        # On clean messages the target would only repeat `flag`, so it is scored on harmful ones only.
        if gold(example, FLAG) is not True or example.target is None:
            return None
        return example.target == "player"
    return example.labels.get(label)


def has_gold(example: Example) -> bool:
    return any(gold(example, label) is not None for label in SCORED)
