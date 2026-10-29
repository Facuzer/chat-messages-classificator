"""Brazilian Portuguese sources: ToLD-Br (annotator votes per category) and OLID-BR (offensive, target,
categories).

When annotators had the chance to call a message an attack and only marked swearing, or nothing,
it counts as clean. That gives `flag` its hard negatives: swearing that hurts no one.
"""

from __future__ import annotations

from bench.taxonomy import Example

TOLDBR_REPO = "mteb/told-br"
OLIDBR_REPO = "dougtrajano/olid-br"
# The commits the preliminary scoreboard read (from the local HF cache on 2026-10-06).
TOLDBR_REVISION = "36b92223b1328b4f524705053a433b81489d501b"
OLIDBR_REVISION = "84b0d7dd4309be677a47c535632a9398ff1897bd"
TOLD_COLUMNS = ("homophobia", "obscene", "insult", "racism", "misogyny", "xenophobia")
TOLD_IDENTITY = ("homophobia", "racism", "misogyny", "xenophobia")
OLID_IDENTITY = ("racism", "sexism", "lgbtqphobia", "xenophobia", "religious_intolerance")
OLID_OTHER = ("health", "ideology", "insult", "other_lifestyle", "physical_aspects")


def _votes(value) -> int:
    votes = int(float(value))
    if not 0 <= votes <= 3:
        raise ValueError(f"ToLD-Br vote count out of range: {value!r}")
    return votes


def _majority(votes: int) -> bool | None:
    """Three annotators: two or more is a yes, none is a no, a single vote is too contested to score."""
    return True if votes >= 2 else False if votes == 0 else None


def toldbr_example(row: dict, index: int) -> Example:
    votes = {c: _votes(row[c]) for c in TOLD_COLUMNS}
    identity = [votes[c] for c in TOLD_IDENTITY]
    if not any(votes[c] for c in TOLD_COLUMNS if c != "obscene"):
        labels = {"insult": False, "threat": False, "identity_hate": False, "profanity": _majority(votes["obscene"])}
        target = "none"
    else:
        labels = {
            "insult": _majority(votes["insult"]),
            "identity_hate": True if any(v >= 2 for v in identity) else False if not any(identity) else None,
            "profanity": _majority(votes["obscene"]),
        }
        target = None
    return Example(id=f"toldbr:{index}", source="toldbr", lang="pt", text=row["text"].strip(), labels=labels, target=target)


def olidbr_example(row: dict) -> Example:
    offensive, targeted, kind = row["is_offensive"], row["is_targeted"], row["targeted_type"]
    if offensive not in ("OFF", "NOT"):
        raise ValueError(f"unexpected OLID-BR is_offensive {offensive!r}")
    if targeted not in ("TIN", "UNT"):
        raise ValueError(f"unexpected OLID-BR is_targeted {targeted!r}")
    if kind not in ("IND", "GRP", "OTH", None):
        raise ValueError(f"unexpected OLID-BR targeted_type {kind!r}")

    if offensive == "NOT":
        labels = {"insult": False, "threat": False, "identity_hate": False, "profanity": False}
        target = "none"
    else:
        attack_categories = [c for c in OLID_IDENTITY + OLID_OTHER if row[c]]
        if targeted == "UNT" and not attack_categories:
            labels = {"insult": False, "threat": False, "identity_hate": False,
                      "profanity": True if row["profanity_obscene"] else None}
            target = "none"
        else:
            identity = any(row[c] for c in OLID_IDENTITY)
            # Ours is an insult aimed at a person: OLID's covers groups (GRP) and things (OTH), and an
            # untargeted insult contradicts ours. IND is often a public figure. A False column does not
            # rule an insult out: body, lifestyle, ideology and health attacks are insults to us.
            not_personal = targeted == "UNT" or kind in ("GRP", "OTH")
            labels = {
                "insult": None if not_personal else (True if row["insult"] else None),
                # `health` mixes disability (protected) with other health insults.
                "identity_hate": True if identity else (None if row["health"] else False),
                "profanity": bool(row["profanity_obscene"]),
            }
            # An untargeted attack says nothing about who it hits; OTH is an organization or a thing.
            target = None if targeted == "UNT" else {"IND": "player", "GRP": "group"}.get(kind)
    return Example(id=f"olidbr:{row['id']}", source="olidbr", lang="pt", text=row["text"].strip(), labels=labels, target=target)


def load_toldbr() -> list[Example]:
    from datasets import load_dataset

    return [toldbr_example(row, i) for i, row in enumerate(load_dataset(TOLDBR_REPO, split="train", revision=TOLDBR_REVISION))]


def load_olidbr() -> list[Example]:
    from datasets import load_dataset

    return [olidbr_example(row) for row in load_dataset(OLIDBR_REPO, split="test", revision=OLIDBR_REVISION)]
