"""The training set: public Portuguese corpora plus the app's labeled Spanish examples.

No public row repeats a benchmark text, so the benchmark measures texts the model never saw. The app's
examples are the exception: most of them are also suite rows, so the checkpoint records which texts it
trained on and the benchmark leaves those out (see bench.contestants.finetuned).
"""

from __future__ import annotations

import random

from app.examples import normalize_text, text_key  # noqa: F401  (text_key is part of this module's API)
from bench.taxonomy import CATEGORIES, FLAG_CATEGORIES, Example, has_gold

# The Spanish rows are few next to the Portuguese ones, so each is seen this many times per epoch.
SPANISH_UPSAMPLE = 3
# ToLD-Br is mostly clean: all its positives go in, and this many of its negatives.
TOLD_NEGATIVES = 6000
# Fixed, so every run samples the same texts and reuses their sentiment labels; `--seed` varies training only.
DATA_SEED = 0


def app_labels(example: dict, suite_labels: dict[str, dict]) -> dict:
    """Taxonomy labels for an app example (`insult` / `clean`, labeled under the app's binary policy).

    A text that is also a suite row takes the suite's per-category labels. Otherwise `insult` only says
    the message is an insult, and `clean` rules out every `flag` category. Neither says anything about
    profanity or sexual harassment, which stay unknown and out of the loss.
    """
    suite = suite_labels.get(text_key(example["text"]))
    if suite is not None:
        return {c: suite.get(c) for c in CATEGORIES}
    if example["label"] == "insult":
        return {c: True if c == "insult" else None for c in CATEGORIES}
    if example["label"] == "clean":
        return {c: False if c in FLAG_CATEGORIES else None for c in CATEGORIES}
    raise ValueError(f"etiqueta desconocida {example['label']!r} en el ejemplo {example['text']!r}")


def _row(text: str, lang: str, source: str, labels: dict) -> dict:
    return {"text": text, "lang": lang, "source": source, "labels": {c: labels.get(c) for c in CATEGORIES}}


def build_rows(app_examples: list[dict], benchmark: list[Example], told: list[Example], olid: list[Example],
               told_negatives: int = TOLD_NEGATIVES) -> tuple[list[dict], dict]:
    """Shuffled training rows and what went into them.

    `benchmark` is every example of every benchmark source, `told` all of ToLD-Br, `olid` the OLID-BR
    train split and `app_examples` the app's stored examples (only labeled `train` ones are used).
    """
    rng = random.Random(DATA_SEED)
    benchmark_texts = {normalize_text(e.text) for e in benchmark}
    in_benchmark = {e.id for e in benchmark}
    suite_labels = {text_key(e.text): e.labels for e in benchmark if e.source == "suite-es"}

    told = [e for e in told if has_gold(e) and e.id not in in_benchmark]
    positives = [e for e in told if any(e.labels.values())]
    negatives = [e for e in told if not any(e.labels.values())]
    told = positives + rng.sample(negatives, min(told_negatives, len(negatives)))

    rows, dropped, kept = [], 0, {"toldbr": 0, "olidbr": 0}
    for e in told + olid:
        if normalize_text(e.text) in benchmark_texts:
            dropped += 1
            continue
        rows.append(_row(e.text, e.lang, e.source, e.labels))
        kept[e.source] += 1

    spanish = [e for e in app_examples if e["status"] == "labeled" and e["split"] == "train"]
    for e in spanish:
        rows += [_row(e["text"], "es", "app", app_labels(e, suite_labels))] * SPANISH_UPSAMPLE

    rng.shuffle(rows)
    stats = {**kept, "spanish": len(spanish), "dropped": dropped,
             "trained_texts": sorted({text_key(e["text"]) for e in spanish})}
    return rows, stats


def load_inputs(app_examples: list[dict]) -> tuple[list[dict], list[Example], list[Example], list[Example]]:
    """build_rows' arguments, from the Hugging Face cache (pinned revisions) and the suite."""
    from datasets import load_dataset

    from bench.sources import SOURCES, load_sources
    from bench.sources.portuguese import OLIDBR_REPO, OLIDBR_REVISION, load_toldbr, olidbr_example

    # Unreviewed suite rows count too: no training text may repeat any of them.
    benchmark = load_sources(list(SOURCES), include_unreviewed=True, log=lambda _: None)
    olid = [olidbr_example(r) for r in load_dataset(OLIDBR_REPO, split="train", revision=OLIDBR_REVISION)]
    return app_examples, benchmark, load_toldbr(), olid
