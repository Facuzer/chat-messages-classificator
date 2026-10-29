"""How a checkpoint is measured: the benchmark's own sources, splits and metrics, plus sentiment.

`flag` scores are per example id. Thresholds come from the benchmark's `calib` split and every reported
number from its `eval` split, so the trainer's numbers match the scoreboard's.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.model import SENTIMENTS
from bench.metrics import at_threshold, auroc, threshold_at_fpr
from bench.sources import split_of
from bench.taxonomy import FLAG, gold
from trainer.metrics import macro_f1

SPANISH_SOURCE = "suite-es"
# Below this many Spanish calibration negatives, 5% of them is not a threshold: every source is pooled.
MIN_CALIB_NEGATIVES = 20
SENTIMENT_DIR = Path(__file__).resolve().parent.parent / "data" / "sentiment"
# DynaSent round 2 test (English, CC BY 4.0, human majority label) and an unreviewed draft of the
# suite's Spanish sentiment, keyed by suite id. `?` marks a row the drafter found ambiguous.
DYNASENT_FILE = "dynasent-v1.1-round02-dynabench-test.jsonl"
DRAFT_FILE = "suite_sentiment_draft.csv"


def _scored(examples, scores, keep):
    return [(gold(e, FLAG), scores[e.id]) for e in examples if keep(e) and e.id in scores and gold(e, FLAG) is not None]


def choose_threshold(examples, scores: dict[str, float], split=split_of) -> tuple[float | None, dict]:
    """The `flag` score above which at most 5% of the Spanish calibration negatives fall."""
    spanish = _scored(examples, scores, lambda e: split(e) == "calib" and e.source == SPANISH_SOURCE)
    negatives = sum(not g for g, _ in spanish)
    source = SPANISH_SOURCE
    if negatives < MIN_CALIB_NEGATIVES:
        spanish = _scored(examples, scores, lambda e: split(e) == "calib")
        negatives, source = sum(not g for g, _ in spanish), "todas"
    if not spanish:
        return None, {"source": source, "negatives": 0}
    return threshold_at_fpr([g for g, _ in spanish], [s for _, s in spanish]), {"source": source, "negatives": negatives}


def flag_auroc_by_source(examples, scores: dict[str, float], split=split_of) -> dict[str, float | None]:
    out = {}
    for source in sorted({e.source for e in examples}):
        pairs = _scored(examples, scores, lambda e: split(e) == "eval" and e.source == source)
        if pairs:
            out[source] = auroc([g for g, _ in pairs], [s for _, s in pairs])
    return out


def operating_point(examples, scores: dict[str, float], threshold: float, source: str = SPANISH_SOURCE,
                    split=split_of) -> dict:
    pairs = _scored(examples, scores, lambda e: split(e) == "eval" and e.source == source)
    m = at_threshold([g for g, _ in pairs], [s for _, s in pairs], threshold)
    return {"recall": m["recall"], "fpr": m["fpr"]}


def load_sentiment_sets(directory: Path, suite_examples) -> dict[str, list[tuple[str, str]]]:
    """(text, gold sentiment) per set, for the files present in `directory`."""
    directory = Path(directory)
    sets = {}
    dynasent = directory / DYNASENT_FILE
    if dynasent.exists():
        rows = [json.loads(line) for line in dynasent.read_text(encoding="utf-8").splitlines() if line.strip()]
        sets["dynasent-r2-en"] = [(r["sentence"], r["gold_label"]) for r in rows if r["gold_label"] in SENTIMENTS]
    draft = directory / DRAFT_FILE
    if draft.exists():
        texts = {e.id.split(":", 1)[1]: e.text for e in suite_examples if e.source == SPANISH_SOURCE}
        with draft.open(encoding="utf-8-sig", newline="") as f:
            rows = [r for r in csv.DictReader(f, delimiter=";") if r["sentiment"] in SENTIMENTS and r["id"] in texts]
        sets["suite-es-draft"] = [(texts[r["id"]], r["sentiment"]) for r in rows]
    return sets


def sentiment_f1(predict, sets: dict[str, list[tuple[str, str]]], batch_size: int = 64) -> dict[str, float]:
    """Macro-F1 per set; `predict(texts)` returns raw outputs with `sentiment.*` probabilities."""
    out = {}
    for name, items in sets.items():
        predicted = []
        for start in range(0, len(items), batch_size):
            for raw in predict([text for text, _ in items[start:start + batch_size]]):
                predicted.append(max(SENTIMENTS, key=lambda s: raw[f"sentiment.{s}"]))
        out[name] = macro_f1([g for _, g in items], predicted)
    return out
