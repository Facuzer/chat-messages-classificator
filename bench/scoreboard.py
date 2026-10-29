"""From raw predictions to the numbers in the report.

Each contestant gets one operating threshold per label, chosen on the calibration split where 5% of the
negatives get flagged. Everything reported is measured on the eval split, per source: pooling sources
would mostly measure which dataset a text came from.
"""

from __future__ import annotations

from collections import defaultdict

from bench.contestants.base import effective_scores
from bench.metrics import auroc, paired_auroc_diff, slice_metrics, threshold_at_fpr
from bench.sources import split_of
from bench.taxonomy import FLAG, SCORED, gold

MIN_PER_CLASS = 30


def slice_keys(example) -> list[str]:
    """The source, plus source/author for the suite, so generated and human-written rows show apart."""
    keys = [example.source]
    if example.author:
        keys.append(f"{example.source}/{example.author}")
    return keys


def functionality_label(source: str, functionality: str) -> str:
    # HateCheck measures hate speech; the suite's functionalities are about moderation as a whole,
    # except sexual harassment, which `flag` leaves out and is scored on its own label.
    if source.startswith("hatecheck"):
        return "identity_hate"
    return "sexual_harassment" if functionality == "sexual_harassment" else FLAG


def _pairs(examples, scores: dict, label: str):
    """(example, gold, score) for every example the contestant scored and that has gold for `label`."""
    for e in examples:
        value, expected = scores.get(e.id, {}).get(label), gold(e, label)
        if value is not None and expected is not None:
            yield e, expected, value


def thresholds(examples, scores_by_contestant, split=split_of) -> dict[str, dict[str, float | None]]:
    calib = [e for e in examples if split(e) == "calib"]
    out = {}
    for name, scores in scores_by_contestant.items():
        out[name] = {}
        for label in SCORED:
            pairs = list(_pairs(calib, scores, label))
            out[name][label] = threshold_at_fpr([g for _, g, _ in pairs], [s for _, _, s in pairs]) if pairs else None
    return out


def coverage(examples, split=split_of) -> list[dict]:
    counts = defaultdict(lambda: [0, 0])
    for e in examples:
        if split(e) != "eval":
            continue
        for label in SCORED:
            expected = gold(e, label)
            if expected is None:
                continue
            for key in slice_keys(e):
                counts[(key, e.lang, label)][0 if expected else 1] += 1
    return [{"source": s, "lang": lang, "label": label, "positives": p, "negatives": n}
            for (s, lang, label), (p, n) in sorted(counts.items())]


def slices(examples, scores_by_contestant, thr, n_boot=1000, seed=0, split=split_of) -> list[dict]:
    evaluated = [e for e in examples if split(e) == "eval"]
    rows = []
    for name, scores in scores_by_contestant.items():
        groups = defaultdict(lambda: ([], [], []))
        for label in SCORED:
            for e, expected, value in _pairs(evaluated, scores, label):
                for key in slice_keys(e):
                    golds, values, clusters = groups[(key, e.lang, label)]
                    golds.append(expected)
                    values.append(value)
                    clusters.append(e.cluster_key)
        for (key, lang, label), (golds, values, clusters) in sorted(groups.items()):
            m = slice_metrics(golds, values, clusters, thr[name][label], n_boot=n_boot, seed=seed)
            m["enough"] = m["positives"] >= MIN_PER_CLASS and m["n"] - m["positives"] >= MIN_PER_CLASS
            rows.append({"contestant": name, "source": key, "lang": lang, "label": label, **m})
    return rows


def comparisons(examples, scores_by_contestant, n_boot=1000, seed=0, split=split_of) -> list[dict]:
    """Per source, `flag` AUROC of each contestant minus the best one's, on the items all of them scored.

    Sources with fewer than MIN_PER_CLASS eval examples of either class are left out, like their "pocos datos" cells.
    """
    by_source = defaultdict(list)
    for e in examples:
        if split(e) == "eval" and gold(e, FLAG) is not None:
            for key in slice_keys(e):
                by_source[key].append(e)
    rows = []
    for key, items in sorted(by_source.items()):
        golds = [gold(e, FLAG) for e in items]
        if min(golds.count(True), golds.count(False)) < MIN_PER_CLASS:
            continue
        names = [n for n, s in scores_by_contestant.items() if all(FLAG in s.get(e.id, {}) for e in items)]
        values = {n: [scores_by_contestant[n][e.id][FLAG] for e in items] for n in names}
        aurocs = {n: auroc(golds, values[n]) for n in names}
        names = [n for n in names if aurocs[n] is not None]
        if len(names) < 2:
            continue
        best = max(names, key=lambda n: aurocs[n])
        clusters = [e.cluster_key for e in items]
        for n in names:
            if n != best:
                diff = paired_auroc_diff(golds, values[n], values[best], clusters, n_boot=n_boot, seed=seed)
                rows.append({"source": key, "best": best, "contestant": n, **diff})
    return rows


def functionalities(examples, scores_by_contestant, thr, split=split_of) -> list[dict]:
    """Share of cases per functionality on the right side of the contestant's operating threshold."""
    hits = defaultdict(list)
    evaluated = [e for e in examples if split(e) == "eval" and e.functionality]
    for name, scores in scores_by_contestant.items():
        for e in evaluated:
            label = functionality_label(e.source, e.functionality)
            threshold, value, expected = thr[name].get(label), scores.get(e.id, {}).get(label), gold(e, label)
            if threshold is not None and value is not None and expected is not None:
                hits[(name, e.source, e.functionality)].append((value > threshold) == expected)
    return [{"contestant": n, "source": s, "functionality": f, "n": len(h), "accuracy": sum(h) / len(h)}
            for (n, s, f), h in sorted(hits.items())]


def build_scoreboard(examples, contestants, raw, n_boot: int = 1000, seed: int = 0, split=split_of) -> dict:
    scores = {c.name: {i: effective_scores(c, r) for i, r in raw.get(c.name, {}).items()} for c in contestants}
    thr = thresholds(examples, scores, split)
    return {
        "coverage": coverage(examples, split),
        "thresholds": thr,
        "slices": slices(examples, scores, thr, n_boot, seed, split),
        "comparisons": comparisons(examples, scores, n_boot, seed, split),
        "functionalities": functionalities(examples, scores, thr, split),
    }
