"""Metrics for one slice of the scoreboard: gold booleans and one contestant's scores for them.

Each checkpoint's scores sit on their own scale and are not calibrated probabilities, so nothing here
assumes 0.5 means anything. Threshold-free AUROC ranks contestants. A per-contestant
operating threshold, chosen where 5% of the calibration negatives get flagged, shows what one fixed
threshold would do on each source.
"""

from __future__ import annotations

import random

MAX_FPR = 0.05


def _both_classes(gold: list[bool]) -> bool:
    return 0 < sum(gold) < len(gold)


def auroc(gold: list[bool], scores: list[float]) -> float | None:
    if not _both_classes(gold):
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(gold, scores))


def average_precision(gold: list[bool], scores: list[float]) -> float | None:
    if not _both_classes(gold):
        return None
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(gold, scores))


def threshold_at_fpr(gold: list[bool], scores: list[float], max_fpr: float = MAX_FPR) -> float | None:
    """The score above which at most `max_fpr` of the negatives fall. Flag when score > threshold."""
    negatives = sorted(s for g, s in zip(gold, scores) if not g)
    if not negatives:
        return None
    allowed = int(max_fpr * len(negatives) + 1e-9)
    return negatives[len(negatives) - 1 - allowed]


def at_threshold(gold: list[bool], scores: list[float], threshold: float) -> dict:
    flagged = [s > threshold for s in scores]
    tp = sum(f and g for f, g in zip(flagged, gold))
    fp = sum(f and not g for f, g in zip(flagged, gold))
    positives = sum(gold)
    negatives = len(gold) - positives
    return {
        "recall": tp / positives if positives else None,
        "fpr": fp / negatives if negatives else None,
        "precision": tp / (tp + fp) if tp + fp and _both_classes(gold) else None,
    }


def _resample(clusters: list[str], rng: random.Random) -> list[int]:
    """Indices of a bootstrap sample that draws whole clusters with replacement."""
    members: dict[str, list[int]] = {}
    for i, cluster in enumerate(clusters):
        members.setdefault(cluster, []).append(i)
    keys = list(members)
    return [i for _ in keys for i in members[keys[rng.randrange(len(keys))]]]


def _interval(values: list[float], alpha: float = 0.05) -> tuple[float, float] | None:
    if not values:
        return None
    values = sorted(values)
    lo = values[int(alpha / 2 * len(values))]
    hi = values[min(len(values) - 1, int((1 - alpha / 2) * len(values)))]
    return lo, hi


def bootstrap_auroc_ci(gold, scores, clusters, n_boot: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    if auroc(gold, scores) is None:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        idx = _resample(clusters, rng)
        value = auroc([gold[i] for i in idx], [scores[i] for i in idx])
        if value is not None:
            values.append(value)
    return _interval(values)


def paired_auroc_diff(gold, a, b, clusters, n_boot: int = 1000, seed: int = 0) -> dict | None:
    """AUROC(a) - AUROC(b) on the same items. Both are scored on each resample, so item difficulty
    cancels out; comparing two separate intervals would not."""
    base_a, base_b = auroc(gold, a), auroc(gold, b)
    if base_a is None or base_b is None:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _resample(clusters, rng)
        g = [gold[i] for i in idx]
        va, vb = auroc(g, [a[i] for i in idx]), auroc(g, [b[i] for i in idx])
        if va is not None and vb is not None:
            diffs.append(va - vb)
    interval = _interval(diffs)
    return {"diff": base_a - base_b, "lo": interval[0] if interval else None, "hi": interval[1] if interval else None}


def slice_metrics(gold, scores, clusters, threshold: float | None, n_boot: int = 1000, seed: int = 0) -> dict:
    positives = sum(gold)
    n = len(gold)
    op = at_threshold(gold, scores, threshold) if threshold is not None else {"recall": None, "fpr": None, "precision": None}
    return {
        "n": n,
        "positives": positives,
        "prevalence": positives / n if n else None,
        "auroc": auroc(gold, scores),
        "auroc_ci": bootstrap_auroc_ci(gold, scores, clusters, n_boot=n_boot, seed=seed),
        "ap": average_precision(gold, scores),
        "threshold": threshold,
        "recall_at_op": op["recall"],
        "fpr_at_op": op["fpr"],
        "precision_at_op": op["precision"],
    }
