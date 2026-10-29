import random

from bench.metrics import (
    _resample,
    at_threshold,
    auroc,
    average_precision,
    bootstrap_auroc_ci,
    paired_auroc_diff,
    slice_metrics,
    threshold_at_fpr,
)


def test_auroc_is_one_for_perfect_separation_and_none_with_a_single_class():
    assert auroc([True, False, True, False], [0.9, 0.1, 0.8, 0.2]) == 1.0
    assert auroc([True, False], [0.1, 0.9]) == 0.0
    assert auroc([True, True], [0.1, 0.9]) is None
    assert average_precision([False, False], [0.1, 0.2]) is None


def test_threshold_lets_through_at_most_the_allowed_share_of_negatives():
    gold = [False] * 100 + [True] * 10
    scores = [i / 100 for i in range(100)] + [0.99] * 10

    threshold = threshold_at_fpr(gold, scores, max_fpr=0.05)

    assert threshold == 0.94
    assert at_threshold(gold, scores, threshold) == {"recall": 1.0, "fpr": 0.05, "precision": 10 / 15}


def test_with_few_negatives_the_threshold_is_the_highest_negative():
    assert threshold_at_fpr([False, False, True], [0.2, 0.4, 0.9], max_fpr=0.05) == 0.4


def test_threshold_is_none_without_negatives():
    assert threshold_at_fpr([True], [0.5]) is None


def test_only_scores_strictly_above_the_threshold_are_flagged():
    assert at_threshold([True, False], [0.5, 0.5], 0.5) == {"recall": 0.0, "fpr": 0.0, "precision": None}


def test_resample_keeps_clusters_whole():
    clusters = ["a", "a", "b", "c", "c", "c"]

    drawn = _resample(clusters, random.Random(1))

    for cluster in "abc":
        members = [i for i, c in enumerate(clusters) if c == cluster]
        assert len({drawn.count(i) for i in members}) == 1


def test_bootstrap_ci_brackets_the_estimate_and_is_reproducible():
    rng = random.Random(0)
    gold = [i % 2 == 0 for i in range(200)]
    scores = [(0.6 if g else 0.4) + rng.uniform(-0.3, 0.3) for g in gold]
    clusters = [str(i) for i in range(200)]

    lo, hi = bootstrap_auroc_ci(gold, scores, clusters, n_boot=200, seed=3)

    assert lo <= auroc(gold, scores) <= hi
    assert bootstrap_auroc_ci(gold, scores, clusters, n_boot=200, seed=3) == (lo, hi)


def test_paired_difference_of_a_model_with_itself_is_zero():
    rng = random.Random(0)
    gold = [i % 2 == 0 for i in range(200)]
    scores = [(0.6 if g else 0.4) + rng.uniform(-0.3, 0.3) for g in gold]
    clusters = [str(i) for i in range(200)]

    assert paired_auroc_diff(gold, scores, scores, clusters, n_boot=200) == {
        "diff": 0.0, "lo": 0.0, "hi": 0.0,
    }


def test_slice_metrics_reports_none_for_a_single_class_or_no_threshold():
    metrics = slice_metrics([True, True], [0.2, 0.9], ["1", "2"], threshold=None, n_boot=10)

    assert metrics["n"] == 2 and metrics["positives"] == 2 and metrics["prevalence"] == 1.0
    assert metrics["auroc"] is None and metrics["auroc_ci"] is None and metrics["ap"] is None
    assert metrics["recall_at_op"] is None and metrics["fpr_at_op"] is None


def test_precision_at_op_is_none_for_all_positive_slice():
    metrics = slice_metrics([True, True, True], [0.9, 0.2, 0.7], ["1", "2", "3"], threshold=0.5, n_boot=10)

    assert metrics["precision_at_op"] is None
    assert metrics["recall_at_op"] == 2 / 3
    assert metrics["fpr_at_op"] is None


def test_precision_at_op_is_none_for_all_negative_slice():
    metrics = slice_metrics([False, False], [0.9, 0.2], ["1", "2"], threshold=0.5, n_boot=10)

    assert metrics["precision_at_op"] is None
    assert metrics["fpr_at_op"] == 0.5
    assert metrics["recall_at_op"] is None


def test_precision_at_op_is_defined_for_two_class_slice():
    metrics = slice_metrics([True, False, True, False], [0.9, 0.6, 0.4, 0.1], ["1", "2", "3", "4"], threshold=0.5, n_boot=10)

    assert metrics["recall_at_op"] == 0.5
    assert metrics["fpr_at_op"] == 0.5
    assert metrics["precision_at_op"] == 0.5
    assert metrics["auroc"] == 0.75
    assert metrics["threshold"] == 0.5
