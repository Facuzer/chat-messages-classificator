import json

import pytest
import torch

from bench.taxonomy import Example
from trainer.evaluation import (
    MIN_CALIB_NEGATIVES, choose_threshold, flag_auroc_by_source, load_sentiment_sets, operating_point, sentiment_f1,
)
from trainer.finetune import loss_fn, targets
from trainer.metrics import MIN_SPANISH_AUROC, activation_decision, macro_f1


def ex(i, flag, source="suite-es", split="eval", text=None):
    labels = {"insult": flag, "threat": False, "identity_hate": False} if flag is not None else {"insult": None}
    return Example(id=f"{source}:{split}{i}", source=source, lang="es", text=text or f"{source} {split} {i}", labels=labels)


def split(example):
    return "calib" if ":calib" in example.id else "eval"


# --- threshold ------------------------------------------------------------------------------


def calib_set(n_spanish_negatives):
    examples, scores = [], {}
    for i in range(n_spanish_negatives):
        e = ex(i, False, split="calib")
        examples.append(e)
        scores[e.id] = i / 100
    for i in range(40):
        e = ex(i, False, source="toldbr", split="calib")
        examples.append(e)
        scores[e.id] = 0.9
    positive = ex(999, True, split="calib")
    examples.append(positive)
    scores[positive.id] = 0.95
    return examples, scores


def test_the_threshold_flags_at_most_five_percent_of_spanish_calibration_negatives():
    examples, scores = calib_set(40)  # Spanish negatives score 0.00 … 0.39

    threshold, info = choose_threshold(examples, scores, split=split)

    assert threshold == pytest.approx(0.37)  # 2 of 40 negatives above it
    assert info == {"source": "suite-es", "negatives": 40}


def test_too_few_spanish_negatives_pools_every_source():
    examples, scores = calib_set(MIN_CALIB_NEGATIVES - 1)

    threshold, info = choose_threshold(examples, scores, split=split)

    assert info["source"] == "todas" and info["negatives"] == MIN_CALIB_NEGATIVES - 1 + 40
    assert threshold == pytest.approx(0.9)


def test_eval_rows_never_set_the_threshold():
    examples, scores = calib_set(40)
    leak = ex(5, False)  # an eval row scoring above everything
    examples.append(leak)
    scores[leak.id] = 0.99

    assert choose_threshold(examples, scores, split=split)[0] == pytest.approx(0.37)


# --- flag metrics ---------------------------------------------------------------------------


def test_auroc_is_per_source_on_the_eval_split_only():
    examples = [ex(1, True), ex(2, False), ex(3, True, source="toldbr"), ex(4, False, source="toldbr"),
                ex(5, False, split="calib"), ex(6, None)]
    scores = {examples[0].id: 0.9, examples[1].id: 0.1, examples[2].id: 0.2, examples[3].id: 0.8,
              examples[4].id: 0.99, examples[5].id: 0.5}

    assert flag_auroc_by_source(examples, scores, split=split) == {"suite-es": 1.0, "toldbr": 0.0}


def test_operating_point_reports_recall_and_false_positive_rate_on_one_source():
    examples = [ex(1, True), ex(2, True), ex(3, False), ex(4, False), ex(5, True, source="toldbr")]
    scores = {examples[0].id: 0.9, examples[1].id: 0.3, examples[2].id: 0.6, examples[3].id: 0.1, examples[4].id: 0.0}

    assert operating_point(examples, scores, 0.5, source="suite-es", split=split) == {"recall": 0.5, "fpr": 0.5}


# --- sentiment ------------------------------------------------------------------------------


def test_macro_f1_averages_the_three_sentiments():
    assert macro_f1(["positive", "negative", "neutral"], ["positive", "negative", "neutral"]) == 1.0
    assert macro_f1(["positive", "positive"], ["negative", "negative"]) == 0.0


def test_sentiment_sets_read_dynasent_and_the_suite_draft(tmp_path):
    rows = [{"text_id": "a", "sentence": "Great food.", "gold_label": "positive"},
            {"text_id": "b", "sentence": "Mixed.", "gold_label": "mixed"}]
    (tmp_path / "dynasent-v1.1-round02-dynabench-test.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (tmp_path / "suite_sentiment_draft.csv").write_text(
        "﻿id;sentiment;note\nes-0001;negative;\nes-0002;?;dudoso\n", encoding="utf-8")
    suite = [Example(id="suite-es:es-0001", source="suite-es", lang="es", text="callate gil", labels={}),
             Example(id="suite-es:es-0002", source="suite-es", lang="es", text="bue", labels={})]

    sets = load_sentiment_sets(tmp_path, suite)

    assert sets == {"dynasent-r2-en": [("Great food.", "positive")], "suite-es-draft": [("callate gil", "negative")]}


def test_sentiment_sets_are_empty_without_the_files(tmp_path):
    assert load_sentiment_sets(tmp_path, []) == {}


def test_sentiment_f1_reads_the_most_likely_sentiment():
    def predict(texts):
        return [{"sentiment.positive": 0.8 if "bien" in t else 0.1, "sentiment.negative": 0.15,
                 "sentiment.neutral": 0.05} for t in texts]

    sets = {"es": [("todo bien", "positive"), ("todo mal", "negative")]}

    assert sentiment_f1(predict, sets) == {"es": pytest.approx(macro_f1(["positive", "negative"], ["positive", "negative"]))}


# --- activation rule ------------------------------------------------------------------------


def test_the_first_checkpoint_activates_when_it_separates_spanish_well():
    ok, reason = activation_decision({"auroc": 0.9, "sentiment_f1": 0.6}, None)

    assert ok is True and reason


def test_the_first_checkpoint_does_not_activate_below_the_floor():
    ok, _ = activation_decision({"auroc": MIN_SPANISH_AUROC - 0.01, "sentiment_f1": 0.6}, None)

    assert ok is False


def test_a_new_checkpoint_must_beat_the_active_one():
    assert activation_decision({"auroc": 0.91, "sentiment_f1": 0.6}, {"auroc": 0.90, "sentiment_f1": 0.6})[0] is True
    assert activation_decision({"auroc": 0.90, "sentiment_f1": 0.6}, {"auroc": 0.90, "sentiment_f1": 0.6})[0] is False


def test_a_new_checkpoint_that_loses_sentiment_does_not_activate():
    ok, reason = activation_decision({"auroc": 0.95, "sentiment_f1": 0.50}, {"auroc": 0.90, "sentiment_f1": 0.60})

    assert ok is False and "sentimiento" in reason


def test_without_a_spanish_auroc_nothing_activates():
    assert activation_decision({"auroc": None, "sentiment_f1": None}, None)[0] is False


# --- training targets and loss --------------------------------------------------------------


def row(**labels):
    return {"text": "x", "labels": {c: labels.get(c) for c in ("insult", "threat", "identity_hate", "sexual_harassment", "profanity")}}


def test_targets_mask_unknown_categories_and_missing_sentiment():
    rows = [row(insult=True, threat=False), row(profanity=True)]
    sentiment = [{"positive": 0.2, "negative": 0.6, "neutral": 0.2}, None]

    cats, cat_mask, sent, sent_mask = targets(rows, sentiment)

    assert cats.tolist() == [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]]
    assert cat_mask.tolist() == [[1, 1, 0, 0, 0], [0, 0, 0, 0, 1]]
    assert sent[0].tolist() == pytest.approx([0.2, 0.6, 0.2])
    assert sent[1].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert sent_mask.tolist() == [1, 0]


def test_unknown_labels_do_not_move_the_loss():
    rows = [row(insult=True), row(insult=False)]
    cats, cat_mask, sent, sent_mask = targets(rows, [None, None])
    logits = torch.zeros((2, 8))
    changed = logits.clone()
    changed[:, 1:5] = 7.0  # every category but insult is unknown
    changed[:, 5:] = torch.tensor([3.0, -1.0, 0.0])  # sentiment is unknown too

    assert loss_fn(changed, cats, cat_mask, sent, sent_mask).item() == pytest.approx(
        loss_fn(logits, cats, cat_mask, sent, sent_mask).item())


def test_sentiment_loss_is_soft_cross_entropy_against_the_labels():
    rows = [row()]
    cats, cat_mask, sent, sent_mask = targets(rows, [{"positive": 1.0, "negative": 0.0, "neutral": 0.0}])
    logits = torch.zeros((1, 8))

    assert loss_fn(logits, cats, cat_mask, sent, sent_mask).item() == pytest.approx(torch.log(torch.tensor(3.0)).item())


def test_the_seed_also_fixes_the_initial_weights_of_the_new_outputs(monkeypatch):
    import transformers

    from trainer.finetune import TrainConfig, _load_base

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: "tokenizer")
    # A fresh layer draws its initial weights from torch's global generator, like the new classifier head.
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained",
                        lambda *a, **k: torch.nn.Linear(4, 8))

    torch.manual_seed(123)
    _, first = _load_base(TrainConfig(seed=0))
    torch.manual_seed(456)
    _, second = _load_base(TrainConfig(seed=0))

    assert torch.equal(first.weight, second.weight)
