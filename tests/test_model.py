import math
from types import SimpleNamespace

import pytest
import torch

from app.model import LABELS, MAX_LENGTH, output_names, predict_raw, raw_outputs
from bench.taxonomy import CATEGORIES


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def test_labels_are_the_categories_then_the_three_sentiments():
    assert LABELS == CATEGORIES + ("sentiment.positive", "sentiment.negative", "sentiment.neutral")


def test_categories_are_independent_sigmoids_and_sentiment_is_one_softmax():
    raw = raw_outputs([0.0, 2.0, -2.0, 0.0, 1.0, 1.0, 1.0, 1.0], list(LABELS))

    assert raw["insult"] == pytest.approx(0.5)
    assert raw["threat"] == pytest.approx(sigmoid(2.0))
    assert raw["identity_hate"] == pytest.approx(sigmoid(-2.0))
    assert [raw[f"sentiment.{s}"] for s in ("positive", "negative", "neutral")] == pytest.approx([1 / 3] * 3)


def config(id2label):
    return SimpleNamespace(id2label=dict(enumerate(id2label)))


def test_output_names_come_from_the_config():
    assert output_names(config(LABELS)) == list(LABELS)


@pytest.mark.parametrize("names", [[f"LABEL_{i}" for i in range(8)], ["toxicity", "insult"], list(CATEGORIES),
                                   list(LABELS[:-1])])
def test_other_heads_are_not_checkpoints_of_this_model(names):
    with pytest.raises(ValueError, match="no es un checkpoint"):
        output_names(config(names))


class FakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def __call__(self, texts, **kwargs):
        self.kwargs = kwargs
        n = len(texts)
        return {"input_ids": torch.zeros((n, 3), dtype=torch.long), "attention_mask": torch.ones((n, 3), dtype=torch.long)}


class FakeModel:
    config = config(LABELS)

    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(logits=torch.tensor([[2.0, 0, 0, 0, 0, 0, 3.0, 0]] * input_ids.shape[0]))


def test_predict_raw_runs_one_batch_and_names_every_output():
    tokenizer = FakeTokenizer()

    raw = predict_raw(tokenizer, FakeModel(), "cpu", ["sos un gil", "gracias"])

    assert len(raw) == 2
    assert raw[0]["insult"] == pytest.approx(sigmoid(2.0))
    assert max(("positive", "negative", "neutral"), key=lambda s: raw[0][f"sentiment.{s}"]) == "negative"
    assert tokenizer.kwargs["truncation"] is True and tokenizer.kwargs["max_length"] == MAX_LENGTH
