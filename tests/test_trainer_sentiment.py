import re
from types import SimpleNamespace

import pytest
import torch

from trainer.labeler import (
    LETTERS, QWEN3_4B_REVISION, SENTIMENT_QUESTIONS, SentimentLabeler, fingerprint, prompt_fingerprint, user_prompt,
)
from trainer.sentiment import SentimentLabels, label_missing, sentiment_labeler
from trainer.throttle import pause


class FakeLabeler:
    name, version = "fake", "v1"

    def __init__(self):
        self.batches = []

    def predict(self, texts, lang):
        self.batches.append(list(texts))
        return [{"sentiment.positive": 0.7, "sentiment.negative": 0.1, "sentiment.neutral": 0.2} for _ in texts]


def test_only_texts_without_a_label_are_sent_once_each_in_batches(tmp_path):
    labels = SentimentLabels(tmp_path / "labels.jsonl")
    labeler = FakeLabeler()

    n = label_missing(["a", "b", "A ", "c"], labels, labeler, batch_size=2, log=lambda _: None)

    assert n == 3
    assert labeler.batches == [["a", "b"], ["c"]]


def test_labels_persist_and_a_second_run_labels_nothing(tmp_path):
    path = tmp_path / "labels.jsonl"
    label_missing(["gracias"], SentimentLabels(path), FakeLabeler(), log=lambda _: None)
    labeler = FakeLabeler()

    assert label_missing(["Gracias", "otra"], SentimentLabels(path), labeler, log=lambda _: None) == 1
    assert labeler.batches == [["otra"]]
    assert SentimentLabels(path).get("gracias") == {"positive": 0.7, "negative": 0.1, "neutral": 0.2}
    assert SentimentLabels(path).get("nunca etiquetado") is None


def test_each_label_records_who_labeled_it(tmp_path):
    path = tmp_path / "labels.jsonl"
    label_missing(["gracias"], SentimentLabels(path), FakeLabeler(), log=lambda _: None)

    assert '"labeler": "fake@v1"' in path.read_text(encoding="utf-8")


def test_a_truncated_last_line_is_ignored(tmp_path):
    path = tmp_path / "labels.jsonl"
    label_missing(["gracias"], SentimentLabels(path), FakeLabeler(), log=lambda _: None)
    with path.open("a", encoding="utf-8") as f:
        f.write('{"key": "abc", "pro')

    labels = SentimentLabels(path)
    assert labels.get("gracias") is not None
    label_missing(["otra"], labels, FakeLabeler(), log=lambda _: None)
    assert SentimentLabels(path).get("otra") is not None


def test_the_labeler_is_qwen_asking_only_the_sentiment_question():
    labeler = sentiment_labeler()

    assert labeler.model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert list(labeler.questions) == ["sentiment"]
    assert labeler.name == "qwen3-4b-sentiment"


VOCAB = {letter: i for i, letter in enumerate(LETTERS)}


class FakeChatTokenizer:
    pad_token = None
    eos_token = "<eos>"

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return " | ".join(m["content"] for m in messages)

    def encode(self, text, add_special_tokens):
        return [VOCAB[text]]

    def __call__(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        # The fake model reads from the input id whether the prompt is about an angry message.
        ids = torch.tensor([[1 if "mierda" in p else 0] for p in prompts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class FakeLM:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, logits_to_keep):
        assert logits_to_keep == 1
        logits = torch.zeros((input_ids.shape[0], 1, len(LETTERS)))
        logits[:, 0, 1] = input_ids[:, 0] * 2.0  # the angry message prefers option B, negative
        return SimpleNamespace(logits=logits)


def test_labeler_reads_one_probability_per_sentiment_from_the_option_letters():
    tokenizer = FakeChatTokenizer()
    labeler = SentimentLabeler(device="cpu", micro_batch=1, loader=lambda: (tokenizer, FakeLM()))

    angry, calm = labeler.predict(["qué día de mierda", "gracias"], "es")

    assert len(tokenizer.prompts) == 2
    assert set(angry) == {"sentiment.positive", "sentiment.negative", "sentiment.neutral"}
    assert sum(angry.values()) == pytest.approx(1.0)
    assert angry["sentiment.negative"] > angry["sentiment.positive"]
    assert calm["sentiment.negative"] == pytest.approx(1 / 3)


def test_prompt_lists_the_sentiments_as_letters_and_quotes_the_message():
    prompt = user_prompt("callate gil", SENTIMENT_QUESTIONS["sentiment"])

    assert '"""\ncallate gil\n"""' in prompt
    assert "A) positive:" in prompt and "C) neutral:" in prompt
    assert "`message`" not in prompt


def test_labeler_rejects_a_tokenizer_that_splits_option_letters():
    class Splitting(FakeChatTokenizer):
        def encode(self, text, add_special_tokens):
            return [0, 1]

    with pytest.raises(ValueError, match="splits option letter"):
        SentimentLabeler.letter_ids(Splitting(), 2)


def test_qwen_is_pinned_to_a_full_revision_and_names_it_in_its_version():
    assert re.fullmatch(r"[0-9a-f]{40}", QWEN3_4B_REVISION)
    assert f"-{QWEN3_4B_REVISION[:12]}-" in SentimentLabeler().version


def test_question_and_prompt_wording_match_the_cached_labels():
    # data/training/sentiment_labels.jsonl was drafted with this wording. Editing it is allowed, but new
    # labels would then follow a different criterion than the cached ones, so it should be deliberate.
    assert fingerprint(SENTIMENT_QUESTIONS) == "bf6ea07d"
    assert prompt_fingerprint() == "c3afe33e"


def test_sentiment_question_has_the_three_sentiments():
    assert list(SENTIMENT_QUESTIONS["sentiment"]["criteria"]) == ["positive", "negative", "neutral"]


@pytest.mark.parametrize("duty,expected", [(0.5, 2.0), (0.6, 2.0 * (1 / 0.6 - 1)), (1.0, 0.0)])
def test_pause_sleeps_in_proportion_to_the_busy_time(duty, expected):
    slept = []

    pause(10.0, duty=duty, clock=lambda: 12.0, sleep=slept.append, sync=lambda: None)

    assert sum(slept) == pytest.approx(expected)
