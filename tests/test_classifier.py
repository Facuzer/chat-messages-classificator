import json

import pytest

from app.checkpoints import ActiveCheckpoint
from app.classifier import Classification, Classifier, Flag, Verdict, load_classifier, threshold_of, to_classification


def raw(insult=0.1, threat=0.05, identity_hate=0.02, sexual_harassment=0.01, profanity=0.3,
        sentiment=(0.1, 0.8, 0.1)):
    positive, negative, neutral = sentiment
    return {"insult": insult, "threat": threat, "identity_hate": identity_hate, "sexual_harassment": sexual_harassment,
            "profanity": profanity, "sentiment.positive": positive, "sentiment.negative": negative,
            "sentiment.neutral": neutral}


def test_flag_score_is_the_highest_of_insult_threat_and_identity_hate():
    c = to_classification(raw(insult=0.3, threat=0.9), threshold=0.5, model="ft-1", latency_ms=20)

    assert c.flag == Flag(score=0.9, category="threat", threshold=0.5)
    assert c.flagged is True


def test_profanity_and_sexual_harassment_never_set_the_flag():
    c = to_classification(raw(insult=0.1, profanity=0.99, sexual_harassment=0.99), threshold=0.5, model="ft-1", latency_ms=1)

    assert c.flagged is False
    assert c.flag.category == "insult"


def test_a_score_equal_to_the_threshold_is_not_flagged():
    # The threshold is the score of a calibration negative: flagging starts above it.
    assert to_classification(raw(insult=0.5), threshold=0.5, model="m", latency_ms=1).flagged is False


def test_classification_carries_categories_sentiment_model_and_latency():
    c = to_classification(raw(), threshold=0.5, model="ft-1", latency_ms=42)

    assert c == Classification(
        flagged=False,
        flag=Flag(score=0.1, category="insult", threshold=0.5),
        categories={"insult": 0.1, "threat": 0.05, "identity_hate": 0.02, "sexual_harassment": 0.01, "profanity": 0.3},
        sentiment=Verdict(label="negative", confidence=0.8, probabilities={"positive": 0.1, "negative": 0.8, "neutral": 0.1}),
        model="ft-1",
        latency_ms=42,
    )


class RecordingPredict:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, texts):
        self.calls.append(texts)
        return [self.result for _ in texts]


def test_classify_predicts_one_text_and_applies_the_threshold():
    predict = RecordingPredict(raw(insult=0.8))

    assert Classifier(predict, threshold=0.9, model="ft-1").classify("hola").flagged is False
    assert Classifier(predict, threshold=0.75, model="ft-1").classify("hola").flagged is True
    assert predict.calls == [["hola"], ["hola"]]


def test_classify_reports_the_model_and_non_negative_latency():
    c = Classifier(RecordingPredict(raw()), threshold=0.5, model="ft-20261007-1200").classify("hola")

    assert c.model == "ft-20261007-1200"
    assert c.latency_ms >= 0


def checkpoint(models_dir, name="ft-1", meta=None):
    path = models_dir / name
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "training_meta.json").write_text(json.dumps(meta or {"labels": ["insult"], "threshold": 0.37}), encoding="utf-8")
    return path


def test_the_threshold_comes_from_the_checkpoint_unless_flag_threshold_is_set(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    monkeypatch.delenv("FLAG_THRESHOLD", raising=False)
    assert threshold_of(path) == 0.37

    monkeypatch.setenv("FLAG_THRESHOLD", "0.8")
    assert threshold_of(path) == 0.8


def test_without_an_active_checkpoint_the_server_explains_how_to_train_one(tmp_path):
    with pytest.raises(RuntimeError, match="finetune.py"):
        load_classifier(ActiveCheckpoint(tmp_path))


def test_an_active_checkpoint_of_another_model_is_refused_before_loading(tmp_path):
    other = tmp_path / "ft-otro"
    other.mkdir()
    (other / "weights.bin").write_bytes(b"x")
    ActiveCheckpoint(tmp_path).activate(other, {})

    with pytest.raises(RuntimeError, match="ft-otro"):
        load_classifier(ActiveCheckpoint(tmp_path), loader=lambda *a: pytest.fail("must not load"))


def test_load_classifier_loads_the_active_checkpoint_with_its_threshold(tmp_path, monkeypatch):
    monkeypatch.delenv("FLAG_THRESHOLD", raising=False)
    path = checkpoint(tmp_path, "ft-2")
    ActiveCheckpoint(tmp_path).activate(path, {"x": 1})
    loaded = []

    def loader(p, device):
        loaded.append((p, device))
        return "tok", "model", "cpu"

    classifier = load_classifier(ActiveCheckpoint(tmp_path), loader=loader, device="cpu")

    assert loaded == [(path, "cpu")]
    assert classifier.threshold == 0.37
    assert classifier.model == "ft-2"
    assert classifier.checkpoint["metrics"] == {"x": 1}


def test_an_active_checkpoint_whose_evaluation_never_finished_is_refused(tmp_path):
    path = checkpoint(tmp_path, "ft-cortado", meta={"labels": ["insult"], "threshold": None})
    ActiveCheckpoint(tmp_path).activate(path, {})

    with pytest.raises(RuntimeError, match="incompleto"):
        load_classifier(ActiveCheckpoint(tmp_path), loader=lambda *a: pytest.fail("must not load"))
