import json
from types import SimpleNamespace

import pytest
import torch

from app.checkpoints import ActiveCheckpoint
from bench.contestants import FineTunedContestant, checkpoint_factories, effective_scores
from bench.taxonomy import CATEGORIES, FLAG
from trainer.data import text_key

LABELS8 = list(CATEGORIES) + ["sentiment.positive", "sentiment.negative", "sentiment.neutral"]


class FakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def __call__(self, texts, **kwargs):
        self.kwargs = kwargs
        n = len(texts)
        return {"input_ids": torch.zeros((n, 4), dtype=torch.long), "attention_mask": torch.ones((n, 4), dtype=torch.long)}


class FakeHead:
    config = SimpleNamespace(id2label=dict(enumerate(LABELS8)))

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(logits=torch.tensor([[0.0, 3.0, -3.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * input_ids.shape[0]))


def make_checkpoint(models_dir, name, created="2026-10-07T12:00:00Z", trained=("sos un gil",), threshold=0.4):
    path = models_dir / name
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    meta = {"created_at": created, "labels": LABELS8, "threshold": threshold,
            "counts": {"rows": 15297, "spanish": 306}, "trained_texts": [text_key(t) for t in trained]}
    (path / "training_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def contestant(path, **kwargs):
    return FineTunedContestant(path, device="cpu", loader=lambda: (FakeTokenizer(), FakeHead()), **kwargs)


def test_scores_are_the_category_outputs_and_flag_is_their_max(tmp_path):
    c = contestant(make_checkpoint(tmp_path, "ft-1"))

    [raw] = c.predict(["callate"], "es")

    assert raw["threat"] == pytest.approx(torch.sigmoid(torch.tensor(3.0)).item())
    scores = effective_scores(c, raw)
    assert set(scores) == set(CATEGORIES) | {FLAG}
    assert scores[FLAG] == scores["threat"]
    assert c.license == "MIT"


def test_name_is_the_folder_and_version_changes_when_retrained_into_it(tmp_path):
    first = FineTunedContestant(make_checkpoint(tmp_path / "a", "ft-1"))
    again = FineTunedContestant(make_checkpoint(tmp_path / "b", "ft-1", created="2026-10-08T09:00:00Z"))

    assert first.name == again.name == "ft-1"
    assert first.version == "20261007T120000Z"
    assert first.version != again.version


def test_the_contestant_knows_which_texts_its_checkpoint_trained_on(tmp_path):
    c = FineTunedContestant(make_checkpoint(tmp_path, "ft-1", trained=("Sos un GIL", "gracias")))

    assert c.trained_on == frozenset({text_key("sos un gil"), text_key("gracias")})


def test_notes_say_when_and_with_what_it_trained(tmp_path):
    notes = FineTunedContestant(make_checkpoint(tmp_path, "ft-1")).notes

    assert "2026-10-07" in notes and "15297 filas" in notes and "0.400" in notes
    assert "activo" not in notes


def test_unload_releases_the_model_and_the_next_prediction_reloads_it(tmp_path):
    loads = []

    def loader():
        loads.append(1)
        return FakeTokenizer(), FakeHead()

    c = FineTunedContestant(make_checkpoint(tmp_path, "ft-1"), device="cpu", loader=loader)
    c.predict(["a"], "es")
    c.unload()
    c.predict(["b"], "es")

    assert len(loads) == 2


def test_loads_the_checkpoint_folder_as_safetensors(tmp_path, monkeypatch):
    import transformers

    calls = {}

    def recorder(kind):
        def from_pretrained(*args, **kwargs):
            calls[kind] = (args, kwargs)
            return kind
        return from_pretrained

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", recorder("tokenizer"))
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", recorder("model"))
    path = make_checkpoint(tmp_path, "ft-1")

    assert FineTunedContestant(path)._from_disk() == ("tokenizer", "model")
    assert calls["tokenizer"][0] == calls["model"][0] == (path,)
    assert calls["model"][1]["use_safetensors"] is True


def test_factories_list_every_complete_checkpoint_oldest_first(tmp_path):
    make_checkpoint(tmp_path, "ft-20261008-090000")
    make_checkpoint(tmp_path, "ft-20261007-120000")
    make_checkpoint(tmp_path, "ft-cortado", threshold=None)
    (tmp_path / "otra-carpeta").mkdir()

    factories = checkpoint_factories(tmp_path)

    assert list(factories) == ["ft-20261007-120000", "ft-20261008-090000"]
    built = factories["ft-20261007-120000"]("cpu")
    assert built.name == "ft-20261007-120000" and built.device == "cpu"


def test_factories_mark_the_active_checkpoint(tmp_path):
    old = make_checkpoint(tmp_path, "ft-20261007-120000")
    make_checkpoint(tmp_path, "ft-20261008-090000")
    ActiveCheckpoint(tmp_path).activate(old, {})

    factories = checkpoint_factories(tmp_path)

    assert "activo" in factories["ft-20261007-120000"](None).notes
    assert "activo" not in factories["ft-20261008-090000"](None).notes


def test_without_checkpoints_there_are_no_factories(tmp_path):
    assert checkpoint_factories(tmp_path) == {}
