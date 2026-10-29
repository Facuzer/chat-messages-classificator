"""End-to-end fine-tuning of the real mmBERT-base with a tiny synthetic dataset and a fake sentiment labeler.

Slow (loads mmBERT-base from the Hugging Face cache and trains for one epoch), so it only runs with
`uv run pytest -m slow`.
"""

import json
import math

import pytest

from app.checkpoints import ActiveCheckpoint
from app.examples import ExampleStore
from app.model import LABELS
from bench.taxonomy import Example
from trainer.finetune import NotEnoughData, TrainConfig, run
from trainer.sentiment import SentimentLabels

pytestmark = pytest.mark.slow

NAMES = ["Juan", "Caro", "Pedro", "Lu", "Martín", "Sofi", "Nico", "Vale", "Tomi", "Agus", "Fer", "Majo"]
INSULTS = ["sos un pelotudo", "sos una forra", "callate gil", "sos un inútil de mierda", "sos un tarado",
           "andate a la mierda, imbécil"]
FRIENDLY = ["gracias por la ayuda", "nos vemos mañana", "qué lindo día", "¿venís al asado?",
            "buenísimo, dale", "mandame el archivo"]


def app_rows():
    rows = [{"text": f"{name}, {phrase}", "label": "insult"} for name in NAMES for phrase in INSULTS]
    rows += [{"text": f"{name}, {phrase}", "label": "clean"} for name in NAMES for phrase in FRIENDLY]
    return rows


def suite():
    """Spanish benchmark rows the model never trains on, enough to land in both benchmark splits."""
    examples = []
    for i in range(120):
        insult = i % 2 == 0
        text = f"che jugador {i}, {'sos un boludo de mierda' if insult else 'buena mano, suerte'}"
        examples.append(Example(id=f"suite-es:es-{i:04d}", source="suite-es", lang="es", text=text,
                                labels={"insult": insult, "threat": False, "identity_hate": False}))
    return examples


class FakeLabeler:
    name, version = "fake", "1"

    def predict(self, texts, lang):
        return [{"sentiment.positive": 0.1, "sentiment.negative": 0.8, "sentiment.neutral": 0.1}
                if "mierda" in t or "imbécil" in t else
                {"sentiment.positive": 0.7, "sentiment.negative": 0.1, "sentiment.neutral": 0.2} for t in texts]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("ft")
    store = ExampleStore(root / "examples.json")
    store.add_candidates(app_rows(), source="imported", accept_labels=True)
    models_dir = root / "models"
    result = run(store, models_dir, TrainConfig(epochs=1, batch_size=16), activate=False,
                 log=lambda *_: None, inputs=(store.all(), suite(), [], []), labeler=FakeLabeler(),
                 sentiment_labels=SentimentLabels(root / "sentiment.jsonl"), sentiment_dir=root / "none",
                 cache=root / "cache", cooldown=0)
    return result, models_dir


def test_checkpoint_folder_has_the_model_the_tokenizer_and_the_meta(trained):
    result, _ = trained

    for name in ("config.json", "model.safetensors", "tokenizer.json", "training_meta.json"):
        assert (result["checkpoint"] / name).exists(), name


def test_training_meta_records_outputs_threshold_trained_texts_and_metrics(trained):
    result, _ = trained
    meta = json.loads((result["checkpoint"] / "training_meta.json").read_text(encoding="utf-8"))

    assert meta["labels"] == list(LABELS) and meta["base"]["id"] == "jhu-clsp/mmBERT-base"
    assert isinstance(meta["threshold"], float)
    assert len(meta["trained_texts"]) > 0
    assert all(math.isfinite(loss) for loss in meta["losses"])
    assert "suite-es" in meta["metrics"]["flag_auroc"]
    assert meta["decision"]["reason"]


def test_activate_false_leaves_nothing_active(trained):
    _, models_dir = trained

    assert ActiveCheckpoint(models_dir).get() is None


def test_the_saved_checkpoint_serves_the_chat_on_cpu(trained):
    from app.classifier import load_classifier

    result, models_dir = trained
    ActiveCheckpoint(models_dir).activate(result["checkpoint"], result["metrics"])

    c = load_classifier(ActiveCheckpoint(models_dir), device="cpu").classify("Juan, sos un pelotudo")

    assert set(c.categories) == {"insult", "threat", "identity_hate", "sexual_harassment", "profanity"}
    assert sum(c.sentiment.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert c.flag.category in ("insult", "threat", "identity_hate")


def test_too_few_examples_refuses_before_loading_anything(tmp_path):
    store = ExampleStore(tmp_path / "examples.json")
    store.add_candidates([{"text": "sos un gil", "label": "insult"}], source="imported", accept_labels=True)

    with pytest.raises(NotEnoughData):
        run(store, tmp_path / "models", log=lambda *_: None)


def test_a_retrain_is_compared_with_the_active_checkpoint_and_activated_only_if_better(trained):
    from trainer.metrics import activation_decision

    first, models_dir = trained
    ActiveCheckpoint(models_dir).activate(first["checkpoint"], first["metrics"])
    root = models_dir.parent
    store = ExampleStore(root / "examples.json")

    second = run(store, models_dir, TrainConfig(epochs=1, batch_size=16, seed=1), activate=True,
                 log=lambda *_: None, inputs=(store.all(), suite(), [], []), labeler=FakeLabeler(),
                 sentiment_labels=SentimentLabels(root / "sentiment.jsonl"), sentiment_dir=root / "none",
                 cache=root / "cache", cooldown=0)

    comparison = second["metrics"]["comparison"]
    assert comparison["active_name"] == first["checkpoint"].name
    assert comparison["new"]["auroc"] is not None and comparison["active"]["auroc"] is not None
    expected, reason = activation_decision(comparison["new"], comparison["active"])
    assert second["activated"] is expected and second["reason"] == reason
    assert ActiveCheckpoint(models_dir).get()["path"] == (second["checkpoint"] if expected else first["checkpoint"])
