from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.classifier import Classification, Flag, Verdict
from app.examples import ExampleStore
from app.main import create_app
from app.store import MessageStore


INSULT = {"insult": 0.91, "threat": 0.02, "identity_hate": 0.01, "sexual_harassment": 0.0, "profanity": 0.3}
CLEAN = {"insult": 0.03, "threat": 0.01, "identity_hate": 0.01, "sexual_harassment": 0.0, "profanity": 0.0}


class FakeClassifier:
    """Stands in for the model: any text containing "idiota" is an insult, everything else is positive."""

    def __init__(self):
        self.texts = []

    def classify(self, text: str) -> Classification:
        self.texts.append(text)
        if "idiota" in text:
            return Classification(
                flagged=True,
                flag=Flag(score=0.91, category="insult", threshold=0.4),
                categories=INSULT,
                sentiment=Verdict("negative", 0.8, {"positive": 0.05, "negative": 0.8, "neutral": 0.15}),
                model="ft-1",
                latency_ms=12,
            )
        return Classification(
            flagged=False,
            flag=Flag(score=0.03, category="insult", threshold=0.4),
            categories=CLEAN,
            sentiment=Verdict("positive", 0.9, {"positive": 0.9, "negative": 0.02, "neutral": 0.08}),
            model="ft-1",
            latency_ms=9,
        )


@pytest.fixture
def classifier():
    return FakeClassifier()


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "messages.json"


@pytest.fixture
def examples(tmp_path):
    return ExampleStore(tmp_path / "training" / "examples.json")


@pytest.fixture
def client(classifier, store_path, examples):
    app = create_app(classifier=classifier, store=MessageStore(store_path), examples=examples)
    with TestClient(app) as c:
        yield c


def test_post_returns_the_classified_message(client, classifier):
    response = client.post("/api/messages", json={"text": "  sos un idiota  "})

    assert response.status_code == 201
    body = response.json()
    assert classifier.texts == ["sos un idiota"]
    assert body["text"] == "sos un idiota"
    assert body["flagged"] is True
    assert body["flag"] == {"score": 0.91, "category": "insult", "threshold": 0.4}
    assert body["categories"] == INSULT
    assert body["sentiment"] == {
        "label": "negative",
        "confidence": 0.8,
        "probabilities": {"positive": 0.05, "negative": 0.8, "neutral": 0.15},
    }
    assert body["model"] == "ft-1"
    assert body["latency_ms"] == 12
    assert isinstance(body["id"], str) and body["id"]
    assert body["created_at"].endswith("Z")
    datetime.fromisoformat(body["created_at"])


def test_each_message_gets_its_own_id(client):
    first = client.post("/api/messages", json={"text": "hola"}).json()
    second = client.post("/api/messages", json={"text": "hola"}).json()

    assert first["id"] != second["id"]


def test_posted_messages_are_listed_in_order(client):
    first = client.post("/api/messages", json={"text": "hola"}).json()
    second = client.post("/api/messages", json={"text": "sos un idiota"}).json()

    listed = client.get("/api/messages").json()
    assert [m["id"] for m in listed] == [first["id"], second["id"]]
    assert listed[1]["flagged"] is True


def test_posted_message_is_persisted_to_the_json_file(client, store_path):
    posted = client.post("/api/messages", json={"text": "hola"}).json()

    assert MessageStore(store_path).list() == [posted]


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_text_is_rejected_without_classifying(client, classifier, text):
    response = client.post("/api/messages", json={"text": text})

    assert response.status_code == 422
    assert classifier.texts == []
    assert client.get("/api/messages").json() == []


def test_text_up_to_1000_characters_is_accepted(client):
    assert client.post("/api/messages", json={"text": "a" * 1000}).status_code == 201


def test_text_over_1000_characters_is_rejected(client, classifier):
    assert client.post("/api/messages", json={"text": "a" * 1001}).status_code == 422
    assert classifier.texts == []


def test_stats_reflect_posted_messages(client):
    client.post("/api/messages", json={"text": "gracias, genial"})
    client.post("/api/messages", json={"text": "sos un idiota"})
    client.post("/api/messages", json={"text": "otro idiota"})

    assert client.get("/api/stats").json() == {"total": 3, "positive": 1, "negative": 2, "neutral": 0, "flagged": 2}


def test_root_serves_the_chat_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


# --- labeling chat messages -------------------------------------------------------------------


def test_labeling_a_message_shows_up_in_the_message_list(client, examples):
    message = client.post("/api/messages", json={"text": "boludo vení"}).json()

    response = client.post(f"/api/messages/{message['id']}/label", json={"label": "clean"})

    assert response.status_code == 200
    assert response.json()["label"] == "clean"
    [listed] = client.get("/api/messages").json()
    assert listed["label"] == "clean"
    [example] = examples.all()
    assert (example["source"], example["message_id"], example["text"]) == ("chat", message["id"], "boludo vení")


def test_unlabeled_messages_list_with_no_label(client):
    client.post("/api/messages", json={"text": "hola"})

    assert client.get("/api/messages").json()[0]["label"] is None


def test_labeling_an_unknown_message_is_404(client, examples):
    assert client.post("/api/messages/nope/label", json={"label": "insult"}).status_code == 404
    assert examples.all() == []


@pytest.mark.parametrize("label", ["discard", "maybe"])
def test_chat_labels_are_only_insult_or_clean(client, label):
    message = client.post("/api/messages", json={"text": "hola"}).json()

    assert client.post(f"/api/messages/{message['id']}/label", json={"label": label}).status_code == 422


# --- review queue -------------------------------------------------------------------------------


def test_next_example_comes_with_the_models_prediction(client, classifier, examples):
    examples.add_candidates([{"text": "sos un idiota", "label": "insult"}], source="generated")

    body = client.get("/api/training/next").json()

    assert body["example"]["text"] == "sos un idiota"
    assert body["example"]["suggested"] == "insult"
    assert body["prediction"] == {"p_flag": 0.91, "category": "insult", "flagged": True, "model": "ft-1"}
    assert classifier.texts == ["sos un idiota"]


def test_next_example_is_null_when_nothing_is_pending(client, classifier):
    assert client.get("/api/training/next").json() == {"example": None, "prediction": None}
    assert classifier.texts == []


def test_reviewing_an_example_labels_it(client, examples):
    examples.add_candidates([{"text": "te voy a romper la cara"}], source="generated")
    example_id = examples.all()[0]["id"]

    response = client.post(f"/api/training/examples/{example_id}/label", json={"label": "insult"})

    assert response.status_code == 200
    assert examples.all()[0]["label"] == "insult"
    assert client.get("/api/training/next").json()["example"] is None


def test_reviewing_can_discard(client, examples):
    examples.add_candidates([{"text": "asdf"}], source="generated")
    example_id = examples.all()[0]["id"]

    client.post(f"/api/training/examples/{example_id}/label", json={"label": "discard"})

    assert examples.all()[0]["status"] == "discarded"


def test_reviewing_an_unknown_example_is_404(client):
    assert client.post("/api/training/examples/nope/label", json={"label": "clean"}).status_code == 404


def test_reviewing_with_an_unknown_label_is_422(client, examples):
    examples.add_candidates([{"text": "hola"}], source="generated")
    example_id = examples.all()[0]["id"]

    assert client.post(f"/api/training/examples/{example_id}/label", json={"label": "maybe"}).status_code == 422


def test_training_summary_reports_progress(client, examples):
    examples.add_candidates([{"text": "a", "label": "insult"}], source="imported", accept_labels=True)
    examples.add_candidates([{"text": "b"}], source="generated")

    summary = client.get("/api/training/summary").json()

    assert summary["labeled"] == {"insult": 1, "clean": 0}
    assert summary["pending"] == 1
    assert summary["target"] > 0
    assert summary["active_model"] is None


def test_training_summary_reports_the_fine_tuned_model_in_use(tmp_path, store_path, examples):
    classifier = FakeClassifier()
    classifier.checkpoint = {
        "path": tmp_path / "ft-20260930-2310",
        "activated_at": "2026-09-30T23:10:00Z",
        "metrics": {"base": {"f1": 0.5}, "tuned": {"f1": 0.8}},
    }
    app = create_app(classifier=classifier, store=MessageStore(store_path), examples=examples)

    with TestClient(app) as c:
        active = c.get("/api/training/summary").json()["active_model"]

    assert active == {
        "name": "ft-20260930-2310",
        "activated_at": "2026-09-30T23:10:00Z",
        "metrics": {"base": {"f1": 0.5}, "tuned": {"f1": 0.8}},
    }


def test_training_page_is_served(client):
    response = client.get("/training.html")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_metrics_endpoint_tolerates_a_classifier_without_a_router(client):
    response = client.get("/api/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["models"] == []
    assert body["performance"] is None
    assert {"uptime_s", "gpu_memory", "gpu", "host"} <= body.keys()


def test_pages_include_the_metrics_panel(client):
    assert 'id="metrics-body"' in client.get("/training.html").text
    assert 'id="metrics-body"' in client.get("/").text
