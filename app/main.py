from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints

from app.classifier import load_classifier
from app.examples import DEFAULT_PATH as DEFAULT_EXAMPLES_PATH
from app.examples import ExampleStore
from app.stats import compute_stats
from app.store import MessageStore

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
DEFAULT_STORE_PATH = ROOT / "data" / "messages.json"

# Labeled examples to aim for; the training page shows progress toward it.
TRAINING_TARGET = 800


class MessageIn(BaseModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class ChatLabelIn(BaseModel):
    label: Literal["insult", "clean"]


class ReviewIn(BaseModel):
    label: Literal["insult", "clean", "discard"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _active_model(classifier) -> dict | None:
    checkpoint = getattr(classifier, "checkpoint", None)
    if checkpoint is None:
        return None
    return {
        "name": Path(checkpoint["path"]).name,
        "activated_at": checkpoint["activated_at"],
        "metrics": checkpoint["metrics"],
    }


def create_app(classifier=None, store: MessageStore | None = None, examples: ExampleStore | None = None) -> FastAPI:
    """Without a `classifier`, the active checkpoint loads at startup, so the server only accepts
    requests once the model is ready."""
    store = store or MessageStore(DEFAULT_STORE_PATH)
    examples = examples or ExampleStore(DEFAULT_EXAMPLES_PATH)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = classifier or load_classifier()
        yield

    app = FastAPI(title="Message Flagger", lifespan=lifespan)

    @app.get("/api/messages")
    def list_messages() -> list[dict]:
        labels = examples.labels_by_message()
        return [{**m, "label": labels.get(m["id"])} for m in store.list()]

    # Sync on purpose: FastAPI runs it in a worker thread, so inference never blocks the event loop.
    @app.post("/api/messages", status_code=201)
    def post_message(body: MessageIn, request: Request) -> dict:
        classification = request.app.state.classifier.classify(body.text)
        message = {"id": uuid4().hex, "text": body.text, "created_at": _utc_now(), **asdict(classification)}
        store.add(message)
        return message

    @app.post("/api/messages/{message_id}/label")
    def label_message(message_id: str, body: ChatLabelIn) -> dict:
        message = next((m for m in store.list() if m["id"] == message_id), None)
        if message is None:
            raise HTTPException(status_code=404, detail="message not found")
        return examples.label_message(message_id, message["text"], body.label)

    @app.get("/api/stats")
    def stats() -> dict[str, int]:
        return compute_stats(store.list())

    @app.get("/api/training/summary")
    def training_summary(request: Request) -> dict:
        return {**examples.summary(), "target": TRAINING_TARGET, "active_model": _active_model(request.app.state.classifier)}

    @app.get("/api/training/next")
    def next_example(request: Request) -> dict:
        example = examples.next_pending()
        if example is None:
            return {"example": None, "prediction": None}
        c = request.app.state.classifier.classify(example["text"])
        prediction = {"p_flag": c.flag.score, "category": c.flag.category, "flagged": c.flagged, "model": c.model}
        return {"example": example, "prediction": prediction}

    @app.post("/api/training/examples/{example_id}/label")
    def review_example(example_id: str, body: ReviewIn) -> dict:
        try:
            return examples.set_label(example_id, body.label)
        except KeyError:
            raise HTTPException(status_code=404, detail="example not found") from None

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
