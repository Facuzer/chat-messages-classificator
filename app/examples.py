"""Labeled examples for fine-tuning: messages marked `insult` or `clean`.

Every example comes from one of three sources: a chat message labeled in the UI (`chat`), a
generated candidate (`generated`) or a row from an imported file (`imported`). Candidates start
`pending` until someone reviews them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from app.jsonfile import read_json, write_json_atomic

LABELS = ("insult", "clean")
SPLITS = ("train", "calib", "test")
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "training" / "examples.json"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def text_key(text: str) -> str:
    """A short id for a text that ignores case and spacing, for comparing texts across datasets."""
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:16]


def _text_hash(text: str) -> int:
    return int(hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest(), 16)


def assign_split(text: str) -> str:
    """70% train, 10% calib, 20% test, decided by the normalized text alone.

    The same text always lands in the same split, so a phrase seen in training can never be
    part of the test set, whichever source it came from.
    """
    bucket = _text_hash(text) % 10
    if bucket < 2:
        return "test"
    if bucket == 2:
        return "calib"
    return "train"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _check_label(label, allowed=LABELS) -> None:
    if label not in allowed:
        raise ValueError(f"unknown label {label!r}; expected one of {', '.join(allowed)}")


class ExampleStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def all(self) -> list[dict]:
        with self._lock:
            return read_json(self.path, [])

    def add_candidates(self, candidates: Iterable[dict], source: str, accept_labels: bool = False) -> dict:
        candidates = list(candidates)
        for c in candidates:
            if c.get("label") is not None:
                _check_label(c["label"])
        with self._lock:
            examples = read_json(self.path, [])
            seen = {normalize_text(e["text"]) for e in examples}
            added = duplicates = 0
            for c in candidates:
                key = normalize_text(c["text"])
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                labeled = accept_labels and c.get("label") is not None
                examples.append(self._new(
                    c["text"], source,
                    status="labeled" if labeled else "pending",
                    label=c["label"] if labeled else None,
                    suggested=c.get("label"),
                    category=c.get("category"),
                ))
                added += 1
            write_json_atomic(self.path, examples)
        return {"added": added, "duplicates": duplicates}

    def label_message(self, message_id: str, text: str, label: str) -> dict:
        _check_label(label)
        with self._lock:
            examples = read_json(self.path, [])
            key = normalize_text(text)
            example = next((e for e in examples if e.get("message_id") == message_id), None) or next(
                (e for e in examples if normalize_text(e["text"]) == key), None
            )
            if example is None:
                example = self._new(text, "chat", status="pending")
                examples.append(example)
            example["message_id"] = message_id
            self._apply_label(example, label)
            write_json_atomic(self.path, examples)
            return example

    def set_label(self, example_id: str, label: str) -> dict:
        """`label` is "insult", "clean" or "discard" (drop the example from training)."""
        _check_label(label, LABELS + ("discard",))
        with self._lock:
            examples = read_json(self.path, [])
            example = next((e for e in examples if e["id"] == example_id), None)
            if example is None:
                raise KeyError(example_id)
            self._apply_label(example, label)
            write_json_atomic(self.path, examples)
            return example

    def next_pending(self) -> dict | None:
        # Ordered by text hash rather than insertion, so a batch imported category by category
        # still reaches the reviewer shuffled.
        pending = [e for e in self.all() if e["status"] == "pending"]
        return min(pending, key=lambda e: _text_hash(e["text"]), default=None)

    def labels_by_message(self) -> dict[str, str]:
        return {e["message_id"]: e["label"] for e in self.all() if e.get("message_id") and e["status"] == "labeled"}

    def labeled(self, split: str) -> list[dict]:
        return [e for e in self.all() if e["status"] == "labeled" and e["split"] == split]

    def summary(self) -> dict:
        examples = self.all()
        labeled = [e for e in examples if e["status"] == "labeled"]
        return {
            "pending": sum(e["status"] == "pending" for e in examples),
            "discarded": sum(e["status"] == "discarded" for e in examples),
            "labeled": {label: sum(e["label"] == label for e in labeled) for label in LABELS},
            "splits": {
                split: {label: sum(e["split"] == split and e["label"] == label for e in labeled) for label in LABELS}
                for split in SPLITS
            },
        }

    @staticmethod
    def _new(text, source, status, label=None, suggested=None, category=None) -> dict:
        return {
            "id": uuid4().hex,
            "text": text,
            "source": source,
            "category": category,
            "status": status,
            "label": label,
            "suggested": suggested,
            "split": assign_split(text),
            "message_id": None,
            "created_at": _utc_now(),
            "labeled_at": _utc_now() if status == "labeled" else None,
        }

    @staticmethod
    def _apply_label(example: dict, label: str) -> None:
        if label == "discard":
            example["status"], example["label"] = "discarded", None
        else:
            example["status"], example["label"] = "labeled", label
        example["labeled_at"] = _utc_now()


def parse_import_lines(lines: Iterable[str], fmt: str) -> list[dict]:
    """Rows of `{"text", "label", "category"}` from JSONL or CSV (header row with a `text` column)."""
    lines = list(lines)
    if fmt == "jsonl":
        rows = [(n, json.loads(line)) for n, line in enumerate(lines, start=1) if line.strip()]
    elif fmt == "csv":
        rows = list(enumerate(csv.DictReader(lines), start=2))
    else:
        raise ValueError(f"unknown format {fmt!r}; expected jsonl or csv")

    parsed = []
    for n, row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            raise ValueError(f"line {n}: missing text")
        parsed.append({
            "text": text,
            "label": (row.get("label") or "").strip() or None,
            "category": (row.get("category") or "").strip() or None,
        })
    return parsed
