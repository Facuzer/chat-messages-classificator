"""Raw contestant outputs on disk: one JSONL per contestant version and source.

Outputs are stored before any mapping onto the taxonomy, so fixing a mapping needs no new inference.
Each line also stores a hash of the text: when a reviewer edits a suite row, its old output is ignored.
Lines are appended after every batch, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class PredictionCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, contestant, source: str) -> Path:
        folder = re.sub(r"[^A-Za-z0-9._@-]", "_", f"{contestant.name}@{contestant.version}")
        return self.root / folder / f"{source}.jsonl"

    def load(self, contestant, source: str) -> dict[str, tuple[str, dict]]:
        path = self._path(contestant, source)
        if not path.exists():
            return {}
        cached = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a run killed mid-write leaves a partial line
            cached[row["id"]] = (row["text_sha"], row["raw"])
        return cached

    def append(self, contestant, source: str, rows: list[tuple[str, str, dict]]) -> None:
        path = self._path(contestant, source)
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_newline = False
        if path.exists() and path.stat().st_size:
            with path.open("rb") as f:
                f.seek(-1, 2)
                needs_newline = f.read(1) != b"\n"
        # Build the entire batch's text first, so serialization errors fail before anything is written.
        payload = "".join(json.dumps({"id": example_id, "text_sha": sha, "raw": raw}) + "\n" for example_id, sha, raw in rows)
        with path.open("a", encoding="utf-8", newline="\n") as f:
            if needs_newline:
                f.write("\n")
            f.write(payload)
