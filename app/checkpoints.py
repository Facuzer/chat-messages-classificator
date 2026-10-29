"""Which fine-tuned checkpoint the app loads."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.jsonfile import read_json, write_json_atomic

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


class ActiveCheckpoint:
    """`<models_dir>/active.json` names the checkpoint in use. Without it the app has no model to serve."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.file = self.models_dir / "active.json"

    def get(self) -> dict | None:
        data = read_json(self.file, None)
        if data is None:
            return None
        path = self.models_dir / data["path"]
        if not path.is_dir():
            return None
        return {**data, "path": path}

    def activate(self, path: Path, metrics: dict) -> None:
        relative = Path(path).resolve().relative_to(self.models_dir.resolve())
        activated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        write_json_atomic(self.file, {"path": relative.as_posix(), "activated_at": activated_at, "metrics": metrics})

    def deactivate(self) -> None:
        self.file.unlink(missing_ok=True)


def checkpoint_problem(path: Path) -> str | None:
    """Why the app cannot serve this folder, in Spanish, or None when it is a complete mmBERT checkpoint.

    The trainer writes `training_meta.json` before evaluating and sets the threshold once evaluation ends,
    so a folder without a threshold is a run that stopped halfway."""
    path = Path(path)
    meta = read_json(path / "training_meta.json", None)
    if not (path / "config.json").exists() or meta is None or "labels" not in meta:
        return "no es un checkpoint de mmBERT de este proyecto"
    if meta.get("threshold") is None:
        return "está incompleto: el entrenamiento se cortó antes de evaluarlo"
    return None
