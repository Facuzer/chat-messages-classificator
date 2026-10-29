"""The project's fine-tuned mmBERT-base checkpoints (scripts/finetune.py), one per folder in models/.

One forward pass gives the five categories and the sentiment; the benchmark scores the categories. Most
of the app's Spanish training examples are also suite rows, so each checkpoint lists the texts it trained
on and `bench.py run` leaves them out for every checkpoint it compares.
"""

from __future__ import annotations

from pathlib import Path

from app.checkpoints import MODELS_DIR, ActiveCheckpoint, checkpoint_problem
from app.jsonfile import read_json
from bench.contestants.base import free_gpu
from bench.taxonomy import CATEGORIES, LANGS


def _notes(meta: dict, active: bool) -> str:
    counts = meta.get("counts", {})
    threshold = meta.get("threshold")
    notes = f"mmBERT-base ajustado el {meta['created_at'][:10]}"
    if counts:
        notes += f" con {counts['rows']} filas ({counts['spanish']} ejemplos en español)"
    if threshold is not None:
        notes += f"; umbral del entrenamiento {threshold:.3f}"
    return notes + (". Es el checkpoint activo de la app." if active else ".")


class FineTunedContestant:
    languages = frozenset(LANGS)
    license = "MIT"

    def __init__(self, checkpoint: Path, device: str | None = None, loader=None, active: bool = False):
        self.checkpoint = Path(checkpoint)
        self.name = self.checkpoint.name
        meta = read_json(self.checkpoint / "training_meta.json", {})
        # When it was trained: a retrain into the same folder must not reuse cached predictions.
        self.version = meta["created_at"].replace(":", "").replace("-", "")
        self.trained_on = frozenset(meta.get("trained_texts", ()))
        self.notes = _notes(meta, active)
        self.device = device
        self._loader = loader or self._from_disk
        self._loaded = None

    def _from_disk(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        return (AutoTokenizer.from_pretrained(self.checkpoint),
                AutoModelForSequenceClassification.from_pretrained(self.checkpoint, use_safetensors=True))

    def _get(self):
        if self._loaded is None:
            import torch

            tokenizer, model = self._loader()
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._loaded = (tokenizer, model.to(device).eval(), device)
        return self._loaded

    def predict(self, texts: list[str], lang: str) -> list[dict[str, float]]:
        from app.model import predict_raw

        tokenizer, model, device = self._get()
        return predict_raw(tokenizer, model, device, texts)

    def to_scores(self, raw: dict[str, float]) -> dict[str, float]:
        return {c: raw[c] for c in CATEGORIES if c in raw}

    def unload(self) -> None:
        self._loaded = None
        free_gpu()


def checkpoint_factories(models_dir: Path = MODELS_DIR) -> dict:
    """{folder name: factory(device)} for every complete checkpoint in `models_dir`, oldest first."""
    models_dir = Path(models_dir)
    active = ActiveCheckpoint(models_dir).get()
    active_path = active["path"] if active else None
    paths = [m.parent for m in sorted(models_dir.glob("*/training_meta.json")) if checkpoint_problem(m.parent) is None]
    return {p.name: (lambda device, p=p: FineTunedContestant(p, device=device, active=p == active_path))
            for p in paths}
