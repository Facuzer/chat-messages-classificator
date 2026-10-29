"""Builds the first draft of suites/es_casino.csv: the 421 generated Rioplatense examples plus new
casino drafts.

A generated example only gets the labels its category implies; every other cell stays empty for the
reviewers. The suggestions come from a generator, so they count only after a person reviews them.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.examples import normalize_text
from bench.sources.suite import COLUMNS, write_suite
from bench.taxonomy import CATEGORIES, TARGETS

CLEAN = {"insult": "0", "threat": "0", "identity_hate": "0"}
# category in datasets/generated_rioplatense_v1.jsonl → (functionality, implied labels, target)
SEED_MAP = {
    "insulto directo": ("insult_direct", {"insult": "1"}, "player"),
    "amenaza": ("threat", {"threat": "1"}, "player"),
    # Several of these are class, body or age insults ("villero", "gorda", "viejo"): reviewers move them
    # to insult. Target is left for them too, since many are aimed at the person.
    "discriminación": ("identity_hate", {"identity_hate": "1"}, ""),
    "puteada sin destinatario": ("profanity_untargeted", {**CLEAN, "profanity": "1"}, "none"),
    "boludo amistoso": ("banter_friendly", CLEAN, "none"),
    "negación": ("negation", CLEAN, "none"),
    "sarcasmo": ("sarcasm", CLEAN, "none"),
    "hablar de insultos": ("quoting_reporting", CLEAN, "none"),
    "crítica sin insulto": ("criticism_no_insult", CLEAN, "none"),
    "cotidiano": ("everyday", {**CLEAN, "sexual_harassment": "0", "profanity": "0"}, "none"),
}


def _row(functionality: str, text: str, labels: dict, target: str, notes: str = "") -> dict[str, str]:
    if target not in TARGETS + ("",):
        raise ValueError(f"target desconocido {target!r} en {text!r}")
    row = dict.fromkeys(COLUMNS, "")
    row.update(lang="es", functionality=functionality, author="generated", text=text.strip(), target=target,
               reviewed="0", notes=notes)
    for category, value in labels.items():
        value = str(value)
        if category not in CATEGORIES or value not in ("1", "0", "?", ""):
            raise ValueError(f"etiqueta inválida {category}={value!r} en {text!r}")
        row[category] = value
    return row


def seed_rows(generated: list[dict]) -> list[dict[str, str]]:
    rows = []
    for example in generated:
        if example["category"] not in SEED_MAP:
            raise ValueError(f"categoría sin mapeo: {example['category']!r}")
        functionality, labels, target = SEED_MAP[example["category"]]
        rows.append(_row(functionality, example["text"], labels, target))
    return rows


def draft_rows(drafts: list[dict]) -> list[dict[str, str]]:
    return [
        _row(d["functionality"], d["text"], {c: d.get(c, "") for c in CATEGORIES}, d.get("target", ""), d.get("notes", ""))
        for d in drafts
    ]


def build_suite(generated: list[dict], drafts: list[dict]) -> list[dict[str, str]]:
    rows = seed_rows(generated) + draft_rows(drafts)
    seen = set()
    for i, row in enumerate(rows, start=1):
        key = normalize_text(row["text"])
        if key in seen:
            raise ValueError(f"texto repetido: {row['text']!r}")
        seen.add(key)
        row["id"] = f"es-{i:04d}"
    return rows


def write_new_suite(path: Path, rows: list[dict[str, str]], force: bool = False) -> None:
    if Path(path).exists() and not force:
        raise FileExistsError(f"{path} ya existe y puede tener revisiones. Usá --force para pisarlo.")
    write_suite(path, rows)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
