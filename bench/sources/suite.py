"""The hand-written suite: a CSV that reviewers edit in Excel or Google Sheets.

Category cells hold 1 (yes), 0 (no), ? (reviewers disagree or can't tell) or nothing (not reviewed).
? and empty are both left out of scoring. Spanish-locale Excel saves with `;` and a BOM, so both
delimiters are read; the file is written with `;` so a double click opens it in columns.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from bench.taxonomy import CATEGORIES, Example

COLUMNS = ("id", "lang", "functionality", "author", "text", *CATEGORIES, "target", "reviewed", "notes")
AUTHORS = ("generated", "human")


class SuiteError(ValueError):
    pass


def _cell(value: str, line: int, column: str) -> bool | None:
    if value == "1":
        return True
    if value == "0":
        return False
    if value in ("", "?"):
        return None
    raise SuiteError(f"línea {line}, columna {column}: {value!r} no es 1, 0, ? ni vacío")


def parse_suite(text: str, source: str) -> list[tuple[Example, bool]]:
    """(example, reviewed) for every non-empty row. Raises SuiteError naming the line of the first problem."""
    text = text.lstrip("\ufeff")
    header = text.split("\n", 1)[0]
    delimiter = ";" if header.count(";") > header.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise SuiteError(f"faltan columnas: {', '.join(missing)}")

    parsed, seen = [], set()
    for raw in reader:
        row = {c: (raw.get(c) or "").strip() for c in COLUMNS}
        if not any(row.values()):
            continue
        line = reader.line_num
        if not row["id"] or row["id"] in seen:
            raise SuiteError(f"línea {line}: id vacío o repetido ({row['id']!r})")
        seen.add(row["id"])
        if row["author"] not in AUTHORS:
            raise SuiteError(f"línea {line}: author tiene que ser {' o '.join(AUTHORS)}, no {row['author']!r}")
        if row["reviewed"] not in ("1", "0", ""):
            raise SuiteError(f"línea {line}: reviewed tiene que ser 1 o 0, no {row['reviewed']!r}")
        labels = {c: _cell(row[c], line, c) for c in CATEGORIES}
        try:
            example = Example(
                id=f"{source}:{row['id']}",
                source=source,
                lang=row["lang"],
                text=row["text"],
                labels=labels,
                target=row["target"] or None,
                functionality=row["functionality"] or None,
                author=row["author"],
            )
        except ValueError as e:
            raise SuiteError(f"línea {line}: {e}") from None
        parsed.append((example, row["reviewed"] == "1"))
    return parsed


def read_suite_text(path: Path) -> str:
    """The text of a suite CSV, or a SuiteError a reviewer can act on."""
    try:
        return Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise SuiteError(f"{path}: no existe") from None
    except UnicodeDecodeError:
        raise SuiteError(f"{path}: el archivo no está en UTF-8. En Excel usá «Guardar como» → «CSV UTF-8 (delimitado por comas)».") from None


def load_suite(path: Path, source: str, include_unreviewed: bool = False) -> list[Example]:
    rows = parse_suite(read_suite_text(path), source)
    return [example for example, reviewed in rows if reviewed or include_unreviewed]


def write_suite(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, delimiter=";")
    writer.writeheader()
    writer.writerows({c: row.get(c, "") for c in COLUMNS} for row in rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buffer.getvalue(), encoding="utf-8-sig", newline="")


def agreement(a: list[Example], b: list[Example]) -> dict[str, dict]:
    """Cohen's kappa per category between two reviewers, on the rows both labeled 1 or 0."""
    from sklearn.metrics import cohen_kappa_score

    other = {e.id: e for e in b}
    result = {}
    for category in CATEGORIES:
        pairs = [(e.labels.get(category), other[e.id].labels.get(category)) for e in a if e.id in other]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        xs, ys = [x for x, _ in pairs], [y for _, y in pairs]
        # With a single class in both columns kappa is undefined (sklearn returns nan).
        kappa = float(cohen_kappa_score(xs, ys)) if len(set(xs) | set(ys)) > 1 else None
        result[category] = {"n": len(pairs), "kappa": kappa}
    return result
