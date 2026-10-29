"""Every benchmark source by name, and how examples are sampled and split."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bench.sources.hatecheck import load_hatecheck
from bench.sources.portuguese import load_olidbr, load_toldbr
from bench.sources.suite import load_suite
from bench.taxonomy import Example, has_gold

SUITES_DIR = Path(__file__).resolve().parents[2] / "suites"
DEFAULT_MAX_PER_SOURCE = 4000
CALIB_ONE_IN = 5


@dataclass(frozen=True)
class Source:
    name: str
    lang: str
    license: str
    # Called with include_unreviewed; only the suite uses it.
    load: Callable[[bool], list[Example]]
    # Large corpora are sampled; functional tests and the suite are used whole.
    sampled: bool = False


SOURCES = {s.name: s for s in (
    Source("suite-es", "es", "propia",
           lambda unreviewed: load_suite(SUITES_DIR / "es_casino.csv", "suite-es", unreviewed)),
    Source("hatecheck-es", "es", "CC BY 4.0", lambda _: load_hatecheck("es")),
    Source("hatecheck-pt", "pt", "CC BY 4.0", lambda _: load_hatecheck("pt")),
    Source("hatecheck-en", "en", "CC BY 4.0", lambda _: load_hatecheck("en")),
    Source("toldbr", "pt", "CC BY-SA 4.0", lambda _: load_toldbr(), sampled=True),
    Source("olidbr", "pt", "CC BY 4.0", lambda _: load_olidbr()),
)}


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16)


def sample(examples: list[Example], max_n: int) -> list[Example]:
    """Up to `max_n` examples that have something to score, the same ones on every run."""
    keep = [e for e in examples if has_gold(e)]
    if len(keep) <= max_n:
        return keep
    return sorted(keep, key=lambda e: stable_hash(e.id))[:max_n]


def split_of(example: Example) -> str:
    """`calib` (operating thresholds are chosen here) or `eval` (what the report shows).

    Decided by cluster, so a HateCheck template never lands on both sides."""
    return "calib" if stable_hash(example.cluster_key) % CALIB_ONE_IN == 0 else "eval"


def load_sources(names, max_per_source: int = DEFAULT_MAX_PER_SOURCE, include_unreviewed: bool = False,
                 sources: dict | None = None, log=print) -> list[Example]:
    sources = SOURCES if sources is None else sources
    unknown = [n for n in names if n not in sources]
    if unknown:
        raise ValueError(f"fuente desconocida: {', '.join(unknown)}. Opciones: {', '.join(sources)}")
    examples = []
    for name in names:
        source = sources[name]
        loaded = source.load(include_unreviewed)
        loaded = sample(loaded, max_per_source) if source.sampled else [e for e in loaded if has_gold(e)]
        if not loaded:
            log(f"Aviso: {name} no tiene ejemplos con etiquetas (¿la suite todavía no está revisada?). Se saltea.")
        examples.extend(loaded)
    return examples
