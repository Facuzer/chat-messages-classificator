"""Command line for the moderation benchmark (scripts/bench.py). See docs/benchmark.md."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from app.examples import text_key
from app.jsonfile import read_json, write_json_atomic
from bench.cache import PredictionCache
from bench.contestants import checkpoint_factories
from bench.latency import measure
from bench.report import render
from bench.runner import run
from bench.scoreboard import build_scoreboard
from bench.sources import DEFAULT_MAX_PER_SOURCE, SOURCES, load_sources, stable_hash
from bench.sources.suite import SuiteError, agreement, parse_suite, read_suite_text
from bench.taxonomy import LANGS, SCORED, gold

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "bench_out"


class CliError(Exception):
    pass


def _pick(value: str | None, valid, default, kind: str) -> list[str]:
    names = [n.strip() for n in value.split(",") if n.strip()] if value else list(default)
    unknown = [n for n in names if n not in valid]
    if unknown:
        raise CliError(f"{kind} inexistente: {', '.join(unknown)}. Opciones: {', '.join(valid)}")
    return names


def cmd_sources(args, sources, checkpoints) -> int:
    names = _pick(args.sources, sources, sources, "fuente")
    examples = load_sources(names, args.max_per_source, args.include_unreviewed, sources=sources)
    print(f"{'fuente':<14} {'idioma':<6} {'ejemplos':>8}  " + "  ".join(f"{label} (+/-)" for label in SCORED))
    for name in names:
        mine = [e for e in examples if e.source == name]
        counts = []
        for label in SCORED:
            values = [gold(e, label) for e in mine]
            counts.append(f"{values.count(True)}/{values.count(False)}")
        print(f"{name:<14} {sources[name].lang:<6} {len(mine):>8}  " + "  ".join(counts))
    return 0


def _checkpoints(value: str | None, checkpoints: dict, device: str | None) -> list:
    """The named checkpoints (all of them by default), built for `device`."""
    if not checkpoints:
        raise CliError("No hay checkpoints entrenados en models/: corré scripts/finetune.py.")
    names = _pick(value, checkpoints, checkpoints, "checkpoint")
    return [checkpoints[n](device) for n in names]


def _without_trained_texts(examples, contestants):
    """Every checkpoint is measured without the texts any of them trained on, so all of them face the
    same examples and none is scored on what it memorized."""
    trained = {c.name: getattr(c, "trained_on", frozenset()) for c in contestants}
    keys = frozenset().union(*trained.values())
    if not keys:
        return examples, None
    kept = [e for e in examples if text_key(e.text) not in keys]
    n = len(examples) - len(kept)
    print(f"Se excluyen {n} ejemplos con los que se entrenó algún checkpoint.")
    return kept, {"n": n, "contestants": [name for name, k in trained.items() if k]}


def cmd_run(args, sources, checkpoints) -> int:
    source_names = _pick(args.sources, sources, sources, "fuente")
    contestants = _checkpoints(args.checkpoints, checkpoints, None if args.device == "auto" else args.device)
    examples = load_sources(source_names, args.max_per_source, args.include_unreviewed, sources=sources)
    if not examples:
        raise CliError("No hay ejemplos para evaluar.")
    examples, excluded = _without_trained_texts(examples, contestants)
    out = Path(args.out)
    raw = run(contestants, examples, PredictionCache(out / "predictions"), batch_size=args.batch_size)
    print("Calculando métricas…")
    board = build_scoreboard(examples, contestants, raw, n_boot=args.bootstrap)
    board["meta"] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "preliminary": args.include_unreviewed,
        "n_boot": args.bootstrap,
        "excluded": excluded,
        "contestants": [{"name": c.name, "version": c.version, "license": c.license, "notes": c.notes} for c in contestants],
        "sources": [{"name": n, "lang": sources[n].lang, "license": sources[n].license,
                     "n": sum(e.source == n for e in examples)} for n in source_names],
    }
    board["latency"] = read_json(out / "latency.json", [])
    write_json_atomic(out / "scoreboard.json", board)
    (out / "scoreboard.md").write_text(render(board), encoding="utf-8")
    print(f"Listo: {out / 'scoreboard.md'}")
    return 0


def cmd_latency(args, sources, checkpoints) -> int:
    contestants = _checkpoints(args.checkpoints, checkpoints, args.device)
    in_lang = [n for n, s in sources.items() if s.lang == args.lang]
    examples = load_sources(in_lang, include_unreviewed=True, sources=sources, log=lambda _: None)
    texts = [e.text for e in sorted(examples, key=lambda e: stable_hash(e.id))][: args.n]
    if not texts:
        raise CliError(f"No hay mensajes en '{args.lang}' para medir.")
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    out = Path(args.out)
    entries = read_json(out / "latency.json", [])
    for contestant in contestants:
        result = measure(contestant, texts, args.lang, batch_size=args.batch_size)
        unload = getattr(contestant, "unload", None)
        if unload is not None:
            unload()
        # Saved after every contestant: a CUDA out-of-memory on a later one must not discard the earlier numbers.
        key = (contestant.name, args.device, args.lang, args.threads)
        entries = [e for e in entries if (e["contestant"], e["device"], e["lang"], e.get("threads")) != key]
        entries.append({"contestant": contestant.name, "device": args.device, "lang": args.lang,
                        "threads": args.threads, **result})
        write_json_atomic(out / "latency.json", entries)
        print(f"{contestant.name:<18} p50 {result['p50_ms']:.0f} ms · p95 {result['p95_ms']:.0f} ms · "
              f"{result['throughput_per_s'] or 0:.0f} mensajes/s en lotes de {args.batch_size}")
    return 0


def cmd_agreement(args, sources, checkpoints) -> int:
    a, b = ([e for e, _ in parse_suite(read_suite_text(p), "suite")] for p in (args.a, args.b))
    print(f"{'categoría':<20} {'filas':>6} {'kappa':>6}")
    for category, result in agreement(a, b).items():
        kappa = "n/a" if result["kappa"] is None else f"{result['kappa']:.2f}"
        print(f"{category:<20} {result['n']:>6} {kappa:>6}")
    return 0


def main(argv=None, sources=None, checkpoints=None) -> int:
    """`checkpoints` maps a name to a factory(device); by default, every complete checkpoint in models/."""
    sources = SOURCES if sources is None else sources
    parser = argparse.ArgumentParser(prog="bench.py", description="Compara los checkpoints con el mismo examen (ver docs/benchmark.md).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sources", help="cuántos ejemplos hay por fuente y categoría")
    p.add_argument("--sources", help="lista separada por comas (por defecto, todas)")
    p.add_argument("--max-per-source", type=int, default=DEFAULT_MAX_PER_SOURCE)
    p.add_argument("--include-unreviewed", action="store_true")
    p.set_defaults(handler=cmd_sources)

    p = sub.add_parser("run", help="evalúa los checkpoints y escribe el scoreboard")
    p.add_argument("--sources", help="lista separada por comas (por defecto, todas)")
    p.add_argument("--checkpoints", help="carpetas de models/ separadas por comas (por defecto, todas)")
    p.add_argument("--max-per-source", type=int, default=DEFAULT_MAX_PER_SOURCE)
    p.add_argument("--include-unreviewed", action="store_true", help="incluir filas sin revisar (scoreboard PRELIMINAR)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--bootstrap", type=int, default=1000, help="remuestreos para los intervalos de confianza")
    p.add_argument("--batch-size", type=int, default=8, help="mensajes por lote; 8 anda bien en una GPU de 4 GB")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.set_defaults(handler=cmd_run)

    p = sub.add_parser("latency", help="mide cuánto tarda cada checkpoint por mensaje")
    p.add_argument("--checkpoints", help="carpetas de models/ separadas por comas (por defecto, todas)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--lang", choices=list(LANGS), default="es")
    p.add_argument("--n", type=int, default=200, help="cantidad de mensajes")
    p.add_argument("--threads", type=int, help="hilos de CPU (por ejemplo 4, como un server chico)")
    p.add_argument("--batch-size", type=int, default=8, help="mensajes por lote; 8 anda bien en una GPU de 4 GB")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.set_defaults(handler=cmd_latency)

    p = sub.add_parser("agreement", help="kappa entre dos revisiones de la suite")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(handler=cmd_agreement)

    args = parser.parse_args(argv)
    try:
        return args.handler(args, sources, checkpoint_factories() if checkpoints is None else checkpoints)
    except (CliError, SuiteError) as e:
        print(e)
        return 1
