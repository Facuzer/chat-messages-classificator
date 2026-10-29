"""Runs every contestant over every example, through the prediction cache."""

from __future__ import annotations

from collections import defaultdict

from bench.cache import PredictionCache, text_hash


def run(contestants, examples, cache: PredictionCache, batch_size: int = 32, log=print) -> dict[str, dict[str, dict]]:
    groups = defaultdict(list)
    for e in examples:
        groups[(e.source, e.lang)].append(e)

    raw: dict[str, dict[str, dict]] = {}
    for contestant in contestants:
        raw[contestant.name] = {}
        for (source, lang), group in sorted(groups.items()):
            if lang not in contestant.languages:
                log(f"{contestant.name}: no soporta '{lang}', se saltea {source}.")
                continue
            cached = cache.load(contestant, source)
            todo = [e for e in group if cached.get(e.id, ("",))[0] != text_hash(e.text)]
            log(f"{contestant.name} · {source}: {len(todo)} mensajes nuevos, {len(group) - len(todo)} en caché.")
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                outputs = contestant.predict([e.text for e in batch], lang)
                if len(outputs) != len(batch):
                    raise RuntimeError(f"{contestant.name} devolvió {len(outputs)} resultados para {len(batch)} textos")
                rows = [(e.id, text_hash(e.text), out) for e, out in zip(batch, outputs)]
                cache.append(contestant, source, rows)
                cached.update({example_id: (sha, out) for example_id, sha, out in rows})
            raw[contestant.name].update({e.id: cached[e.id][1] for e in group})
        unload = getattr(contestant, "unload", None)
        if unload is not None:
            unload()
    return raw
