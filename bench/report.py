"""bench_out/scoreboard.md, in Spanish like the rest of the team-facing docs."""

from __future__ import annotations

from collections import defaultdict

from bench.taxonomy import FLAG, SCORED

LEGEND = [
    "## Cómo leer esto",
    "",
    "- **AUROC [IC 95%]**: probabilidad de que el modelo puntúe más alto un mensaje dañino que uno sano. "
    "0.50 es azar y 1.00 es perfecto. El intervalo muestra cuánto puede variar por azar.",
    "- **R · FPR**: con el umbral de operación de cada modelo (el que marca al 5% de los mensajes sanos de la "
    "partición de calibración), qué parte de los dañinos detecta (R) y qué parte de los sanos marca (FPR) en esa fuente.",
    "- **pocos datos (a+/b−)**: menos de 30 ejemplos de alguna clase; no sacar conclusiones.",
    "- **n/a**: el modelo no responde esa categoría. **—**: la fuente no tiene etiquetas de esa categoría.",
    "- Estas fuentes tienen muchos más mensajes dañinos que un chat real (donde son el 1–5%): en producción "
    "la precisión va a ser menor.",
    "",
]


def _num(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *("| " + " | ".join(r) + " |" for r in rows)]


def _auroc(row: dict | None) -> str:
    if row is None:
        return "n/a"
    if not row["enough"]:
        return f"pocos datos ({row['positives']}+/{row['n'] - row['positives']}−)"
    ci = row["auroc_ci"]
    return _num(row["auroc"]) + (f" [{_num(ci[0])}–{_num(ci[1])}]" if ci else "")


def _flag(row: dict | None) -> str:
    cell = _auroc(row)
    if row is not None and row["enough"] and row["recall_at_op"] is not None:
        cell += f"<br>R {_pct(row['recall_at_op'])} · FPR {_pct(row['fpr_at_op'])}"
    return cell


def render(board: dict) -> str:
    meta = board["meta"]
    names = [c["name"] for c in meta["contestants"]]
    rows = {(r["contestant"], r["source"], r["label"]): r for r in board["slices"]}
    covered = {(r["source"], r["label"]): r for r in board["coverage"]}
    sources = sorted({(r["source"], r["lang"]) for r in board["coverage"]})

    def cell(name, source, label, fmt=_auroc):
        if (source, label) not in covered:
            return "—"
        return fmt(rows.get((name, source, label)))

    lines = ["# Scoreboard de moderación", ""]
    if meta["preliminary"]:
        lines += ["> **PRELIMINAR:** incluye filas de la suite que nadie revisó todavía. Sirve para probar la "
                  "herramienta, no para decidir.", ""]
    lines += [f"Generado: {meta['created_at']} · Remuestreos bootstrap: {meta['n_boot']}", ""]
    excluded = meta.get("excluded")
    if excluded:
        lines += [f"Se excluyeron {excluded['n']} ejemplos con los que se entrenó {', '.join(excluded['contestants'])}: "
                  "todos los modelos se miden sin ellos.", ""]
    lines += LEGEND

    lines += ["## Cobertura (ejemplos de evaluación: dañinos/sanos)", ""]
    lines += _table(["fuente", "idioma", *SCORED], [
        [s, lang, *(f"{covered[(s, label)]['positives']}/{covered[(s, label)]['negatives']}" if (s, label) in covered else "—"
                    for label in SCORED)]
        for s, lang in sources
    ])

    lines += ["", "## ¿Hay que moderar? (`flag` = insult, threat o identity_hate)", ""]
    lines += _table(["modelo", *(s for s, _ in sources)], [[n, *(cell(n, s, FLAG, _flag) for s, _ in sources)] for n in names])

    if board["comparisons"]:
        lines += ["", "## Diferencia de AUROC contra el mejor de cada fuente (`flag`, bootstrap pareado)", "",
                  "Si el intervalo incluye 0, no hay evidencia de que el mejor sea mejor de verdad.", ""]
        lines += _table(["fuente", "mejor", "modelo", "diferencia [IC 95%]"], [
            [r["source"], r["best"], r["contestant"], f"{_num(r['diff'])} [{_num(r['lo'])} – {_num(r['hi'])}]"]
            for r in board["comparisons"]
        ])

    lines += ["", "## Por categoría (AUROC)", ""]
    for s, lang in sources:
        lines += [f"### {s} ({lang})", ""]
        lines += _table(["modelo", *SCORED], [[n, *(cell(n, s, label) for label in SCORED)] for n in names]) + [""]

    by_functionality = defaultdict(dict)
    for r in board["functionalities"]:
        by_functionality[(r["source"], r["functionality"])][r["contestant"]] = r
    if by_functionality:
        lines += ["## Por funcionalidad (aciertos con el umbral de operación)", "",
                  "En HateCheck se mide `identity_hate`, que es lo que HateCheck evalúa; en la suite propia, `flag`, "
                  "salvo `sexual_harassment`, que se mide con su propia etiqueta.", ""]
        for source in sorted({s for s, _ in by_functionality}):
            table = [
                [f, str(max(r["n"] for r in per.values())), *(_pct(per[n]["accuracy"]) if n in per else "n/a" for n in names)]
                for (s, f), per in sorted(by_functionality.items()) if s == source
            ]
            lines += [f"### {source}", ""] + _table(["funcionalidad", "n", *names], table) + [""]

    if board.get("latency"):
        lines += ["## Latencia", ""]
        lines += _table(["modelo", "dispositivo", "idioma", "lote", "p50 ms", "p95 ms", "mensajes/s en lotes"], [
            [r["contestant"], r["device"] + (f" ({r['threads']} hilos)" if r.get("threads") else ""), r["lang"],
             str(r["batch_size"]), f"{r['p50_ms']:.0f}", f"{r['p95_ms']:.0f}",
             "n/a" if r["throughput_per_s"] is None else f"{r['throughput_per_s']:.0f}"]
            for r in board["latency"]
        ]) + [""]

    lines += ["## Modelos", ""]
    for c in meta["contestants"]:
        shown = ", ".join(f"{label}={t:.3g}" for label, t in board["thresholds"].get(c["name"], {}).items() if t is not None)
        lines.append(f"- **{c['name']}** (`{c['version']}`, {c['license']}): {c['notes']} "
                     f"Umbrales de operación: {shown or 'n/a'}.")
    lines += ["", "## Fuentes", ""]
    lines += _table(["fuente", "idioma", "licencia", "ejemplos"],
                    [[s["name"], s["lang"], s["license"], str(s["n"])] for s in meta["sources"]])
    return "\n".join(lines) + "\n"
