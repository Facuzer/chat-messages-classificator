from bench.report import render

FLAG_ROW = {"contestant": "ft-nuevo", "source": "suite-es", "lang": "es", "label": "flag", "n": 90, "positives": 40,
            "prevalence": 0.44, "auroc": 0.91, "auroc_ci": [0.85, 0.95], "ap": 0.9, "threshold": 0.42,
            "recall_at_op": 0.7, "fpr_at_op": 0.06, "precision_at_op": 0.9, "enough": True}


def board(**overrides):
    b = {
        "meta": {"created_at": "2026-10-06T12:00:00Z", "preliminary": False, "n_boot": 100,
                 "contestants": [{"name": "ft-nuevo", "version": "q1", "license": "MIT", "notes": "Nota de ft-nuevo."},
                                 {"name": "ft-viejo", "version": "m", "license": "MIT", "notes": "Nota de ft-viejo."}],
                 "sources": [{"name": "suite-es", "lang": "es", "license": "propia", "n": 100}]},
        "coverage": [{"source": "suite-es", "lang": "es", "label": "flag", "positives": 40, "negatives": 50},
                     {"source": "suite-es", "lang": "es", "label": "profanity", "positives": 10, "negatives": 5}],
        "thresholds": {"ft-nuevo": {"flag": 0.42, "insult": None}, "ft-viejo": {"flag": None}},
        "slices": [FLAG_ROW, {**FLAG_ROW, "label": "profanity", "n": 15, "positives": 10, "enough": False}],
        "comparisons": [{"source": "suite-es", "best": "ft-nuevo", "contestant": "ft-viejo", "diff": -0.1, "lo": -0.2, "hi": -0.01}],
        "functionalities": [{"contestant": "ft-nuevo", "source": "suite-es", "functionality": "negation", "n": 30, "accuracy": 0.9}],
        "latency": [],
    }
    return b | overrides


def test_flag_cell_shows_auroc_interval_and_operating_point():
    assert "0.91 [0.85–0.95]<br>R 70% · FPR 6%" in render(board())


def test_missing_contestant_rows_say_na_and_uncovered_labels_say_dash():
    md = render(board())
    flag_line = next(line for line in md.splitlines() if line.startswith("| ft-viejo |"))

    assert "n/a" in flag_line
    assert "| ft-nuevo | — |" in md  # per-category table: insult has no coverage, so it is a dash


def test_small_slices_say_pocos_datos_instead_of_numbers():
    assert "pocos datos (10+/5−)" in render(board())


def test_preliminary_banner_only_when_preliminary():
    assert "PRELIMINAR" not in render(board())
    assert "PRELIMINAR" in render(board(meta={**board()["meta"], "preliminary": True}))


def test_comparisons_functionalities_notes_and_thresholds_are_rendered():
    md = render(board())

    assert "| suite-es | ft-nuevo | ft-viejo | -0.10 [-0.20 – -0.01] |" in md
    assert "| negation | 30 | 90% | n/a |" in md
    assert "Nota de ft-nuevo. Umbrales de operación: flag=0.42." in md
    assert "Nota de ft-viejo. Umbrales de operación: n/a." in md


def test_small_thresholds_keep_their_significant_digits():
    md = render(board(thresholds={"ft-nuevo": {"flag": 0.00412}, "ft-viejo": {"flag": None}}))

    assert "Umbrales de operación: flag=0.00412." in md


def test_latency_section_only_when_measured():
    assert "## Latencia" not in render(board())
    latency = [{"contestant": "ft-nuevo", "device": "cpu", "lang": "es", "n": 200, "p50_ms": 120.4, "p95_ms": 300.2,
                "throughput_per_s": 15.2, "batch_size": 32}]
    md = render(board(latency=latency))
    assert "| modelo | dispositivo | idioma | lote | p50 ms | p95 ms | mensajes/s en lotes |" in md
    assert "| ft-nuevo | cpu | es | 32 | 120 | 300 | 15 |" in md


def test_examples_left_out_for_trained_checkpoints_are_explained():
    b = board()
    b["meta"]["excluded"] = {"n": 289, "contestants": ["ft-nuevo"]}

    assert "Se excluyeron 289 ejemplos con los que se entrenó ft-nuevo" in render(b)
    assert "Se excluyeron" not in render(board())


def test_latency_names_the_cpu_threads_when_limited():
    latency = [{"contestant": "ft-nuevo", "device": "cpu", "lang": "es", "n": 200, "p50_ms": 21.0, "p95_ms": 30.0,
                "throughput_per_s": 90.0, "batch_size": 8, "threads": 4}]

    assert "| ft-nuevo | cpu (4 hilos) | es |" in render(board(latency=latency))
