import json

import pytest

from bench.cli import main
from bench.latency import measure, percentile
from bench.sources import Source
from bench.sources.suite import write_suite
from bench.taxonomy import Example
from trainer.data import text_key


class Fake:
    name = "fake"
    version = "1"
    notes = "Modelo de prueba."
    license = "MIT"
    languages = frozenset({"es"})

    def __init__(self, device=None):
        self.calls = 0

    def predict(self, texts, lang):
        self.calls += 1
        return [{"p": 0.9 if "idiota" in t else 0.1} for t in texts]

    def to_scores(self, raw):
        return {"insult": raw["p"], "threat": 0.0, "identity_hate": 0.0}


def fake_sources():
    examples = [Example(id=f"f:{i}", source="fake-es", lang="es", text=f"sos un idiota {i}" if i % 2 else f"hola {i}",
                        labels={"insult": bool(i % 2), "threat": False, "identity_hate": False}) for i in range(400)]
    return {"fake-es": Source("fake-es", "es", "test", lambda _: examples)}


def run_main(*args, checkpoints=None):
    return main(list(args), sources=fake_sources(), checkpoints=checkpoints or {"fake": Fake})


def test_run_writes_markdown_and_json_scoreboards(tmp_path):
    assert run_main("run", "--checkpoints", "fake", "--out", str(tmp_path), "--bootstrap", "20") == 0

    board = json.loads((tmp_path / "scoreboard.json").read_text(encoding="utf-8"))
    [flag] = [r for r in board["slices"] if r["label"] == "flag" and r["source"] == "fake-es"]
    assert flag["auroc"] == 1.0 and flag["enough"] is True
    md = (tmp_path / "scoreboard.md").read_text(encoding="utf-8")
    assert "Modelo de prueba." in md and "PRELIMINAR" not in md


def test_include_unreviewed_marks_the_scoreboard_preliminary(tmp_path):
    run_main("run", "--checkpoints", "fake", "--out", str(tmp_path), "--bootstrap", "5", "--include-unreviewed")

    assert "PRELIMINAR" in (tmp_path / "scoreboard.md").read_text(encoding="utf-8")


def test_unknown_names_are_reported(tmp_path, capsys):
    assert run_main("run", "--sources", "nope", "--checkpoints", "fake", "--out", str(tmp_path)) == 1
    assert "nope" in capsys.readouterr().out


def test_internal_errors_keep_their_traceback(tmp_path):
    class Broken(Fake):
        def predict(self, texts, lang):
            raise ValueError("bug interno")

    with pytest.raises(ValueError, match="bug interno"):
        run_main("run", "--checkpoints", "fake", "--out", str(tmp_path), checkpoints={"fake": Broken})


def test_run_and_latency_send_batches_of_eight_by_default(tmp_path):
    sizes = []

    class Recording(Fake):
        def predict(self, texts, lang):
            sizes.append(len(texts))
            return super().predict(texts, lang)

    assert run_main("run", "--checkpoints", "fake", "--out", str(tmp_path), "--bootstrap", "5",
                    checkpoints={"fake": Recording}) == 0
    assert max(sizes) == 8

    assert run_main("latency", "--checkpoints", "fake", "--n", "20", "--out", str(tmp_path),
                    checkpoints={"fake": Recording}) == 0
    [entry] = json.loads((tmp_path / "latency.json").read_text(encoding="utf-8"))
    assert entry["batch_size"] == 8


def test_latency_keeps_one_entry_per_checkpoint_device_and_language(tmp_path):
    args = ("latency", "--checkpoints", "fake", "--device", "cpu", "--lang", "es", "--n", "5", "--out", str(tmp_path))

    assert run_main(*args) == 0 and run_main(*args) == 0

    entries = json.loads((tmp_path / "latency.json").read_text(encoding="utf-8"))
    assert [(e["contestant"], e["device"], e["lang"], e["n"]) for e in entries] == [("fake", "cpu", "es", 5)]


def test_latency_keeps_measurements_with_different_cpu_threads_apart(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    args = ("latency", "--checkpoints", "fake", "--n", "4", "--out", str(tmp_path))

    assert run_main(*args) == 0 and run_main(*args, "--threads", "4") == 0

    entries = json.loads((tmp_path / "latency.json").read_text(encoding="utf-8"))
    assert sorted(str(e["threads"]) for e in entries) == ["4", "None"]


def test_latency_keeps_earlier_measurements_when_a_later_checkpoint_fails(tmp_path):
    class Exploding(Fake):
        name = "exploding"

        def predict(self, texts, lang):
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError):
        run_main("latency", "--checkpoints", "fake,exploding", "--n", "3", "--out", str(tmp_path),
                 checkpoints={"fake": Fake, "exploding": Exploding})

    entries = json.loads((tmp_path / "latency.json").read_text(encoding="utf-8"))
    assert [e["contestant"] for e in entries] == ["fake"]


def test_latency_unloads_each_checkpoint_after_measuring_it(tmp_path):
    unloaded = []

    class Unloading(Fake):
        def unload(self):
            unloaded.append(self.name)

    assert run_main("latency", "--checkpoints", "fake", "--n", "3", "--out", str(tmp_path), checkpoints={"fake": Unloading}) == 0
    assert unloaded == ["fake"]


def test_percentile_uses_nearest_rank():
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile(list(range(1, 11)), 95) == 10


def test_measure_times_single_messages_and_batches_after_a_warm_up():
    class Clock:
        t = 0.0

        def __call__(self):
            self.t += 0.01
            return self.t

    contestant = Fake()

    result = measure(contestant, [f"m{i}" for i in range(10)], "es", batch_size=4, clock=Clock())

    assert contestant.calls == 1 + 10 + 3
    assert result["p50_ms"] == pytest.approx(10.0) and result["p95_ms"] == pytest.approx(10.0)
    assert result["throughput_per_s"] == pytest.approx(10 / 0.01)
    assert (result["n"], result["batch_size"]) == (10, 4)


def test_agreement_reports_missing_and_non_utf8_suites(tmp_path, capsys):
    other = tmp_path / "other.csv"
    write_suite(other, [])
    ansi = tmp_path / "ansi.csv"
    ansi.write_bytes("id;text\nes-0001;vení\n".encode("cp1252"))

    assert main(["agreement", str(tmp_path / "missing.csv"), str(other)]) == 1
    assert "missing.csv: no existe" in capsys.readouterr().out
    assert main(["agreement", str(ansi), str(other)]) == 1
    assert "UTF-8" in capsys.readouterr().out


def test_run_leaves_out_the_texts_a_checkpoint_trained_on(tmp_path):
    class Trained(Fake):
        trained_on = frozenset({text_key("Sos un idiota 1"), text_key("hola 2")})

    assert run_main("run", "--checkpoints", "fake", "--out", str(tmp_path), "--bootstrap", "5",
                    checkpoints={"fake": Trained}) == 0

    board = json.loads((tmp_path / "scoreboard.json").read_text(encoding="utf-8"))
    assert board["meta"]["excluded"] == {"n": 2, "contestants": ["fake"]}
    assert board["meta"]["sources"][0]["n"] == 398
    assert "Se excluyeron 2 ejemplos" in (tmp_path / "scoreboard.md").read_text(encoding="utf-8")


def test_latency_can_limit_cpu_threads(tmp_path, monkeypatch):
    import torch

    threads = []
    monkeypatch.setattr(torch, "set_num_threads", threads.append)

    assert run_main("latency", "--checkpoints", "fake", "--out", str(tmp_path), "--n", "4", "--threads", "4") == 0
    assert threads == [4]


class Other(Fake):
    name = "other"


def test_run_compares_every_checkpoint_by_default(tmp_path):
    assert run_main("run", "--out", str(tmp_path), "--bootstrap", "5", checkpoints={"fake": Fake, "other": Other}) == 0

    board = json.loads((tmp_path / "scoreboard.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in board["meta"]["contestants"]] == ["fake", "other"]


def test_without_checkpoints_run_and_latency_say_how_to_train_one(tmp_path, capsys):
    for command in ("run", "latency"):
        assert main([command, "--out", str(tmp_path)], sources=fake_sources(), checkpoints={}) == 1
        assert "finetune.py" in capsys.readouterr().out
