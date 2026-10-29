import json

import pytest

from app.checkpoints import ActiveCheckpoint, checkpoint_problem


def make_checkpoint(models_dir, name="ft-1"):
    path = models_dir / name
    path.mkdir(parents=True)
    (path / "training_meta.json").write_text(json.dumps({"threshold": 0.4}), encoding="utf-8")
    return path


def test_no_active_checkpoint_by_default(tmp_path):
    assert ActiveCheckpoint(tmp_path).get() is None


def test_activate_then_get_returns_path_and_metrics(tmp_path):
    path = make_checkpoint(tmp_path)
    active = ActiveCheckpoint(tmp_path)

    active.activate(path, {"f1": 0.8})

    got = ActiveCheckpoint(tmp_path).get()
    assert got["path"] == path
    assert got["metrics"] == {"f1": 0.8}
    assert got["activated_at"]


def test_deactivate_leaves_nothing_active(tmp_path):
    active = ActiveCheckpoint(tmp_path)
    active.activate(make_checkpoint(tmp_path), {})

    active.deactivate()

    assert active.get() is None


def test_deactivate_without_active_checkpoint_is_harmless(tmp_path):
    ActiveCheckpoint(tmp_path).deactivate()


def test_active_checkpoint_whose_folder_was_deleted_counts_as_none(tmp_path):
    path = make_checkpoint(tmp_path)
    active = ActiveCheckpoint(tmp_path)
    active.activate(path, {})

    (path / "training_meta.json").unlink()
    path.rmdir()

    assert active.get() is None



def mmbert_checkpoint(models_dir, name="ft-1", **meta):
    path = models_dir / name
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    meta = {"labels": ["insult"], "threshold": 0.4} | meta
    (path / "training_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_a_trained_and_evaluated_checkpoint_has_no_problem(tmp_path):
    assert checkpoint_problem(mmbert_checkpoint(tmp_path)) is None


def test_a_checkpoint_whose_evaluation_never_finished_is_incomplete(tmp_path):
    assert "incompleto" in checkpoint_problem(mmbert_checkpoint(tmp_path, threshold=None))


@pytest.mark.parametrize("missing", ["config.json", "training_meta.json"])
def test_a_folder_without_the_model_files_is_not_a_checkpoint(tmp_path, missing):
    path = mmbert_checkpoint(tmp_path)
    (path / missing).unlink()

    assert "no es un checkpoint" in checkpoint_problem(path)


def test_a_checkpoint_of_another_model_is_not_a_checkpoint(tmp_path):
    path = tmp_path / "ft-otro"
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "training_meta.json").write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")

    assert "no es un checkpoint" in checkpoint_problem(path)


def load_script(name):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_activate_model_refuses_an_incomplete_checkpoint_and_lists_it_as_such(tmp_path, monkeypatch, capsys):
    script = load_script("activate_model")
    monkeypatch.setattr(script, "MODELS_DIR", tmp_path)
    path = mmbert_checkpoint(tmp_path, "ft-cortado", threshold=None)

    monkeypatch.setattr("sys.argv", ["activate_model.py", str(path)])
    assert script.main() == 1
    assert ActiveCheckpoint(tmp_path).get() is None

    monkeypatch.setattr("sys.argv", ["activate_model.py", "--list"])
    assert script.main() == 0
    assert "ft-cortado" in capsys.readouterr().out.split("incompleto")[0]
