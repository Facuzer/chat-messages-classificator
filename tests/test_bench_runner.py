import pytest

from bench.cache import PredictionCache, text_hash
from bench.contestants.base import effective_scores
from bench.runner import run
from bench.taxonomy import Example

QUIET = {"log": lambda _: None}


class FakeContestant:
    name = "fake"
    version = "1"
    notes = ""

    def __init__(self, languages=("es", "pt"), fail_on_call=None):
        self.languages = frozenset(languages)
        self.calls = []
        self.fail_on_call = fail_on_call

    def predict(self, texts, lang):
        self.calls.append(list(texts))
        if self.fail_on_call == len(self.calls):
            raise ConnectionError("se cortó la conexión")
        return [{"len": float(len(t))} for t in texts]

    def to_scores(self, raw):
        return {"insult": raw["len"]}


def ex(i, text=None, lang="es", source="s"):
    return Example(id=f"{source}:{i}", source=source, lang=lang, text=text or f"mensaje {i}", labels={"insult": True})


def test_predicts_every_example_once_in_batches(tmp_path):
    contestant = FakeContestant()

    raw = run([contestant], [ex(i) for i in range(5)], PredictionCache(tmp_path), batch_size=2, **QUIET)

    assert [len(batch) for batch in contestant.calls] == [2, 2, 1]
    assert raw["fake"]["s:3"] == {"len": float(len("mensaje 3"))}


def test_a_second_run_comes_from_the_cache(tmp_path):
    examples = [ex(i) for i in range(3)]
    run([FakeContestant()], examples, PredictionCache(tmp_path), **QUIET)
    again = FakeContestant()

    raw = run([again], examples, PredictionCache(tmp_path), **QUIET)

    assert again.calls == [] and len(raw["fake"]) == 3


def test_an_edited_text_is_predicted_again(tmp_path):
    run([FakeContestant()], [ex(1), ex(2)], PredictionCache(tmp_path), **QUIET)
    again = FakeContestant()

    raw = run([again], [ex(1), ex(2, text="texto corregido")], PredictionCache(tmp_path), **QUIET)

    assert again.calls == [["texto corregido"]]
    assert raw["fake"]["s:2"] == {"len": float(len("texto corregido"))}


def test_a_crashed_run_resumes_after_the_last_finished_batch(tmp_path):
    examples = [ex(i) for i in range(4)]
    with pytest.raises(ConnectionError):
        run([FakeContestant(fail_on_call=2)], examples, PredictionCache(tmp_path), batch_size=2, **QUIET)
    again = FakeContestant()

    run([again], examples, PredictionCache(tmp_path), batch_size=2, **QUIET)

    assert again.calls == [["mensaje 2", "mensaje 3"]]


def test_a_truncated_last_line_is_ignored_and_does_not_corrupt_the_next_append(tmp_path):
    cache, contestant = PredictionCache(tmp_path), FakeContestant()
    run([contestant], [ex(1)], cache, **QUIET)
    path = next(tmp_path.rglob("*.jsonl"))
    with path.open("a", encoding="utf-8") as f:
        f.write('{"id": "s:2", "text_sh')

    assert set(cache.load(contestant, "s")) == {"s:1"}
    run([FakeContestant()], [ex(1), ex(2)], cache, **QUIET)
    assert set(cache.load(contestant, "s")) == {"s:1", "s:2"}


def test_languages_the_contestant_does_not_support_are_skipped(tmp_path):
    raw = run([FakeContestant(languages=("es",))], [ex(1), ex(2, lang="en", source="e")], PredictionCache(tmp_path), **QUIET)

    assert set(raw["fake"]) == {"s:1"}


def test_the_wrong_number_of_outputs_is_an_error(tmp_path):
    contestant = FakeContestant()
    contestant.predict = lambda texts, lang: []

    with pytest.raises(RuntimeError, match="fake"):
        run([contestant], [ex(1)], PredictionCache(tmp_path), **QUIET)


def test_cache_folder_name_is_safe_on_windows(tmp_path):
    contestant = FakeContestant()
    contestant.version = "rev:abc/def"

    PredictionCache(tmp_path).append(contestant, "s", [("s:1", text_hash("x"), {"len": 1.0})])

    assert [p.parent.name for p in tmp_path.rglob("*.jsonl")] == ["fake@rev_abc_def"]


def test_each_contestant_is_unloaded_once_it_is_done(tmp_path):
    events = []

    class Unloadable(FakeContestant):
        def predict(self, texts, lang):
            events.append("predict")
            return super().predict(texts, lang)

        def unload(self):
            events.append("unload")

    run([Unloadable()], [ex(1), ex(2, lang="pt", source="p")], PredictionCache(tmp_path), **QUIET)

    assert events == ["predict", "predict", "unload"]


class Fixed:
    def __init__(self, scores):
        self.scores = scores

    def to_scores(self, raw):
        return self.scores


def test_flag_is_the_max_of_insult_threat_and_identity_hate_only_when_all_three_are_present():
    assert effective_scores(Fixed({"insult": 0.2, "threat": 0.9, "identity_hate": 0.1}), {})["flag"] == 0.9
    assert "flag" not in effective_scores(Fixed({"identity_hate": 0.7}), {})
    assert effective_scores(Fixed({"flag": 0.3, "insult": 0.9, "threat": 0.9, "identity_hate": 0.9}), {})["flag"] == 0.3


def test_a_line_cut_inside_a_non_ascii_id_does_not_crash_loading(tmp_path):
    cache, contestant = PredictionCache(tmp_path), FakeContestant()
    run([contestant], [ex("señal_1", text="hola")], cache, **QUIET)
    path = next(tmp_path.rglob("*.jsonl"))
    good = path.read_bytes()
    path.write_bytes(good + b'{"id": "s:se' + "ñ".encode("utf-8")[:1])

    assert set(cache.load(contestant, "s")) == {"s:señal_1"}


def test_ids_with_unicode_line_separators_survive_the_cache(tmp_path):
    cache, contestant = PredictionCache(tmp_path), FakeContestant()
    run([contestant], [ex("a\u2028b")], cache, **QUIET)

    assert set(cache.load(contestant, "s")) == {"s:a\u2028b"}
