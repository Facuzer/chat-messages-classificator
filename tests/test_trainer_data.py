import pytest

from bench.taxonomy import CATEGORIES, Example
from trainer.data import SPANISH_UPSAMPLE, app_labels, build_rows, text_key


def ex(i, text, source="toldbr", lang="pt", **labels):
    return Example(id=f"{source}:{i}", source=source, lang=lang, text=text, labels=labels)


def app(text, label, split="train", status="labeled"):
    return {"text": text, "label": label, "split": split, "status": status}


def test_text_key_ignores_case_and_spacing():
    assert text_key("Sos  un GIL ") == text_key("sos un gil")
    assert text_key("sos un gil") != text_key("sos una gil")


def test_an_app_example_in_the_suite_takes_the_suites_categories():
    suite = {text_key("te voy a romper la cara"): {"insult": False, "threat": True}}

    labels = app_labels(app("Te voy a romper la cara", "insult"), suite)

    assert labels == {"insult": False, "threat": True, "identity_hate": None, "sexual_harassment": None, "profanity": None}


def test_an_app_insult_outside_the_suite_only_says_insult():
    labels = app_labels(app("sos un gil", "insult"), {})

    assert labels == {"insult": True, "threat": None, "identity_hate": None, "sexual_harassment": None, "profanity": None}


def test_an_app_clean_example_rules_out_every_flag_category_but_not_profanity():
    labels = app_labels(app("qué día de mierda", "clean"), {})

    assert labels == {"insult": False, "threat": False, "identity_hate": False, "sexual_harassment": None, "profanity": None}


def benchmark():
    return [
        ex(1, "texto del benchmark", insult=True),
        ex(2, "outro texto", insult=False),
        Example(id="suite-es:es-1", source="suite-es", lang="es", text="callate gil", labels={"insult": True, "profanity": False}),
    ]


def told():
    return [
        ex(1, "texto del benchmark", insult=True),  # the benchmark sampled this one
        ex(5, "Texto  del BENCHMARK", insult=True),  # same text as a benchmark row
        ex(6, "um insulto", insult=True),
        ex(7, "neutro a", insult=False),
        ex(8, "neutro b", insult=False),
        ex(9, "neutro c", insult=False),
        ex(10, "sem rótulo"),  # nothing to score
    ]


def olid():
    return [ex(20, "ofensivo", source="olidbr", insult=True), ex(21, "outro texto", source="olidbr", insult=False)]


def build(app_examples=(), told_negatives=2):
    return build_rows(list(app_examples), benchmark(), told(), olid(), told_negatives=told_negatives)


def test_public_rows_never_repeat_a_benchmark_text():
    rows, stats = build()

    texts = {r["text"] for r in rows}
    assert "texto del benchmark" not in texts and "Texto  del BENCHMARK" not in texts and "outro texto" not in texts
    assert stats["dropped"] == 2  # the ToLD duplicate and the OLID duplicate; the sampled row never enters


def test_toldbr_keeps_every_positive_and_samples_negatives():
    rows, stats = build(told_negatives=2)

    told_rows = [r for r in rows if r["source"] == "toldbr"]
    assert sum(r["labels"]["insult"] is True for r in told_rows) == 1
    assert sum(r["labels"]["insult"] is False for r in told_rows) == 2
    assert all(r["text"] != "sem rótulo" for r in told_rows)
    assert stats["toldbr"] == 3 and stats["olidbr"] == 1


def test_spanish_train_rows_are_upsampled_and_other_splits_stay_out():
    examples = [app("callate gil", "insult"), app("gracias", "clean"), app("sos un tarado", "insult", split="test"),
                app("hola", None, status="pending")]

    rows, stats = build(examples)

    spanish = [r for r in rows if r["source"] == "app"]
    assert len(spanish) == 2 * SPANISH_UPSAMPLE
    assert {r["text"] for r in spanish} == {"callate gil", "gracias"}
    gil = next(r for r in spanish if r["text"] == "callate gil")
    assert gil["labels"]["profanity"] is False  # from the suite row with the same text
    assert gil["lang"] == "es"
    assert stats["spanish"] == 2
    assert stats["trained_texts"] == sorted({text_key("callate gil"), text_key("gracias")})


def test_every_row_has_every_category():
    rows, _ = build([app("gracias", "clean")])

    assert all(set(r["labels"]) == set(CATEGORIES) for r in rows)


def test_every_build_samples_the_same_rows_so_sentiment_labels_are_reused():
    first, second = build([app("gracias", "clean")]), build([app("gracias", "clean")])

    assert sorted(map(str, first[0])) == sorted(map(str, second[0]))


def test_an_unknown_app_label_is_an_error():
    with pytest.raises(ValueError, match="gil"):
        app_labels(app("gil", "maybe"), {})
