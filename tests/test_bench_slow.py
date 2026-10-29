"""Benchmark checks against real datasets and checkpoints. They download data, so they only run with
`uv run pytest -m slow`."""

from collections import Counter

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("lang,minimum", [("es", 3500), ("pt", 3400), ("en", 3700)])
def test_real_hatecheck_loads_and_maps_every_case(lang, minimum):
    from bench.sources.hatecheck import load_hatecheck

    examples = load_hatecheck(lang)

    assert len(examples) >= minimum
    assert len(Counter(e.functionality for e in examples)) >= 27


def test_real_toldbr_and_olidbr_load_with_both_flag_classes():
    from bench.sources.portuguese import load_olidbr, load_toldbr
    from bench.taxonomy import FLAG, gold

    told, olid = load_toldbr(), load_olidbr()

    assert len(told) == 21000 and len(olid) == 1738
    for examples in (told, olid):
        flags = Counter(gold(e, FLAG) for e in examples)
        assert flags[True] > 100 and flags[False] > 100, flags


def test_every_real_checkpoint_scores_a_spanish_insult_above_a_greeting():
    from bench.contestants import checkpoint_factories, effective_scores

    factories = checkpoint_factories()
    if not factories:
        pytest.skip("no hay checkpoints entrenados: corré scripts/finetune.py")
    for name, factory in factories.items():
        contestant = factory("cpu")
        insult, greeting = (effective_scores(contestant, r)["flag"]
                            for r in contestant.predict(["sos un pelotudo de mierda", "hola, suerte a todos"], "es"))
        contestant.unload()

        assert insult > greeting, name
