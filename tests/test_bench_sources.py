import pytest

from bench.sources import SOURCES, Source, load_sources, sample, split_of
from bench.taxonomy import Example


def ex(i, labels=None, cluster=None):
    return Example(id=f"s:{i}", source="s", lang="es", text=f"texto {i}",
                   labels={"insult": True} if labels is None else labels, cluster=cluster)


def test_sample_drops_examples_with_nothing_to_score_and_is_stable():
    examples = [ex(i) for i in range(50)] + [ex(f"x{i}", labels={}) for i in range(10)]

    picked = sample(examples, 20)

    assert len(picked) == 20 and all(e.labels for e in picked)
    assert [e.id for e in sample(list(reversed(examples)), 20)] == [e.id for e in picked]
    assert len(sample(examples, 1000)) == 50


def test_a_fifth_goes_to_calibration_and_clusters_stay_together():
    splits = [split_of(ex(i)) for i in range(5000)]

    assert 0.17 < splits.count("calib") / 5000 < 0.23
    assert len({split_of(ex(i, cluster="tmpl-1")) for i in range(30)}) == 1


def test_only_sampled_sources_are_capped_and_an_empty_source_warns():
    logs = []
    sources = {
        "big": Source("big", "es", "x", lambda _: [ex(i) for i in range(30)], sampled=True),
        "whole": Source("whole", "es", "x", lambda _: [ex(f"w{i}") for i in range(30)]),
        "empty": Source("empty", "es", "x", lambda _: [ex("e", labels={})]),
    }

    examples = load_sources(["big", "whole", "empty"], max_per_source=10, sources=sources, log=logs.append)

    assert len(examples) == 40
    assert any("empty" in line for line in logs)


def test_include_unreviewed_reaches_the_loader():
    seen = []
    sources = {"suite": Source("suite", "es", "x", lambda unreviewed: seen.append(unreviewed) or [ex(1)])}

    load_sources(["suite"], include_unreviewed=True, sources=sources, log=lambda _: None)

    assert seen == [True]


def test_unknown_source_is_an_error():
    with pytest.raises(ValueError, match="desconocida"):
        load_sources(["nope"], sources={}, log=lambda _: None)


def test_registry_has_the_planned_sources():
    assert set(SOURCES) == {"suite-es", "hatecheck-es", "hatecheck-pt", "hatecheck-en", "toldbr", "olidbr"}
    assert [n for n, s in SOURCES.items() if s.sampled] == ["toldbr"]
