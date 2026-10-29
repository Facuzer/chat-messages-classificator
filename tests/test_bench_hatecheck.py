import re

import pytest

from bench.sources.hatecheck import ALL_FUNCTIONALITIES, hatecheck_example
from bench.taxonomy import FLAG, TARGETS_PLAYER, gold


def row(functionality, label=None, **extra):
    label = label or ("hateful" if functionality.endswith("_h") else "non-hateful")
    return {"mhc_case_id": "spanish-1", "functionality": functionality, "test_case": " Odio a los negros. ",
            "label_gold": label, "templ_id": 7, "disagreement_in_case": False, **extra}


def test_hateful_case_is_identity_hate_and_leaves_other_categories_unknown():
    e = hatecheck_example(row("derog_neg_emote_h"), "es")

    assert (e.id, e.source, e.lang, e.text) == ("hatecheck-es:spanish-1", "hatecheck-es", "es", "Odio a los negros.")
    assert e.labels == {"identity_hate": True}
    assert gold(e, FLAG) is True and gold(e, "insult") is None
    assert (e.functionality, e.cluster) == ("derog_neg_emote_h", "hatecheck-es:t7")


def test_threat_and_profanity_functionalities_set_those_categories_too():
    assert hatecheck_example(row("threat_dir_h"), "es").labels["threat"] is True
    assert hatecheck_example(row("profanity_h"), "es").labels["profanity"] is True


def test_untargeted_profanity_is_a_flag_negative_with_profanity():
    e = hatecheck_example(row("profanity_nh"), "es")

    assert gold(e, FLAG) is False and gold(e, "profanity") is True and e.target == "none"


def test_abuse_at_an_individual_is_an_insult_at_a_player():
    e = hatecheck_example(row("target_indiv_nh"), "es")

    assert gold(e, FLAG) is True and gold(e, TARGETS_PLAYER) is True and gold(e, "identity_hate") is False


@pytest.mark.parametrize("functionality", ["counter_quote_nh", "counter_ref_nh", "target_group_nh", "slur_reclaimed_nh"])
def test_non_hateful_cases_that_only_rule_out_hate_leave_flag_unknown(functionality):
    e = hatecheck_example(row(functionality), "es")

    assert gold(e, "identity_hate") is False and gold(e, FLAG) is None


def test_negated_hate_is_clean():
    assert gold(hatecheck_example(row("negate_neg_nh"), "es"), FLAG) is False


@pytest.mark.parametrize("value", [True, "True", "true"])
def test_cases_annotators_disagreed_on_are_dropped(value):
    assert hatecheck_example(row("slur_h", disagreement_in_case=value), "es") is None


def test_english_rows_use_case_id():
    english = {"case_id": 12, "functionality": "slur_h", "test_case": "I hate women. ", "label_gold": "hateful", "templ_id": 3}

    e = hatecheck_example(english, "en")

    assert (e.id, e.text) == ("hatecheck-en:12", "I hate women.")


def test_unknown_functionality_or_label_raises_naming_it():
    with pytest.raises(ValueError, match="new_thing_nh"):
        hatecheck_example(row("new_thing_nh"), "es")
    with pytest.raises(ValueError, match="hateful"):
        hatecheck_example(row("slur_h", label="non-hateful"), "es")
    with pytest.raises(ValueError, match="maybe"):
        hatecheck_example(row("slur_h", label="maybe"), "es")


@pytest.mark.parametrize("functionality", ALL_FUNCTIONALITIES)
def test_every_known_functionality_maps(functionality):
    assert hatecheck_example(row(functionality), "en") is not None


def test_every_hatecheck_dataset_loads_at_its_pinned_revision(monkeypatch):
    import datasets

    import bench.sources.hatecheck as hatecheck

    seen = []
    monkeypatch.setattr(datasets, "load_dataset", lambda repo, **kwargs: seen.append((repo, kwargs)) or [])

    for lang in ("es", "pt", "en"):
        hatecheck.load_hatecheck(lang)

    assert [repo for repo, _ in seen] == ["Paul/hatecheck-spanish", "Paul/hatecheck-portuguese", "Paul/hatecheck"]
    assert [kwargs["revision"] for _, kwargs in seen] == [hatecheck.HATECHECK_REVISIONS[lang] for lang in ("es", "pt", "en")]
    assert all(re.fullmatch(r"[0-9a-f]{40}", kwargs["revision"]) for _, kwargs in seen)
