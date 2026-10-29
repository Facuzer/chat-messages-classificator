import re

import pytest

from bench.sources.portuguese import olidbr_example, toldbr_example
from bench.taxonomy import FLAG, TARGETS_PLAYER, gold

TOLD_ZERO = {"text": " texto ", "homophobia": 0, "obscene": 0, "insult": 0, "racism": 0, "misogyny": 0, "xenophobia": 0}
OLID_FALSE = {c: False for c in ("health", "ideology", "insult", "lgbtqphobia", "other_lifestyle", "physical_aspects",
                                   "profanity_obscene", "racism", "religious_intolerance", "sexism", "xenophobia")}


def told(**votes):
    return toldbr_example({**TOLD_ZERO, **votes}, 5)


def olid(**fields):
    base = {"id": "ab12", "text": "texto", "is_offensive": "OFF", "is_targeted": "TIN", "targeted_type": "IND", **OLID_FALSE}
    return olidbr_example({**base, **fields})


def test_told_without_votes_is_clean():
    e = told()

    assert (e.id, e.lang, e.text) == ("toldbr:5", "pt", "texto")
    assert gold(e, FLAG) is False and gold(e, "profanity") is False and e.target == "none"


def test_told_obscene_only_is_profanity_but_not_flagged():
    e = told(obscene="3.0")

    assert gold(e, FLAG) is False and gold(e, "profanity") is True


def test_told_majority_insult_is_flagged_and_a_single_vote_is_unknown():
    assert gold(told(insult=2), FLAG) is True
    single = told(insult=1)
    assert gold(single, "insult") is None and gold(single, FLAG) is None


def test_told_identity_columns_combine():
    assert gold(told(racism=2), "identity_hate") is True
    assert gold(told(racism=1), "identity_hate") is None
    assert gold(told(insult=2), "identity_hate") is False


def test_told_vote_out_of_range_raises():
    with pytest.raises(ValueError, match="7"):
        told(insult=7)


def test_olid_not_offensive_is_clean():
    e = olid(is_offensive="NOT", is_targeted="UNT", targeted_type=None)

    assert gold(e, FLAG) is False and gold(e, "profanity") is False and e.id == "olidbr:ab12"


def test_olid_untargeted_swearing_only_is_a_flag_negative():
    e = olid(is_targeted="UNT", targeted_type=None, profanity_obscene=True)

    assert gold(e, FLAG) is False and gold(e, "profanity") is True


def test_olid_insult_at_an_individual_targets_a_player():
    e = olid(insult=True)

    assert gold(e, FLAG) is True and gold(e, TARGETS_PLAYER) is True


def test_olid_group_insult_is_not_a_personal_insult():
    e = olid(targeted_type="GRP", insult=True, racism=True)

    assert gold(e, "insult") is None and gold(e, "identity_hate") is True and e.target == "group"


def test_olid_health_is_ambiguous_for_identity_hate():
    assert gold(olid(health=True), "identity_hate") is None


def test_olid_insult_column_false_leaves_insult_unknown():
    # A body, lifestyle, ideology or health attack is an insult to us even when OLID's insult column is False.
    e = olid(physical_aspects=True)

    assert gold(e, "insult") is None and gold(e, FLAG) is None and e.target == "player"


def test_olid_attack_on_an_organization_or_thing_has_no_insult_or_target():
    e = olid(targeted_type="OTH", insult=True)

    assert gold(e, "insult") is None and e.target is None


def test_olid_untargeted_attack_has_no_insult_or_target():
    e = olid(is_targeted="UNT", targeted_type=None, insult=True)

    assert gold(e, "insult") is None and e.target is None and gold(e, FLAG) is None


def test_olid_untargeted_identity_attack_is_still_flagged():
    e = olid(is_targeted="UNT", targeted_type=None, insult=True, racism=True)

    assert gold(e, "identity_hate") is True and gold(e, FLAG) is True and gold(e, TARGETS_PLAYER) is None


@pytest.mark.parametrize("field,value", [("is_offensive", "MAYBE"), ("is_targeted", "X"), ("targeted_type", "ORG")])
def test_olid_unexpected_values_raise(field, value):
    with pytest.raises(ValueError, match=value):
        olid(**{field: value})


def test_toldbr_and_olidbr_load_at_their_pinned_revisions(monkeypatch):
    import datasets

    import bench.sources.portuguese as portuguese

    seen = []
    monkeypatch.setattr(datasets, "load_dataset", lambda repo, **kwargs: seen.append((repo, kwargs)) or [])

    portuguese.load_toldbr()
    portuguese.load_olidbr()

    assert seen == [("mteb/told-br", {"split": "train", "revision": portuguese.TOLDBR_REVISION}),
                    ("dougtrajano/olid-br", {"split": "test", "revision": portuguese.OLIDBR_REVISION})]
    assert all(re.fullmatch(r"[0-9a-f]{40}", kwargs["revision"]) for _, kwargs in seen)
