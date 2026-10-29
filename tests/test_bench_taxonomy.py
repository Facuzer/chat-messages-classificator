import pytest

from bench.taxonomy import FLAG, TARGETS_PLAYER, Example, gold, has_gold


def ex(labels, target=None, **kwargs):
    return Example(id="t:1", source="t", lang="es", text="hola", labels=labels, target=target, **kwargs)


def test_flag_is_true_when_any_core_category_is_true_even_if_others_are_unknown():
    assert gold(ex({"insult": True}), FLAG) is True


def test_flag_is_false_only_when_insult_threat_and_identity_hate_are_all_known_false():
    assert gold(ex({"insult": False, "threat": False, "identity_hate": False}), FLAG) is False


def test_flag_is_unknown_when_a_core_category_is_unknown_and_none_is_true():
    assert gold(ex({"insult": False, "identity_hate": False}), FLAG) is None


def test_profanity_and_sexual_harassment_do_not_make_a_message_flagged():
    example = ex({"insult": False, "threat": False, "identity_hate": False, "profanity": True, "sexual_harassment": True})

    assert gold(example, FLAG) is False
    assert gold(example, "sexual_harassment") is True


def test_targets_player_is_only_scored_on_flagged_messages():
    assert gold(ex({"insult": True}, target="player"), TARGETS_PLAYER) is True
    assert gold(ex({"identity_hate": True}, target="group"), TARGETS_PLAYER) is False
    assert gold(ex({"insult": False, "threat": False, "identity_hate": False}, target="none"), TARGETS_PLAYER) is None
    assert gold(ex({"insult": True}), TARGETS_PLAYER) is None


def test_a_category_the_source_does_not_annotate_is_unknown():
    assert gold(ex({}), "threat") is None
    assert has_gold(ex({})) is False
    assert has_gold(ex({"profanity": False})) is True


@pytest.mark.parametrize(
    "override",
    [
        {"lang": "fr"},
        {"labels": {"spam": True}},
        {"labels": {"insult": 1}},
        {"target": "everyone"},
        {"text": "   "},
    ],
)
def test_example_rejects_invalid_fields(override):
    fields = {"id": "t:1", "source": "t", "lang": "es", "text": "hola", "labels": {}}

    with pytest.raises(ValueError):
        Example(**{**fields, **override})


def test_cluster_defaults_to_the_example_id():
    assert ex({}).cluster_key == "t:1"
    assert ex({}, cluster="tmpl-3").cluster_key == "tmpl-3"
