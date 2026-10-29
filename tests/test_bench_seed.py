from collections import Counter
from pathlib import Path

import pytest

from bench.seed import build_suite, draft_rows, read_jsonl, seed_rows, write_new_suite
from bench.sources.suite import load_suite, parse_suite

ROOT = Path(__file__).resolve().parent.parent
DRAFT_COUNTS = {
    "gaming_insult": 25, "bets_amounts": 30, "obfuscated_insult": 25, "identity_hate_casino": 25,
    "obfuscated_slur": 15, "affectionate_identity_term": 10, "sexual_harassment": 20, "threat_doxxing": 15,
    "quoting_reporting_casino": 10, "self_deprecation": 10, "profanity_at_game": 15, "code_mixing_insult": 10,
    "code_mixing_banter": 10, "counter_speech": 10,
}


def test_seed_sets_only_what_the_category_implies():
    [row] = seed_rows([{"text": "callate gil", "label": "insult", "category": "insulto directo"}])

    assert (row["functionality"], row["insult"], row["target"]) == ("insult_direct", "1", "player")
    assert row["threat"] == row["identity_hate"] == row["sexual_harassment"] == row["profanity"] == ""
    assert (row["lang"], row["author"], row["reviewed"]) == ("es", "generated", "0")


def test_everyday_messages_are_seeded_clean_in_every_category():
    [row] = seed_rows([{"text": "mañana a las 10", "label": "clean", "category": "cotidiano"}])

    assert [row[c] for c in ("insult", "threat", "identity_hate", "sexual_harassment", "profanity")] == ["0"] * 5


def test_unknown_seed_category_is_an_error():
    with pytest.raises(ValueError, match="categoría"):
        seed_rows([{"text": "x", "label": "clean", "category": "spam"}])


def test_drafts_keep_their_labels_and_reject_bad_targets():
    [row] = draft_rows([{"functionality": "gaming_insult", "text": "sos un manco", "insult": 1, "target": "player"}])

    assert (row["insult"], row["threat"], row["target"]) == ("1", "", "player")
    with pytest.raises(ValueError):
        draft_rows([{"functionality": "x", "text": "y", "target": "todos"}])


def test_build_assigns_sequential_ids_and_rejects_duplicate_texts():
    rows = build_suite([{"text": "Sos un gil", "label": "insult", "category": "insulto directo"}],
                       [{"functionality": "gaming_insult", "text": "manco", "insult": "1", "target": "player"}])

    assert [r["id"] for r in rows] == ["es-0001", "es-0002"]
    with pytest.raises(ValueError, match="repetido"):
        build_suite([{"text": "Sos un gil", "label": "insult", "category": "insulto directo"}],
                    [{"functionality": "gaming_insult", "text": "sos  un GIL", "insult": "1"}])


def test_refuses_to_overwrite_a_suite_that_may_hold_reviews(tmp_path):
    path = tmp_path / "suite.csv"
    rows = build_suite([{"text": "hola", "label": "clean", "category": "cotidiano"}], [])
    write_new_suite(path, rows)

    with pytest.raises(FileExistsError):
        write_new_suite(path, rows)
    write_new_suite(path, rows, force=True)
    assert [e.text for e, _ in parse_suite(path.read_text(encoding="utf-8-sig"), "s")] == ["hola"]


def test_drafts_have_the_planned_counts_and_the_committed_suite_keeps_them():
    drafts = read_jsonl(ROOT / "suites" / "drafts" / "es_casino_new.jsonl")
    examples = load_suite(ROOT / "suites" / "es_casino.csv", "suite-es", include_unreviewed=True)

    assert Counter(d["functionality"] for d in drafts) == DRAFT_COUNTS
    assert set(DRAFT_COUNTS) <= {e.functionality for e in examples}
