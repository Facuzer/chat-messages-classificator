import pytest

from bench.sources.suite import COLUMNS, SuiteError, agreement, load_suite, parse_suite, write_suite

ROW = ["es-0001", "es", "insult_direct", "generated", "sos un manco", "1", "0", "0", "", "?", "player", "1", ""]


def csv_text(*rows, delimiter=";"):
    return "\n".join([delimiter.join(COLUMNS), *(delimiter.join(r) for r in rows)]) + "\n"


def with_cell(column, value, row=ROW):
    changed = list(row)
    changed[COLUMNS.index(column)] = value
    return changed


def test_parses_a_reviewed_row():
    [(example, reviewed)] = parse_suite(csv_text(ROW), "suite-es")

    assert example.id == "suite-es:es-0001"
    assert example.labels == {
        "insult": True, "threat": False, "identity_hate": False, "sexual_harassment": None, "profanity": None,
    }
    assert (example.target, example.author, example.functionality) == ("player", "generated", "insult_direct")
    assert reviewed is True


def test_excel_semicolons_with_bom_parse_like_plain_commas():
    assert parse_suite("\ufeff" + csv_text(ROW), "s") == parse_suite(csv_text(ROW, delimiter=","), "s")


def test_empty_rows_excel_leaves_at_the_end_are_ignored():
    text = csv_text(ROW) + ";" * (len(COLUMNS) - 1) + "\n\n"

    assert len(parse_suite(text, "s")) == 1


def test_bad_cell_names_its_line_and_column():
    with pytest.raises(SuiteError, match=r"línea 2.*insult"):
        parse_suite(csv_text(with_cell("insult", "si")), "s")


def test_duplicate_ids_are_rejected():
    with pytest.raises(SuiteError, match="repetido"):
        parse_suite(csv_text(ROW, ROW), "s")


@pytest.mark.parametrize("column,value", [("lang", "fr"), ("author", "bot"), ("target", "todos"), ("reviewed", "sí"), ("text", "")])
def test_invalid_values_are_rejected_with_the_line(column, value):
    with pytest.raises(SuiteError, match="línea 2"):
        parse_suite(csv_text(with_cell(column, value)), "s")


def test_missing_column_is_reported():
    with pytest.raises(SuiteError, match="faltan columnas"):
        parse_suite("id;text\nes-1;hola\n", "s")


def test_load_keeps_only_reviewed_rows_unless_asked(tmp_path):
    path = tmp_path / "suite.csv"
    path.write_text(csv_text(ROW, with_cell("reviewed", "0", with_cell("id", "es-0002"))), encoding="utf-8")

    assert [e.id for e in load_suite(path, "s")] == ["s:es-0001"]
    assert len(load_suite(path, "s", include_unreviewed=True)) == 2


def test_written_suite_round_trips_with_bom_semicolons_and_awkward_text(tmp_path):
    path = tmp_path / "suite.csv"
    row = dict(zip(COLUMNS, ROW)) | {"text": 'dale; vení,\n"ya"'}

    write_suite(path, [row])

    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8-sig").splitlines()[0] == ";".join(COLUMNS)
    [example] = load_suite(path, "s")
    assert example.text == 'dale; vení,\n"ya"'


def test_agreement_is_cohens_kappa_per_category():
    rows_a = [with_cell("insult", v, with_cell("id", f"es-{i}")) for i, v in enumerate(["1", "0", "1", "0"])]
    rows_b = [with_cell("insult", v, with_cell("id", f"es-{i}")) for i, v in enumerate(["1", "0", "0", "0"])]
    a = [e for e, _ in parse_suite(csv_text(*rows_a), "s")]
    b = [e for e, _ in parse_suite(csv_text(*rows_b), "s")]

    assert agreement(a, a)["insult"] == {"n": 4, "kappa": 1.0}
    assert agreement(a, b)["insult"] == {"n": 4, "kappa": 0.5}
    assert agreement(a, b)["profanity"] == {"n": 0, "kappa": None}


def test_a_suite_saved_as_ansi_by_excel_asks_for_utf8(tmp_path):
    path = tmp_path / "suite.csv"
    path.write_bytes(csv_text(with_cell("text", "vení, manco")).encode("cp1252"))

    with pytest.raises(SuiteError, match="UTF-8"):
        load_suite(path, "s")
