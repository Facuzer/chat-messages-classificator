from bench.scoreboard import build_scoreboard, functionality_label, slice_keys
from bench.taxonomy import FLAG, Example


def split(example):
    return "calib" if example.id.split(":")[1].startswith("c") else "eval"


def ex(i, insult, source="suite-es", author=None, functionality=None):
    return Example(id=f"{source}:{i}", source=source, lang="es", text=f"t{i}",
                   labels={"insult": insult, "threat": False, "identity_hate": False},
                   author=author, functionality=functionality)


class Perfect:
    name = "perfect"
    version = "1"
    notes = ""
    languages = frozenset({"es"})

    def to_scores(self, raw):
        return {"insult": raw["p"], "threat": 0.0, "identity_hate": 0.0}


class Coin(Perfect):
    name = "coin"


class OnlyHate(Perfect):
    name = "only-hate"

    def to_scores(self, raw):
        return {"identity_hate": raw["p"]}


def dataset():
    """80 eval examples (half insults, half by human authors) and 20 calibration ones."""
    examples, perfect, coin = [], {}, {}
    for i in range(80):
        e = ex(f"e{i}", i % 2 == 0, author="human" if i % 4 < 2 else "generated", functionality="insult_direct")
        examples.append(e)
        perfect[e.id], coin[e.id] = {"p": 0.9 if i % 2 == 0 else 0.05}, {"p": 0.5}
    for i in range(20):
        e = ex(f"c{i}", i % 2 == 0)
        examples.append(e)
        perfect[e.id], coin[e.id] = {"p": 0.9 if i % 2 == 0 else 0.01 * i}, {"p": 0.5}
    return examples, {"perfect": perfect, "coin": coin, "only-hate": perfect}


def board():
    examples, raw = dataset()
    return build_scoreboard(examples, [Perfect(), Coin(), OnlyHate()], raw, n_boot=20, split=split)


def row(b, contestant, source, label):
    return next((r for r in b["slices"] if (r["contestant"], r["source"], r["label"]) == (contestant, source, label)), None)


def test_thresholds_come_from_the_calibration_split_only():
    # Calibration negatives score 0.01..0.19; with the 40 eval negatives (0.05) mixed in it would be 0.15.
    assert board()["thresholds"]["perfect"]["insult"] == 0.19


def test_a_perfect_contestant_gets_auroc_one_and_full_recall_at_its_operating_point():
    r = row(board(), "perfect", "suite-es", FLAG)

    assert (r["n"], r["positives"], r["auroc"], r["enough"]) == (80, 40, 1.0, True)
    assert (r["recall_at_op"], r["fpr_at_op"]) == (1.0, 0.0)


def test_suite_rows_are_also_reported_by_author_and_small_slices_are_marked():
    b = board()

    human = row(b, "perfect", "suite-es/human", FLAG)
    assert (human["n"], human["enough"]) == (40, False)
    assert row(b, "perfect", "suite-es/generated", FLAG) is not None


def test_a_single_class_label_has_no_auroc():
    r = row(board(), "perfect", "suite-es", "threat")

    assert r["positives"] == 0 and r["auroc"] is None and r["enough"] is False


def test_a_contestant_without_every_core_category_gets_no_flag_rows():
    b = board()

    assert row(b, "only-hate", "suite-es", FLAG) is None
    assert row(b, "only-hate", "suite-es", "identity_hate") is not None


def test_coverage_counts_the_eval_split():
    cov = {(c["source"], c["label"]): c for c in board()["coverage"]}

    assert (cov[("suite-es", "insult")]["positives"], cov[("suite-es", "insult")]["negatives"]) == (40, 40)
    assert cov[("suite-es/human", FLAG)]["positives"] == 20


def test_comparisons_measure_each_contestant_against_the_best():
    [c] = [c for c in board()["comparisons"] if c["source"] == "suite-es"]

    assert (c["best"], c["contestant"], c["diff"]) == ("perfect", "coin", -0.5)
    assert c["hi"] < 0


def test_comparisons_skip_sources_with_too_few_examples_of_a_class():
    # suite-es has 40/40 eval `flag` examples; each author slice has 20/20, under MIN_PER_CLASS.
    sources = {c["source"] for c in board()["comparisons"]}

    assert sources == {"suite-es"}


def test_functionality_accuracy_uses_the_operating_threshold():
    rows = {(r["contestant"], r["functionality"]): r for r in board()["functionalities"]}

    assert rows[("perfect", "insult_direct")] == {"contestant": "perfect", "source": "suite-es",
                                                  "functionality": "insult_direct", "n": 80, "accuracy": 1.0}
    assert functionality_label("hatecheck-pt", "slur_h") == "identity_hate"
    assert functionality_label("suite-es", "insult_direct") == FLAG
    assert functionality_label("suite-es", "sexual_harassment") == "sexual_harassment"


def test_slice_keys_add_the_author_when_there_is_one():
    assert slice_keys(ex("e1", True, author="human")) == ["suite-es", "suite-es/human"]
    assert slice_keys(ex("e1", True)) == ["suite-es"]
