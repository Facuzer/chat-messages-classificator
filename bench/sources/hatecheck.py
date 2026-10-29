"""HateCheck (en) and Multilingual HateCheck (es, pt): functional tests of hate speech detection.

Each functionality only sets the categories it determines; the rest stay unknown. HateCheck's
"abuse at an individual" is an insult at a player under our policy, while abuse at non-protected
groups or objects only rules out identity hate.
"""

from __future__ import annotations

from bench.taxonomy import Example

HATECHECK_REPOS = {"es": "Paul/hatecheck-spanish", "pt": "Paul/hatecheck-portuguese", "en": "Paul/hatecheck"}
# The commits the preliminary scoreboard read (from the local HF cache on 2026-10-06).
HATECHECK_REVISIONS = {
    "es": "a7ea759535bb9fad6361cca151cf94a46e88edf3",
    "pt": "323bdf67e0fbd3d7f8086fad0971b5bd5a62524b",
    "en": "9d2ac89df04254e5c427bcc8d61b6d6c83a1f59b",
}
HATEFUL = (
    "derog_neg_emote_h", "derog_neg_attrib_h", "derog_dehum_h", "derog_impl_h", "threat_dir_h", "threat_norm_h",
    "slur_h", "profanity_h", "ref_subs_clause_h", "ref_subs_sent_h", "negate_pos_h", "phrase_question_h",
    "phrase_opinion_h", "spell_char_swap_h", "spell_char_del_h", "spell_space_del_h", "spell_space_add_h", "spell_leet_h",
)
NOT_HATEFUL = (
    "profanity_nh", "negate_neg_nh", "ident_neutral_nh", "ident_pos_nh", "counter_quote_nh", "counter_ref_nh",
    "target_obj_nh", "target_indiv_nh", "target_group_nh", "slur_homonym_nh", "slur_reclaimed_nh",
)
ALL_FUNCTIONALITIES = HATEFUL + NOT_HATEFUL
NO_ATTACK = {"insult": False, "threat": False, "identity_hate": False}


def _labels(functionality: str, hateful: bool) -> tuple[dict, str | None]:
    if hateful:
        labels = {"identity_hate": True}
        if functionality in ("threat_dir_h", "threat_norm_h"):
            labels["threat"] = True
        if functionality == "profanity_h":
            labels["profanity"] = True
        return labels, None
    if functionality == "profanity_nh":
        return {**NO_ATTACK, "profanity": True}, "none"
    if functionality in ("negate_neg_nh", "ident_neutral_nh", "ident_pos_nh"):
        return dict(NO_ATTACK), "none"
    if functionality == "target_obj_nh":
        return {"insult": False, "identity_hate": False}, None
    if functionality == "target_indiv_nh":
        return {"insult": True, "identity_hate": False}, "player"
    # Counter-speech can call someone a bigot, and abuse at non-protected groups is not ruled clean
    # by HateCheck, so only identity hate is known.
    return {"identity_hate": False}, None


def hatecheck_example(row: dict, lang: str) -> Example | None:
    if row.get("disagreement_in_case") in (True, "True", "true"):
        return None
    functionality, label = row["functionality"], row["label_gold"]
    if functionality not in ALL_FUNCTIONALITIES:
        raise ValueError(f"HateCheck functionality {functionality!r} has no mapping")
    if label not in ("hateful", "non-hateful"):
        raise ValueError(f"unexpected HateCheck label {label!r}")
    hateful = label == "hateful"
    if hateful != (functionality in HATEFUL):
        raise ValueError(f"{functionality!r} labeled {label!r}: the suffix says otherwise")
    labels, target = _labels(functionality, hateful)
    source = f"hatecheck-{lang}"
    case_id = row.get("mhc_case_id") or row["case_id"]
    return Example(
        id=f"{source}:{case_id}",
        source=source,
        lang=lang,
        text=row["test_case"].strip(),
        labels=labels,
        target=target,
        functionality=functionality,
        cluster=f"{source}:t{row['templ_id']}",
    )


def load_hatecheck(lang: str) -> list[Example]:
    from datasets import load_dataset

    rows = load_dataset(HATECHECK_REPOS[lang], split="test", revision=HATECHECK_REVISIONS[lang])
    return [e for e in (hatecheck_example(r, lang) for r in rows) if e is not None]
