"""Sentiment scoring, and whether a new checkpoint should replace the active one."""

from __future__ import annotations

from app.model import SENTIMENTS

# A first checkpoint goes into use only if it separates harmful from harmless Spanish chat at least this
# well (flag AUROC on the suite rows it never saw).
MIN_SPANISH_AUROC = 0.75
# A better flag does not make up for a sentiment much worse than the active checkpoint's.
MAX_SENTIMENT_DROP = 0.05


def macro_f1(gold: list[str], predicted: list[str], labels=SENTIMENTS) -> float:
    f1s = []
    for label in labels:
        tp = sum(g == p == label for g, p in zip(gold, predicted))
        fp = sum(p == label != g for g, p in zip(gold, predicted))
        fn = sum(g == label != p for g, p in zip(gold, predicted))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s)


def activation_decision(new: dict, current: dict | None) -> tuple[bool, str]:
    """Both are {"auroc": Spanish flag AUROC, "sentiment_f1": Spanish sentiment macro-F1 or None}, measured on
    the same held-out rows. `current` is None when no checkpoint is active. The reason is in Spanish."""
    auroc = new["auroc"]
    if auroc is None:
        return False, "No hubo filas de la suite en español para evaluar el modelo nuevo."
    if current is None:
        if auroc < MIN_SPANISH_AUROC:
            return False, (f"El modelo nuevo separa poco el chat en español (AUROC {auroc:.2f}, mínimo "
                           f"{MIN_SPANISH_AUROC:.2f}).")
        return True, f"Es el primer modelo y separa bien el chat en español (AUROC {auroc:.2f})."
    if auroc <= current["auroc"]:
        return False, f"El modelo nuevo no separa mejor el chat en español (AUROC {current['auroc']:.2f} → {auroc:.2f})."
    old_f1, new_f1 = current.get("sentiment_f1"), new.get("sentiment_f1")
    if old_f1 is not None and new_f1 is not None and new_f1 < old_f1 - MAX_SENTIMENT_DROP:
        return False, (f"Separa mejor el chat (AUROC {current['auroc']:.2f} → {auroc:.2f}) pero empeora el "
                       f"sentimiento (macro-F1 {old_f1:.2f} → {new_f1:.2f}).")
    return True, f"Separa mejor el chat en español (AUROC {current['auroc']:.2f} → {auroc:.2f})."
