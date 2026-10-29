"""Run sample messages through the active fine-tuned model and print each verdict.

Use it to try FLAG_THRESHOLD values or to look at hard cases without the UI.

    uv run python scripts/try_model.py
    uv run python scripts/try_model.py "otra frase" "another one"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classifier import load_classifier  # noqa: E402

SAMPLES = [
    "Sos un idiota, no servís para nada.",
    "No sos un idiota, tranquilo.",
    "sos un pelotudo",
    "boludo, vení que arrancamos",
    "Qué día de mierda.",
    "Te voy a romper la cara.",
    "all in 500",
    "La app anda lentísima y estoy re decepcionado.",
    "¡Gracias! Me encantó cómo quedó.",
    "You're a complete moron.",
    "This is great, thanks a lot!",
]


def main() -> None:
    texts = sys.argv[1:] or SAMPLES
    classifier = load_classifier()
    print(f"modelo={classifier.model} umbral={classifier.threshold:.3f}\n")
    print(f"{'flag':<5} {'score':>5} {'categoría':<13} {'puteada':>7} {'sentimiento':<11} {'conf':>5} {'ms':>4}  texto")
    for text in texts:
        c = classifier.classify(text)
        print(
            f"{'SÍ' if c.flagged else '-':<5} {c.flag.score:>5.2f} {c.flag.category:<13} {c.categories['profanity']:>7.2f} "
            f"{c.sentiment.label:<11} {c.sentiment.confidence:>5.2f} {c.latency_ms:>4}  {text}"
        )


if __name__ == "__main__":
    main()
