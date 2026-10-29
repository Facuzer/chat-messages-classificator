"""Fine-tune mmBERT-base on the labeled examples and the public corpora, measure it on the benchmark, and
put it in use only if it beats the active checkpoint. See docs/finetuning.md.

    uv run python scripts/finetune.py
    uv run python scripts/finetune.py --seed 1 --no-activate

The GPU runs about 60% of the time (DUTY=0.6) to keep it cool. Restart the server afterwards so it loads
the active checkpoint.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.checkpoints import MODELS_DIR  # noqa: E402
from app.examples import DEFAULT_PATH, ExampleStore  # noqa: E402
from trainer.finetune import NotEnoughData, TrainConfig, run  # noqa: E402

SETS = {"suite-es-draft": "español (borrador sin revisar)", "dynasent-r2-en": "inglés (DynaSent)"}


def num(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def print_report(result: dict) -> None:
    m = result["metrics"]
    where = "del chat en español" if m["threshold_from"]["source"] == "suite-es" else "de todas las fuentes"
    print(f"\nUmbral de flag: {m['threshold']:.3f} (marca al 5% de {m['threshold_from']['negatives']} mensajes sanos "
          f"de calibración {where}).")
    print("\nAUROC de flag por fuente (0.50 es azar, 1.00 es perfecto), sin los textos de entrenamiento:")
    for source, value in m["flag_auroc"].items():
        print(f"  {source:<14}{num(value):>6}")
    op = m["operating_point"]
    print(f"\nCon ese umbral, en el chat en español detecta el {pct(op['recall'])} de los mensajes dañinos y "
          f"marca el {pct(op['fpr'])} de los sanos.")
    if m["sentiment_f1"]:
        print("Sentimiento (macro-F1): " + " · ".join(f"{SETS.get(k, k)} {v:.2f}" for k, v in m["sentiment_f1"].items()))
    else:
        print("Sentimiento: no se evaluó (faltan los archivos de data/sentiment/).")
    comparison = m["comparison"]
    if comparison["active"]:
        new, old = comparison["new"], comparison["active"]
        print(f"\nContra el modelo activo ({comparison['active_name']}), en las filas que ninguno de los dos vio: "
              f"AUROC {num(old['auroc'])} → {num(new['auroc'])}, sentimiento {num(old['sentiment_f1'])} → "
              f"{num(new['sentiment_f1'])}.")
    print(f"\n{result['reason']}")
    if result["activated"]:
        print(f"Activado: {result['checkpoint'].name}. Reiniciá el server para usarlo.")
    else:
        print(f"No se activó. El checkpoint quedó guardado en {result['checkpoint']}.")


def main() -> int:
    # The report contains "→"; a redirected stdout on Windows defaults to cp1252, which can't encode it.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--device", choices=["cuda", "cpu"], help="default: cuda when available")
    parser.add_argument("--no-activate", action="store_true", help="save the checkpoint without putting it in use")
    parser.add_argument("--skip-sentiment-labels", action="store_true",
                        help="train without labeling new texts' sentiment (they stay out of the sentiment loss)")
    args = parser.parse_args()

    try:
        result = run(
            ExampleStore(DEFAULT_PATH),
            MODELS_DIR,
            TrainConfig(seed=args.seed, epochs=args.epochs),
            activate=not args.no_activate,
            device=args.device,
            label_sentiment=not args.skip_sentiment_labels,
            log=lambda message: print(message, flush=True),
        )
    except NotEnoughData as e:
        print(e)
        return 1
    print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
