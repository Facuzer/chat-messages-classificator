"""Choose which fine-tuned checkpoint the app uses.

    uv run python scripts/activate_model.py --list
    uv run python scripts/activate_model.py models/ft-20261007-120000

Restart the server afterwards.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.checkpoints import MODELS_DIR, ActiveCheckpoint, checkpoint_problem  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("checkpoint", nargs="?", type=Path)
    group.add_argument("--list", action="store_true", help="show the fine-tuned checkpoints")
    args = parser.parse_args()
    active = ActiveCheckpoint(MODELS_DIR)

    if args.list:
        current = active.get()
        for meta_path in sorted(MODELS_DIR.glob("*/training_meta.json")):
            problem = checkpoint_problem(meta_path.parent)
            if problem and "incompleto" not in problem:
                continue  # not a checkpoint of this project
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            mark = "*" if current and current["path"] == meta_path.parent else " "
            if problem:
                print(f"{mark} {meta_path.parent.name}  incompleto")
                continue
            auroc = meta.get("metrics", {}).get("flag_auroc", {}).get("suite-es")
            print(f"{mark} {meta_path.parent.name}  AUROC chat ES "
                  f"{'n/a' if auroc is None else f'{auroc:.2f}'}  umbral {meta['threshold']:.3f}")
        print("  (* = activo)" if current else "  Ninguno activo: entrená uno con scripts/finetune.py.")
        return 0

    path = args.checkpoint.resolve()
    meta_path = path / "training_meta.json"
    if MODELS_DIR.resolve() not in path.parents:
        print(f"El checkpoint tiene que estar dentro de {MODELS_DIR}.")
        return 1
    problem = checkpoint_problem(path)
    if problem:
        print(f"{path.name} {problem}.")
        return 1
    active.activate(path, json.loads(meta_path.read_text(encoding="utf-8"))["metrics"])
    print(f"Listo: la app va a usar {path.name}. Reiniciá el server.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
