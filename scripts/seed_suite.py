"""Write the first draft of suites/es_casino.csv for review.

    uv run python scripts/seed_suite.py           # refuses to overwrite a suite that may hold reviews
    uv run python scripts/seed_suite.py --force
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.seed import build_suite, read_jsonl, write_new_suite  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="pisar la suite aunque ya exista")
    args = parser.parse_args()

    rows = build_suite(
        read_jsonl(ROOT / "datasets" / "generated_rioplatense_v1.jsonl"),
        read_jsonl(ROOT / "suites" / "drafts" / "es_casino_new.jsonl"),
    )
    try:
        write_new_suite(ROOT / "suites" / "es_casino.csv", rows, force=args.force)
    except FileExistsError as e:
        print(e)
        return 1
    print(f"Suite escrita con {len(rows)} filas, todas pendientes de revisión:")
    for functionality, n in sorted(Counter(r["functionality"] for r in rows).items()):
        print(f"  {functionality:<28} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
