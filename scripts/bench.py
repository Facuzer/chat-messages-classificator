"""Compare the fine-tuned checkpoints on the moderation exam. See docs/benchmark.md.

    uv run python scripts/bench.py sources
    uv run python scripts/bench.py run [--include-unreviewed] [--checkpoints ft-20261007-184309,ft-20261008-120000] [--device cuda]
    uv run python scripts/bench.py latency --device cpu --threads 4
    uv run python scripts/bench.py agreement suites/es_casino.csv copia.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
