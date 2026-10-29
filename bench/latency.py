"""How long a contestant takes per message, one at a time and in batches."""

from __future__ import annotations

import math
import time


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def measure(contestant, texts: list[str], lang: str, batch_size: int = 32, clock=time.perf_counter) -> dict:
    contestant.predict(texts[:1], lang)  # loads the model; not timed
    per_message = []
    for text in texts:
        start = clock()
        contestant.predict([text], lang)
        per_message.append((clock() - start) * 1000)
    start = clock()
    for i in range(0, len(texts), batch_size):
        contestant.predict(texts[i:i + batch_size], lang)
    total = clock() - start
    return {
        "n": len(texts),
        "p50_ms": percentile(per_message, 50),
        "p95_ms": percentile(per_message, 95),
        "throughput_per_s": len(texts) / total if total > 0 else None,
        "batch_size": batch_size,
    }
