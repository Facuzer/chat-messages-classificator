"""Keep the GPU busy only part of the time, so it runs cooler.

After each unit of GPU work, sleep in proportion to how long it took: DUTY=0.6 (the default, from the
environment) means about 60% busy, and DUTY=1 never sleeps.
"""

from __future__ import annotations

import os
import time

DUTY = float(os.environ.get("DUTY", "0.6"))
# Pause between two long GPU jobs in one run, such as labeling and training.
COOLDOWN_S = float(os.environ.get("COOLDOWN_S", "120"))


def _cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def pause(started: float, duty: float | None = None, clock=time.perf_counter, sleep=time.sleep, sync=_cuda_sync) -> None:
    """Sleep after work that began at `started` (a `clock` reading). CUDA runs asynchronously, so the
    queued work is waited for before measuring it."""
    duty = DUTY if duty is None else duty
    if duty >= 1:
        return
    sync()
    sleep((clock() - started) * (1 / duty - 1))


def throttled(contestant):
    """The contestant, with a pause after each `predict`."""
    predict = contestant.predict

    def predict_then_pause(texts, lang):
        started = time.perf_counter()
        out = predict(texts, lang)
        pause(started)
        return out

    contestant.predict = predict_then_pause
    return contestant
