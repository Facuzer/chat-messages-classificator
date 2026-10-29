"""Fine-tune mmBERT-base on the moderation categories and the sentiment, measure it on the benchmark, and
put it in use only if it beats the active checkpoint.

1. Training set (trainer.data): ToLD-Br and OLID-BR rows that repeat no benchmark text, plus the app's
   labeled Spanish examples.
2. Sentiment (trainer.sentiment): Qwen3-4B labels the texts that have no label yet, once.
3. Training: one encoder with eight outputs (app.model.LABELS), in one forward pass. Binary cross-entropy
   on the five categories, where an unknown label counts for nothing, plus soft cross-entropy on the
   sentiment.
4. Threshold: `flag` at 5% false positives on the Spanish calibration rows (trainer.evaluation).
5. Evaluation on the benchmark's eval split, without the texts the compared checkpoints trained on.
6. Activation (trainer.metrics.activation_decision).

The checkpoint trains on a GPU when there is one and is saved in fp32, so it runs on a CPU.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.checkpoints import ActiveCheckpoint, checkpoint_problem
from app.examples import ExampleStore, text_key
from app.jsonfile import write_json_atomic
from app.model import LABELS, SENTIMENTS
from bench.metrics import auroc
from bench.sources import split_of
from bench.taxonomy import CATEGORIES, FLAG, gold
from trainer.data import build_rows, load_inputs
from trainer.evaluation import (
    SENTIMENT_DIR, SPANISH_SOURCE, choose_threshold, flag_auroc_by_source, load_sentiment_sets, operating_point,
    sentiment_f1,
)
from trainer.metrics import activation_decision
from trainer.sentiment import SentimentLabels, label_missing, sentiment_labeler
from trainer.throttle import COOLDOWN_S, pause, throttled

# A pinned Hub commit. mmBERT ships only `pytorch_model.bin`, which transformers reads with
# torch.load(weights_only=True): no code runs.
BASE_MODEL = ("jhu-clsp/mmBERT-base", "c5955035435e2bf121cde7f3c8863ef52ff35d82")
MIN_TRAIN_PER_CLASS = 20
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "bench_out" / "predictions"
# A threshold of last resort, for a benchmark without calibration rows.
FALLBACK_THRESHOLD = 0.5


class NotEnoughData(Exception):
    pass


@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 32
    lr: float = 4e-5
    weight_decay: float = 0.01
    warmup: float = 0.06
    max_length: int = 128
    seed: int = 0


def _check_enough(app_examples: list[dict]) -> None:
    train = [e for e in app_examples if e["status"] == "labeled" and e["split"] == "train"]
    counts = {label: sum(e["label"] == label for e in train) for label in ("insult", "clean")}
    few = [f"{label} ({n})" for label, n in counts.items() if n < MIN_TRAIN_PER_CLASS]
    if few:
        raise NotEnoughData(
            f"Hacen falta al menos {MIN_TRAIN_PER_CLASS} ejemplos de entrenamiento de cada tipo; "
            f"faltan de: {', '.join(few)}. Etiquetá más ejemplos en /training.html."
        )


def targets(rows: list[dict], sentiment: list[dict | None]):
    """Category targets and mask (1 where the label is known), sentiment targets and mask."""
    import torch

    cats = torch.tensor([[1.0 if r["labels"][c] else 0.0 for c in CATEGORIES] for r in rows])
    cat_mask = torch.tensor([[float(r["labels"][c] is not None) for c in CATEGORIES] for r in rows])
    sent = []
    for p in sentiment:
        values = [float(p[s]) for s in SENTIMENTS] if p else [1.0] * len(SENTIMENTS)
        sent.append([v / sum(values) for v in values])
    sent_mask = torch.tensor([float(p is not None) for p in sentiment])
    return cats, cat_mask, torch.tensor(sent), sent_mask


def loss_fn(logits, cats, cat_mask, sent, sent_mask):
    import torch.nn.functional as F

    n = len(CATEGORIES)
    bce = F.binary_cross_entropy_with_logits(logits[:, :n], cats, reduction="none")
    category_loss = (bce * cat_mask).sum() / cat_mask.sum().clamp(min=1)
    ce = -(sent * F.log_softmax(logits[:, n:], dim=-1)).sum(-1)
    sentiment_loss = (ce * sent_mask).sum() / sent_mask.sum().clamp(min=1)
    return category_loss + sentiment_loss


def _load_base(config: TrainConfig):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # The new outputs start from random weights: seeding here makes `--seed` cover them too.
    torch.manual_seed(config.seed)
    model_id, revision = BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)},
    )
    return tokenizer, model


def _train(model, tokenizer, rows: list[dict], sentiment: list[dict | None], config: TrainConfig, device: str,
           log) -> list[float]:
    import torch

    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    per_epoch = math.ceil(len(rows) / config.batch_size)
    steps = config.epochs * per_epoch
    warm = int(config.warmup * steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (steps - s) / max(1, steps - warm)))
    on_gpu = torch.device(device).type == "cuda"
    use_bf16 = on_gpu and torch.cuda.is_bf16_supported()

    losses, order = [], list(range(len(rows)))
    for epoch in range(config.epochs):
        rng.shuffle(order)
        total, t0 = 0.0, time.time()
        for start in range(0, len(order), config.batch_size):
            started = time.perf_counter()
            idx = order[start:start + config.batch_size]
            encoded = tokenizer([rows[i]["text"] for i in idx], padding=True, truncation=True,
                                max_length=config.max_length, return_tensors="pt").to(device)
            batch_targets = [t.to(device) for t in targets([rows[i] for i in idx], [sentiment[i] for i in idx])]
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**encoded).logits.float()
            loss = loss_fn(logits, *batch_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
            if on_gpu:
                pause(started)
        losses.append(total / per_epoch)
        log(f"  época {epoch + 1}/{config.epochs}: pérdida {losses[-1]:.4f} ({time.time() - t0:.0f} s)")
    model.eval()
    return losses


def _flag_scores(contestant, examples, cache, log) -> dict[str, float]:
    from bench.cache import PredictionCache
    from bench.contestants.base import effective_scores
    from bench.runner import run as bench_run

    raw = bench_run([contestant], examples, PredictionCache(cache), batch_size=64, log=log)[contestant.name]
    return {i: effective_scores(contestant, r)[FLAG] for i, r in raw.items()}


def _spanish_auroc(examples, scores) -> float | None:
    pairs = [(gold(e, FLAG), scores[e.id]) for e in examples if e.id in scores and gold(e, FLAG) is not None]
    return auroc([g for g, _ in pairs], [s for _, s in pairs]) if pairs else None


def _compare_with_active(active: dict | None, new, new_scores, benchmark, sets, device, cache, log):
    """{"auroc", "sentiment_f1"} of the new and the active checkpoint on the Spanish eval rows neither
    trained on; the active side is None when no checkpoint is active."""
    from bench.contestants.finetuned import FineTunedContestant

    path = active["path"] if active else None
    if not path or checkpoint_problem(path):
        return {"auroc": _spanish_auroc(_spanish_eval(benchmark, new.trained_on), new_scores),
                "sentiment_f1": None}, None, None
    current = throttled(FineTunedContestant(path, device=device))
    excluded = new.trained_on | current.trained_on
    rows = _spanish_eval(benchmark, excluded)
    draft = {"es": [(t, g) for t, g in sets.get("suite-es-draft", []) if text_key(t) not in excluded]}
    log(f"Comparando con el modelo activo ({path.name})…")
    current_scores = _flag_scores(current, rows, cache, log)

    def side(contestant, scores):
        f1 = sentiment_f1(lambda texts: contestant.predict(texts, "es"), draft)["es"] if draft["es"] else None
        return {"auroc": _spanish_auroc(rows, scores), "sentiment_f1": f1}

    result = side(new, new_scores), side(current, current_scores), path.name
    current.unload()
    return result


def _spanish_eval(benchmark, excluded: frozenset) -> list:
    return [e for e in benchmark
            if e.source == SPANISH_SOURCE and split_of(e) == "eval" and text_key(e.text) not in excluded]


def run(examples: ExampleStore, models_dir: Path, config: TrainConfig | None = None, activate: bool = True,
        device: str | None = None, log=print, inputs=None, labeler=None, sentiment_labels: SentimentLabels | None = None,
        label_sentiment: bool = True, sentiment_dir: Path = SENTIMENT_DIR, cache: Path = DEFAULT_CACHE,
        cooldown: float = COOLDOWN_S) -> dict:
    """Train, save, evaluate and maybe activate one checkpoint. `inputs` (build_rows' arguments) and
    `labeler` replace the Hugging Face datasets and Qwen3-4B; tests pass small fakes."""
    import torch

    from bench.contestants.finetuned import FineTunedContestant

    config = config or TrainConfig()
    app_examples = examples.all()
    _check_enough(app_examples)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    log("Armando el set de entrenamiento…")
    inputs = inputs or load_inputs(app_examples)
    benchmark = inputs[1]
    rows, stats = build_rows(*inputs)
    log(f"  {len(rows)} filas: ToLD-Br {stats['toldbr']}, OLID-BR {stats['olidbr']}, "
        f"español {stats['spanish']} ×3. Se descartaron {stats['dropped']} que repetían textos del benchmark.")

    labels = sentiment_labels or SentimentLabels()
    if label_sentiment:
        missing = sum(labels.get(r["text"]) is None for r in rows)
        if missing:
            log(f"Etiquetando el sentimiento de los textos nuevos con Qwen3-4B (se hace una sola vez)…")
            own = labeler is None
            labeler = labeler or throttled(sentiment_labeler(device=device))
            label_missing([r["text"] for r in rows], labels, labeler, log=log)
            if own:
                labeler.unload()
                if cooldown and torch.device(device).type == "cuda":
                    log(f"Pausa de {cooldown:.0f} s para que la GPU se enfríe…")
                    time.sleep(cooldown)
    sentiment = [labels.get(r["text"]) for r in rows]
    log(f"  Sentimiento etiquetado en {sum(p is not None for p in sentiment)} de {len(rows)} filas.")

    model_id, revision = BASE_MODEL
    log(f"Entrenando {model_id} en {device} ({config.epochs} épocas)…")
    tokenizer, model = _load_base(config)
    losses = _train(model, tokenizer, rows, sentiment, config, device, log)

    created = datetime.now(timezone.utc)
    out = Path(models_dir) / f"ft-{created:%Y%m%d-%H%M%S}"
    log(f"Guardando el checkpoint en {out}…")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    meta = {
        "created_at": created.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "base": {"id": model_id, "revision": revision},
        "labels": list(LABELS),
        "threshold": None,
        "config": asdict(config),
        "counts": {k: v for k, v in stats.items() if k != "trained_texts"} | {"rows": len(rows)},
        "losses": losses,
        "trained_texts": stats["trained_texts"],
    }
    write_json_atomic(out / "training_meta.json", meta)

    log("Evaluando con el benchmark…")
    new = throttled(FineTunedContestant(out, device=device, loader=lambda: (tokenizer, model)))
    evaluated = [e for e in benchmark if text_key(e.text) not in new.trained_on]
    scores = _flag_scores(new, evaluated, cache, log)
    threshold, threshold_from = choose_threshold(evaluated, scores)
    threshold = FALLBACK_THRESHOLD if threshold is None else threshold
    suite = [e for e in evaluated if e.source == SPANISH_SOURCE]
    sets = load_sentiment_sets(sentiment_dir, suite)
    sentiment_scores = sentiment_f1(lambda texts: new.predict(texts, "es"), sets)

    new_side, active_side, active_name = _compare_with_active(
        ActiveCheckpoint(models_dir).get(), new, scores, benchmark, sets, device, cache, log)
    ok, reason = activation_decision(new_side, active_side)
    metrics = {
        "threshold": threshold,
        "threshold_from": threshold_from,
        "flag_auroc": flag_auroc_by_source(evaluated, scores),
        "operating_point": operating_point(evaluated, scores, threshold),
        "sentiment_f1": sentiment_scores,
        "comparison": {"new": new_side, "active": active_side, "active_name": active_name},
    }
    meta.update(threshold=threshold, metrics=metrics, decision={"activate": ok, "reason": reason})
    write_json_atomic(out / "training_meta.json", meta)
    new.unload()

    activated = activate and ok
    if activated:
        ActiveCheckpoint(models_dir).activate(out, metrics)
    return {"checkpoint": out, "metrics": metrics, "activated": activated, "reason": reason}
