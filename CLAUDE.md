# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Python 3.13 managed with uv. `uv sync` installs the `dev` and `bench` groups by default. On Windows and Linux, torch comes from the PyTorch CUDA 13.0 index (`pyproject.toml`).

```bash
uv run uvicorn app.main:app --port 8080          # chat + review UI; loads the active checkpoint on CPU at startup
uv run pytest                                     # fast suite; no model or dataset downloads
uv run pytest tests/test_bench_contestants.py::test_factories_list_every_complete_checkpoint_oldest_first  # a single test
uv run pytest -m slow                             # real checkpoints / HF datasets (GPU helps)
uv run python scripts/try_model.py "callate gil"  # classify text without the server
uv run python scripts/finetune.py [--seed N] [--no-activate]   # train, evaluate; activates only if better
uv run python scripts/activate_model.py --list | models/ft-...
uv run python scripts/bench.py sources | run [--checkpoints a,b] [--include-unreviewed] | latency --device cpu --threads 4 | agreement a.csv b.csv
```

`pytest` adds `-m 'not slow'` and `pythonpath = ["."]`, so tests import `app`, `bench` and `trainer` directly. Scripts put the repo root on `sys.path` themselves. No linter or formatter is configured.

## Architecture

The project's one model is **mmBERT-base fine-tuned** with eight outputs. Three packages share it:

- **`app/`**: a FastAPI server that also serves the vanilla-JS `static/` (no build step). `StaticFiles` is mounted at `/` last in `create_app`, so API routes must be registered before it.
- **`trainer/`**: fine-tunes mmBERT-base on the taxonomy categories plus sentiment, from ToLD-Br, OLID-BR and the app's labeled Spanish examples.
- **`bench/`**: the es/pt/en exam that compares fine-tuned checkpoints. Its contestants are the complete checkpoints in `models/` (`bench.contestants.checkpoint_factories`), named by folder; `--checkpoints` narrows them.

### Coupling through `app/model.py` and `training_meta.json`

- **A checkpoint** is a plain `ModernBertForSequenceClassification` with eight outputs named in `id2label` (`app.model.LABELS`): the five categories as independent sigmoids and `sentiment.*` as one softmax. `app.model.output_names` refuses any other head. `predict_raw` turns logits into that raw dict for the app classifier, `FineTunedContestant` and the trainer's evaluation.
- **`flag`** is `max(insult, threat, identity_hate)`, flagged when it is above the threshold the trainer saved in `training_meta.json` (`FLAG_THRESHOLD` overrides it). The same rule is `bench.contestants.base.effective_scores`.
- **`training_meta.json`** also lists `trained_texts`: `app.examples.text_key` hashes of the app examples the checkpoint trained on. Most are suite rows, so `bench.py run` drops the union of every compared checkpoint's trained texts for all of them, and the trainer evaluates without them.
- **The trainer evaluates through the benchmark**: same sources, `split_of` splits and metrics, and its predictions land in `bench_out/predictions/<folder>@<created_at>/`, the same cache key `bench.py run` uses, so the scoreboard reuses them.
- **Sentiment labels** come from Qwen3-4B-Instruct (`trainer/labeler.py`, offline, GPU) and are cached by text in `data/training/sentiment_labels.jsonl`. Qwen only labels training data; the app never loads it. The ToLD-Br sample uses a fixed seed (`trainer.data.DATA_SEED`), so `--seed` never forces relabeling.
- **The labeler's wording is pinned**: `test_question_and_prompt_wording_match_the_cached_labels` fixes the question and prompt fingerprints, because every cached label was drafted with that wording.
- **Lazy torch imports.** `torch` and `transformers` are imported inside functions in `app/`, `trainer/` and `bench/contestants/`, so the fast tests load no model. Tests inject fakes through parameters instead: `create_app(classifier=..., store=..., examples=...)`, `load_classifier(active, loader=...)`, `FineTunedContestant(loader=...)`, `SentimentLabeler(loader=...)`, `trainer.finetune.run(inputs=..., labeler=..., ...)` and `bench.cli.main(argv, sources=..., checkpoints=...)`. Keep these seams when changing those signatures.
- **GPU heat.** Long GPU work goes through `trainer.throttle` (`DUTY=0.6` sleeps after each step). Never run two GPU jobs at once.

### State on disk (gitignored)

- `data/messages.json` and `data/training/examples.json` are plain JSON, always written through `app/jsonfile.write_json_atomic`.
- `data/training/sentiment_labels.jsonl` caches Qwen3-4B's sentiment for every training text. `data/sentiment/` holds the sentiment eval sets (DynaSent R2 test and an unreviewed draft of the suite's Spanish sentiment); without them the trainer skips the sentiment report.
- `models/ft-*/` holds the fine-tuned checkpoints (~1.2 GB each). `models/active.json` points at the one the server loads; without it the server refuses to start.
- `bench_out/` holds the prediction cache (`predictions/<checkpoint>@<created_at>/<source>.jsonl`), `scoreboard.{md,json}` and `latency.json`.

### Two different split schemes

- **Training examples** (`app/examples.assign_split`): hash of the normalized text into 70% train, 10% calib, 20% test. The same sentence always lands in the same split. The trainer learns from `train` only; its threshold and metrics come from the benchmark splits below.
- **Benchmark** (`bench/sources.split_of`): hash of `cluster_key`, with 1 in 5 going to `calib`, where each checkpoint's operating threshold is set at 5% FPR. Clustering keeps HateCheck templates on one side, and the bootstrap resamples by cluster too.

### Benchmark pipeline

`Source.load` → `Example` (`bench/taxonomy.py`) → `runner.run` (inference through `PredictionCache`) → `contestant.to_scores` → `scoreboard.build_scoreboard` → `report.render`, Spanish Markdown.

- **The cache stores raw model outputs, not taxonomy scores**, so fixing a mapping in `to_scores` needs no new inference. `to_scores` must stay pure.
- A contestant's **`version` is part of the cache key**. For a checkpoint it is its `created_at`, so a retrain into the same folder never reuses old predictions.
- Each cached line also stores a text hash, so a suite row whose text a reviewer edited is re-run automatically.
- Every model and dataset is **pinned to a Hub revision**: `trainer.finetune.BASE_MODEL` (mmBERT-base), `trainer.labeler.QWEN3_4B_REVISION`, `HATECHECK_REVISIONS` and the Portuguese loaders.
- **Only open-source, self-hosted models and permissively licensed data**: no paid or hosted APIs, and no message leaves our machines. The licenses are listed in `README.md` ("Data and licenses").
- `profanity` and `sexual_harassment` are scored separately and are not part of `flag`. The model has no `target` output, so `targets_player` shows n/a.
- Labels are `True`, `False` or `None`. `None` (unknown) is never scored as negative.

### The reviewer suite (`suites/es_casino.csv`)

- Spanish reviewers edit it in Excel. It is `;`-delimited UTF-8 with a BOM, and the parser also accepts `,`.
- Only rows with `reviewed=1` count, unless `--include-unreviewed` is passed, which marks the scoreboard PRELIMINAR.
- `?` and empty cells are both unscored.
- `scripts/seed_suite.py --force` regenerates it from `datasets/` and `suites/drafts/` and **overwrites the human reviews**: don't run it on a reviewed suite.
- The labeling policy, including hard cases such as friendly "boludo", negations and quoting an insult, is in `docs/benchmark.md`.

## Conventions

- **Language split**:
  - Code, comments, docstrings and `README.md` are in English.
  - Everything a user or reviewer reads is in Rioplatense Spanish: CLI output and error messages, `SuiteError` and `CliError` text, the scoreboard report, the UI, `docs/benchmark.md` and `docs/finetuning.md`.
  - Tests match on the Spanish messages (e.g. `match="desconocido"`).
- **Commit messages** use emoji + type(scope): summary, for example `✨ New Feature(bench): ...`, `👌 Code review(bench): ...`, `📖 DOC(bench): ...`, `➕ Adding Dependency(bench): ...`.
