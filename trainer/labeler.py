"""Qwen3-4B-Instruct (Apache-2.0) drafting the sentiment label of each training text, offline, on the GPU.

The question becomes one prompt that lists the sentiments under letters. The answer is the softmax of the
next-token logits over those letters: a probability per sentiment, read from one forward pass with no
generation. Only the trainer uses it; the served checkpoint never calls it.

Every label in data/training/sentiment_labels.jsonl was drafted with this exact wording, so editing the
prompt or the question mixes two labeling criteria in one training set.
"""

from __future__ import annotations

import hashlib
import json
from importlib import metadata

from bench.contestants.base import free_gpu

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
# The Hub commit checked on 2026-10-07.
QWEN3_4B_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
NAME = "qwen3-4b-sentiment"
SYSTEM_PROMPT = (
    "You moderate the player chat of an online casino and poker room. Players write in Rioplatense Spanish, "
    "Brazilian Portuguese or English, with slang, typos and gaming jargon. Read one chat message and answer a "
    "multiple-choice question about it with the letter of the right option only."
)
LETTERS = "ABCDEFGH"
# The overall sentiment of a message, one option per sentiment. Raw outputs use `sentiment.<option>` keys.
SENTIMENT_QUESTIONS = {
    "sentiment": {
        "type": "choice",
        "instructions": "What is the overall sentiment of `message`?",
        "criteria": {
            "positive": "favorable, happy, grateful, satisfied",
            "negative": "unhappy, angry, disappointed, critical",
            "neutral": "factual, or no clear sentiment",
        },
    },
}


def user_prompt(text: str, question: dict) -> str:
    instructions = question["instructions"].replace("`message`", "the message")
    options = "\n".join(f"{LETTERS[i]}) {option}: {criterion}"
                        for i, (option, criterion) in enumerate(question["criteria"].items()))
    return f'Message:\n"""\n{text}\n"""\n\nQuestion: {instructions}\n{options}\n\nAnswer with one letter.'


def chat(text: str, question: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(text, question)}]


def fingerprint(questions: dict) -> str:
    return hashlib.sha1(json.dumps(questions, sort_keys=True).encode("utf-8")).hexdigest()[:8]


def prompt_fingerprint() -> str:
    probe = chat("<text>", {"instructions": "<q>", "criteria": {"a": "<a>", "b": "<b>"}})
    return hashlib.sha1(repr(probe).encode("utf-8")).hexdigest()[:8]


class SentimentLabeler:
    def __init__(self, device: str | None = None, micro_batch: int = 16, loader=None):
        self.name = NAME
        self.model_id = MODEL_ID
        self.revision = QWEN3_4B_REVISION
        self.questions = SENTIMENT_QUESTIONS
        self.device = device
        self.micro_batch = micro_batch
        self._loader = loader or self._from_hub
        self._loaded = None
        # Recorded next to every label: the question, the prompt, the weights and the transformers version.
        self.version = (f"q{fingerprint(self.questions)}-p{prompt_fingerprint()}-{self.revision[:12]}"
                        f"-tf{metadata.version('transformers')}")

    def _from_hub(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        # Safetensors only and no remote code.
        model = AutoModelForCausalLM.from_pretrained(self.model_id, revision=self.revision, use_safetensors=True,
                                                     dtype=torch.bfloat16)
        return tokenizer, model

    def _get(self):
        if self._loaded is None:
            import torch

            tokenizer, model = self._loader()
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._loaded = (tokenizer, model.to(device).eval(), device)
        return self._loaded

    def _render(self, tokenizer, text: str, question: dict) -> str:
        # `enable_thinking=False` makes Qwen3 answer at once.
        return tokenizer.apply_chat_template(chat(text, question), tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)

    @staticmethod
    def letter_ids(tokenizer, n: int) -> list[int]:
        ids = []
        for letter in LETTERS[:n]:
            encoded = tokenizer.encode(letter, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"the tokenizer splits option letter {letter!r} into {len(encoded)} tokens")
            ids.append(encoded[0])
        return ids

    def predict(self, texts: list[str], lang: str) -> list[dict[str, float]]:
        """`{"sentiment.positive": p, ...}` per text; `lang` is unused, the prompt covers every language."""
        import torch

        tokenizer, model, device = self._get()
        jobs = [(i, qid, self._render(tokenizer, text, question))
                for i, text in enumerate(texts) for qid, question in self.questions.items()]
        raw: list[dict[str, float]] = [{} for _ in texts]
        for start in range(0, len(jobs), self.micro_batch):
            chunk = jobs[start:start + self.micro_batch]
            encoded = tokenizer([prompt for _, _, prompt in chunk], return_tensors="pt", padding=True,
                                add_special_tokens=False)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.inference_mode():
                # Only the last position's logits: the full vocabulary at every position would not fit.
                logits = model(**encoded, logits_to_keep=1).logits[:, -1, :].float()
            for row, (i, qid, _) in enumerate(chunk):
                options = list(self.questions[qid]["criteria"])
                probabilities = torch.softmax(logits[row, self.letter_ids(tokenizer, len(options))], -1).tolist()
                raw[i].update({f"{qid}.{option}": p for option, p in zip(options, probabilities)})
        return raw

    def unload(self) -> None:
        self._loaded = None
        free_gpu()
