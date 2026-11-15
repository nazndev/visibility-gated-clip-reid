"""Hugging Face Qwen2.5-VL backend.

The class names below were checked against transformers 5.22.2:
``Qwen2_5_VLForConditionalGeneration`` and ``AutoProcessor``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from reidvlm.lvlm.parser import parse_judgment
from reidvlm.lvlm.prompts import PAIR_PROMPT, PROMPT_VERSION, RETRY_NOTE
from reidvlm.lvlm.schema import SCHEMA_VERSION, PairJudgment
from reidvlm.runtime import pick_device


class LVLMUnavailable(RuntimeError):
    pass


@dataclass
class Attempt:
    raw: str
    valid: bool
    error: str | None
    seconds: float
    new_tokens: int | None = None


@dataclass
class JudgmentRecord:
    valid: bool
    parsed: PairJudgment | None
    attempts: list[Attempt]
    cached: bool = False

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


def completion_span(sequence_length: int, prompt_length: int) -> tuple[int, int]:
    """Return the start index and new-token count for one generate() result.

    The installed transformers build returns the prompt followed by the
    completion. A build that returns only new tokens is shorter than or equal
    to the prompt; that tensor is decoded whole.
    """
    if sequence_length <= prompt_length:
        return 0, sequence_length
    return prompt_length, sequence_length - prompt_length


def _dtype(name: str, device: str) -> torch.dtype:
    if device == "cpu" and name == "float16":
        return torch.float32
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[name]


class QwenVLBackend:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        revision: str | None = None,
        dtype: str = "float16",
        max_new_tokens: int = 384,
        max_attempts: int = 2,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.requested_revision = revision
        self.requested_dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.max_attempts = max_attempts
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.device = device or pick_device()
        self.model = None
        self.processor = None
        self.resolved_revision = None
        self.processor_revision = None
        self.actual_dtype = None

    def load(self) -> None:
        if self.model is not None:
            return
        from reidvlm.runtime import configure_hf_environment

        configure_hf_environment()
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise LVLMUnavailable(f"Could not import Qwen2.5-VL: {exc}") from exc
        dtype = _dtype(self.requested_dtype, self.device)
        self.actual_dtype = str(dtype).replace("torch.", "")
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                revision=self.requested_revision,
                torch_dtype=dtype,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id, revision=self.requested_revision)
        except Exception as exc:
            raise LVLMUnavailable(f"Could not load {self.model_id}: {exc}") from exc
        if self.min_pixels is not None:
            self.processor.image_processor.min_pixels = self.min_pixels
        if self.max_pixels is not None:
            self.processor.image_processor.max_pixels = self.max_pixels
        self.model.to(self.device)
        self.model.eval()
        commit = getattr(self.model.config, "_commit_hash", None)
        if not commit:
            try:
                from huggingface_hub import snapshot_download

                snapshot = snapshot_download(
                    self.model_id,
                    revision=self.requested_revision,
                    local_files_only=True,
                )
                commit = Path(snapshot).name
            except Exception:
                commit = self.requested_revision or "unknown"
        self.resolved_revision = commit
        self.processor_revision = (
            getattr(getattr(self.processor, "tokenizer", None), "init_kwargs", {}).get("revision")
            or self.resolved_revision
        )

    def identity(self) -> dict:
        self.load()
        image_processor = self.processor.image_processor
        return {
            "backend": "transformers",
            "model_id": self.model_id,
            "model_revision": self.resolved_revision,
            "processor_revision": self.processor_revision,
            "dtype": self.actual_dtype,
            "device": self.device,
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "min_pixels": getattr(image_processor, "min_pixels", None),
            "max_pixels": getattr(image_processor, "max_pixels", None),
        }

    def _generate_once(self, image_a: Image.Image, image_b: Image.Image, retry: bool) -> tuple[str, int]:
        if not isinstance(image_a, Image.Image) or not isinstance(image_b, Image.Image):
            raise TypeError("The LVLM receives decoded images only.")
        text = PAIR_PROMPT if not retry else PAIR_PROMPT + "\n\n" + RETRY_NOTE
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_a},
                    {"type": "image", "image": image_b},
                    {"type": "text", "text": text},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            inputs = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        prompt_length = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        start, new_tokens = completion_span(int(generated.shape[-1]), prompt_length)
        trimmed = generated[:, start:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0], new_tokens

    def judge(self, image_a: Image.Image, image_b: Image.Image) -> JudgmentRecord:
        self.load()
        attempts: list[Attempt] = []
        parsed = None
        for attempt_index in range(self.max_attempts):
            started = time.perf_counter()
            try:
                raw, new_tokens = self._generate_once(image_a, image_b, retry=attempt_index > 0)
                parsed = parse_judgment(raw)
                attempts.append(Attempt(raw, True, None, time.perf_counter() - started, new_tokens))
                break
            except Exception as exc:
                raw = locals().get("raw", "")
                new_tokens = locals().get("new_tokens")
                attempts.append(
                    Attempt(str(raw), False, f"{type(exc).__name__}: {exc}", time.perf_counter() - started, new_tokens)
                )
        return JudgmentRecord(valid=parsed is not None, parsed=parsed, attempts=attempts)
