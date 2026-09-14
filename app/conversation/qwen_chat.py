from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from collections import Counter


class LocalQwenChat:
    """
    Offline Qwen3 analyst Q&A — grounded strictly in supplied evidence.
    
    - Only answers from the provided change-detection evidence.
    - Does NOT hallucinate locations, dates, or observations.
    - Falls back to deterministic keyword-based answers when model unavailable.
    """

    def __init__(self, model_path: str | Path, offline: bool = True):
        self.model_path = Path(model_path)
        self.offline = offline
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.backend = "deterministic-local-answer"

        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if self.model_path.exists():
            try:
                children = list(self.model_path.iterdir())
            except (PermissionError, OSError):
                children = []
            if children:
                self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                torch_dtype="auto",
            )
            self.model.eval()
            self.backend = "local-qwen3"
        except Exception:
            self.tokenizer = None
            self.model = None

    def answer(self, question: str, evidence: list[dict[str, Any]]) -> str | None:
        """
        Answer a question strictly from supplied evidence.
        Returns None to trigger fallback keyword logic.
        """
        if self.model is None or self.tokenizer is None:
            return None
        try:
            import torch

            summary = json.dumps(
                [
                    {
                        k: v for k, v in event.items()
                        if k in {
                            "change_type", "confidence", "earliest_detected_date",
                            "before_date", "after_date", "sensor", "location", "status",
                        }
                    }
                    for event in evidence[:20]
                ],
                sort_keys=True,
                default=str,
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an offline satellite imagery analyst assistant. "
                        "Answer ONLY from the supplied change-detection evidence JSON. "
                        "If the evidence does not support the answer, say so clearly. "
                        "Be concise. Include dates, change types, and confidence values. "
                        "Do NOT invent locations, coordinates, or observations not in the evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Evidence (JSON):\n{summary}\n\nQuestion: {question.strip()}",
                },
            ]

            if hasattr(self.tokenizer, "apply_chat_template"):
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                )
            else:
                prompt = f"Evidence:\n{summary}\n\nQuestion: {question.strip()}\nAnswer:"
                inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"]

            with torch.no_grad():
                generated = self.model.generate(
                    inputs, max_new_tokens=256, do_sample=False
                )

            output = generated[0][inputs.shape[-1]:]
            text = self.tokenizer.decode(output, skip_special_tokens=True).strip()

            if "<think>" in text:
                text = text.split("</think>", 1)[-1].strip()

            return text or None
        except Exception:
            return None

    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None
