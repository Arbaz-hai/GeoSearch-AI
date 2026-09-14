from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


class LocalImageCaptioner:
    """
    Offline BLIP image captioner.

    - Loads from a locally staged model directory.
    - Never downloads weights at runtime.
    - Uses a satellite-specific prompt prefix for better domain accuracy.
    - Captions are visual descriptions only — NOT geographic identifications.
    """

    _SATELLITE_PROMPT = "An aerial satellite image showing"

    def __init__(self, model_path: str | Path, offline: bool = True):
        self.model_path = Path(model_path)
        self.offline = offline
        self.processor: Any | None = None
        self.model: Any | None = None
        self.backend = "filename-and-metadata"

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
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self.processor = BlipProcessor.from_pretrained(
                self.model_path, local_files_only=True
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_path, local_files_only=True
            )
            self.model.eval()
            self.backend = "local-blip"
        except Exception:
            self.processor = None
            self.model = None

    def caption(self, image: np.ndarray) -> str | None:
        """
        Generate a satellite-imagery description of the image.

        Returns a string caption or None if model is unavailable.
        The returned text is a VISUAL description, not a geographic identification.
        """
        if self.model is None or self.processor is None:
            return None
        try:
            import torch
            from PIL import Image

            values = np.asarray(image, dtype=np.float32)

            if values.ndim == 3:
                values = np.moveaxis(values[:3], 0, -1)
            if values.ndim == 2:
                values = np.repeat(values[:, :, None], 3, axis=2)
            elif values.shape[2] == 1:
                values = np.repeat(values, 3, axis=2)
            elif values.shape[2] == 2:
                values = np.concatenate([values, values[:, :, :1]], axis=2)

            finite = values[np.isfinite(values)]
            low, high = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
            if high <= low:
                high = low + 1.0
            rendered = (
                np.clip(
                    (np.nan_to_num(values, nan=low) - low) / (high - low), 0, 1
                ) * 255
            ).astype(np.uint8)

            pil_image = Image.fromarray(rendered)

            inputs = self.processor(
                images=pil_image,
                text=self._SATELLITE_PROMPT,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=60)

            raw = self.processor.decode(output[0], skip_special_tokens=True).strip()

            for prefix in (self._SATELLITE_PROMPT, "an aerial satellite image showing"):
                if raw.lower().startswith(prefix.lower()):
                    raw = raw[len(prefix):].strip().lstrip(",").strip()
                    break

            if not raw:
                return None

            return raw[0].upper() + raw[1:] if len(raw) > 1 else raw.upper()

        except Exception:
            return None

    def is_available(self) -> bool:
        return self.model is not None and self.processor is not None
