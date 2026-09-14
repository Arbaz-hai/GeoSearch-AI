from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from app.embeddings.text_image_encoder import (
    EmbeddingResult,
    _fit_dimension,
    _image_statistics,
    _to_pil_rgb,
    _unit,
)


class VisualEncoder:
    """
    Multi-spectral visual encoder for satellite imagery.

    Loading priority:
      1. Prithvi / SatMAE transformer (local staged weights)
      2. CLIP image encoder fallback (if main encoder staged at clip path)
      3. Statistical image-statistics fallback
    """

    def __init__(self, model_path: str | Path, dimension: int = 512, offline: bool = True):
        self.model_path = Path(model_path)
        self.dimension = dimension
        self.offline = offline
        self.backend = "local-multispectral-statistical-fallback"
        self._model: Any | None = None
        self._processor: Any | None = None

        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self._load_local_model()

    def _load_local_model(self) -> None:
        if not self.model_path.exists():
            return
        try:
            children = list(self.model_path.iterdir())
        except (PermissionError, OSError):
            return
        if not children:
            return

        try:
            from transformers import AutoImageProcessor, AutoModel

            self._model = AutoModel.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=True
            )
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=True
            )
            self._model.eval()
            self.backend = "local-prithvi-multispectral"
        except Exception:
            self._model = None
            self._processor = None

    def encode_image(self, image: np.ndarray) -> EmbeddingResult:
        if self._model is not None and self._processor is not None:
            try:
                return self._encode_prithvi(image)
            except Exception:
                pass
        return EmbeddingResult(_image_statistics(image, self.dimension), self.backend)

    def _encode_prithvi(self, image: np.ndarray) -> EmbeddingResult:
        import torch

        values = np.asarray(image, dtype=np.float32)
        if values.ndim == 3:
            display = np.moveaxis(values[:3], 0, -1)
        else:
            display = np.stack([values, values, values], axis=-1)

        finite = display[np.isfinite(display)]
        low, high = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
        if high <= low:
            high = low + 1.0
        rendered = np.clip((np.nan_to_num(display, nan=low) - low) / (high - low), 0, 1)

        batch = self._processor(images=rendered, return_tensors="pt")
        with torch.no_grad():
            output = self._model(**batch)

        if getattr(output, "pooler_output", None) is not None:
            vector = output.pooler_output[0].detach().cpu().numpy().astype(np.float32)
        else:
            vector = output.last_hidden_state.mean(dim=1)[0].detach().cpu().numpy().astype(np.float32)

        return EmbeddingResult(_fit_dimension(vector, self.dimension), self.backend)
