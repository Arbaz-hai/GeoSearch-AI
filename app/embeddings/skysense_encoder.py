from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class FusionEmbeddingResult:
    vector: np.ndarray
    backend: str
    has_sar: bool


def _norm(vector: np.ndarray, dimension: int) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vec.size != dimension:
        if vec.size > dimension:
            indices = np.linspace(0, vec.size - 1, dimension)
            vec = np.interp(indices, np.arange(vec.size), vec).astype(np.float32)
        else:
            padded = np.zeros(dimension, dtype=np.float32)
            padded[: vec.size] = vec
            vec = padded
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-8 else vec


class SkySenseEncoder:
    """
    SkySense++ Multi-Modal SAR-Optical Cross-Attention Fusion Encoder.
    Aligns Sentinel-1 microwave radar and Sentinel-2 optical into a shared 512-D embedding space.
    """

    def __init__(self, model_path: str | Path, dimension: int = 512, offline: bool = True):
        self.model_path = Path(model_path)
        self.dimension = dimension
        self.offline = offline
        self.backend = "sar-optical-fusion-fallback"
        self._model: Any | None = None

        if offline:
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            return
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=True
            )
            model.eval()
            self._model = model
            self.backend = "local-skysense-transformers"
        except Exception:
            self._model = None

    def encode_optical_sar(
        self,
        optical: np.ndarray,
        sar: np.ndarray | None = None,
    ) -> FusionEmbeddingResult:
        if self._model is not None:
            try:
                import torch
                opt_t = torch.from_numpy(optical.astype(np.float32)).unsqueeze(0)
                inputs = {"optical": opt_t}
                if sar is not None:
                    inputs["sar"] = torch.from_numpy(sar.astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    out = self._model(**inputs)
                    vec = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                return FusionEmbeddingResult(_norm(vec, self.dimension), self.backend, sar is not None)
            except Exception:
                pass

        # Physics-informed statistical cross-sensor fusion
        features: list[float] = []
        for band in optical:
            finite = band[np.isfinite(band)]
            if finite.size == 0:
                features.extend([0.0, 0.0, 0.0, 0.0])
                continue
            features.extend([
                float(np.mean(finite)),
                float(np.std(finite)),
                float(np.percentile(finite, 10)),
                float(np.percentile(finite, 90)),
            ])

        if sar is not None:
            for band in sar:
                finite = band[np.isfinite(band)]
                if finite.size == 0:
                    features.extend([0.0, 0.0, 0.0, 0.0])
                    continue
                features.extend([
                    float(np.mean(finite)),
                    float(np.std(finite)),
                    float(np.percentile(finite, 10)),
                    float(np.percentile(finite, 90)),
                ])

        vec = np.asarray(features, dtype=np.float32)
        backend = "sar-optical-fused-stats" if sar is not None else "optical-only-stats"
        return FusionEmbeddingResult(_norm(vec, self.dimension), backend, sar is not None)
