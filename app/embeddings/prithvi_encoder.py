from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class TemporalEmbeddingResult:
    vector: np.ndarray   # shape (dim,)
    backend: str
    time_steps: int


def _fit_and_normalize(vector: np.ndarray, dimension: int) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vec.size == dimension:
        pass
    elif vec.size > dimension:
        vec = np.interp(
            np.linspace(0, vec.size - 1, dimension),
            np.arange(vec.size),
            vec,
        ).astype(np.float32)
    else:
        padded = np.zeros(dimension, dtype=np.float32)
        padded[: vec.size] = vec
        vec = padded
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-8 else vec


class PrithviEncoder:
    """
    Prithvi-EO-2.0 temporal sequence encoder.
    Processes sequences of co-registered satellite chips across time (T, C, H, W).
    """

    def __init__(self, model_path: str | Path, dimension: int = 512, offline: bool = True):
        self.model_path = Path(model_path)
        self.dimension = dimension
        self.offline = offline
        self.backend = "temporal-statistical-fallback"
        self._model: Any | None = None

        if offline:
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
            self.backend = "local-prithvi-transformers"
        except Exception:
            self._model = None

    def encode_sequence(
        self,
        chips: list[np.ndarray],
        dates: list[str] | None = None,
    ) -> TemporalEmbeddingResult:
        if not chips:
            return TemporalEmbeddingResult(
                np.zeros(self.dimension, dtype=np.float32),
                self.backend,
                0,
            )

        if self._model is not None:
            try:
                import torch
                stacked = np.stack(chips, axis=0).astype(np.float32)
                tensor = torch.from_numpy(stacked).unsqueeze(0)
                with torch.no_grad():
                    output = self._model(tensor)
                    if hasattr(output, "last_hidden_state"):
                        vector = output.last_hidden_state.mean(dim=(1, 2)).squeeze().cpu().numpy()
                    else:
                        vector = output[0].squeeze().cpu().numpy()
                return TemporalEmbeddingResult(
                    _fit_and_normalize(vector, self.dimension),
                    self.backend,
                    len(chips),
                )
            except Exception:
                pass

        # Statistical fallback across temporal sequence
        arr = np.stack(chips, axis=0).astype(np.float32)
        T, C, H, W = arr.shape
        features: list[float] = []
        for c in range(C):
            band = arr[:, c, :, :]
            flat = band.reshape(T, -1)
            spatial_mean = np.nanmean(flat, axis=1)
            spatial_std = np.nanstd(flat, axis=1)
            features.extend(spatial_mean.tolist())
            features.extend(spatial_std.tolist())
            if T > 1:
                deltas = np.diff(spatial_mean)
                features.extend(deltas.tolist())

        vec = np.asarray(features, dtype=np.float32)
        return TemporalEmbeddingResult(
            _fit_and_normalize(vec, self.dimension),
            "temporal-statistical-fallback",
            len(chips),
        )

    def compute_cosine_distance_series(
        self,
        baseline: np.ndarray,
        sequence: list[np.ndarray],
    ) -> list[float]:
        distances = []
        b_vec = _fit_and_normalize(baseline, self.dimension)
        for chip in sequence:
            res = self.encode_sequence([chip])
            dist = float(np.clip(1.0 - np.dot(b_vec, res.vector), 0.0, 1.0))
            distances.append(round(dist, 4))
        return distances
