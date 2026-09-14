from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class EmbeddingResult:
    vector: np.ndarray
    backend: str


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    return values / norm if norm else values


def _fit_dimension(vector: np.ndarray, dimension: int) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size == dimension:
        return _unit(values)
    return _unit(
        np.interp(
            np.linspace(0, values.size - 1, dimension),
            np.arange(values.size),
            values,
        ).astype(np.float32)
    )


def _hashed_vector(tokens: list[str], dimension: int) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        seed = int.from_bytes(digest[:8], "little")
        generator = np.random.default_rng(seed)
        vector += generator.standard_normal(dimension).astype(np.float32)
    return _unit(vector)


def _image_statistics(image: np.ndarray, dimension: int) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim == 2:
        values = values[np.newaxis, :, :]
    features: list[float] = []
    for band in values:
        finite = band[np.isfinite(band)]
        if not finite.size:
            features.extend([0.0] * 13)
            continue
        features.extend(
            [float(np.mean(finite)), float(np.std(finite)),
             *[float(x) for x in np.percentile(finite, [1, 10, 25, 50, 75, 90, 99])]]
        )
        vertical = np.diff(np.nan_to_num(band), axis=0)
        horizontal = np.diff(np.nan_to_num(band), axis=1)
        features.extend([
            float(np.mean(np.abs(vertical))),
            float(np.mean(np.abs(horizontal))),
            float(np.mean(vertical * vertical)),
            float(np.mean(horizontal * horizontal)),
        ])
    base = np.asarray(features, dtype=np.float32)
    seed = int.from_bytes(
        hashlib.blake2b(str(base.shape).encode("utf-8"), digest_size=8).digest(), "little"
    )
    projection = np.random.default_rng(seed).standard_normal((dimension, base.size)).astype(np.float32)
    return _unit(projection @ base)


def _to_pil_rgb(image: np.ndarray) -> Any:
    """Convert a CHW float32 satellite image to an 8-bit PIL RGB image."""
    from PIL import Image

    display = np.asarray(image, dtype=np.float32)
    if display.ndim == 3:
        display = np.moveaxis(display[:3], 0, -1)
    elif display.ndim == 2:
        display = np.stack([display, display, display], axis=-1)

    finite = display[np.isfinite(display)]
    low, high = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
    if high <= low:
        low, high = float(display.min()), float(display.max())
    if high <= low:
        high = low + 1.0

    rendered = np.clip(
        (np.nan_to_num(display, nan=low) - low) / (high - low), 0, 1
    )
    if rendered.shape[-1] == 1:
        rendered = np.repeat(rendered, 3, axis=-1)
    elif rendered.shape[-1] == 2:
        rendered = np.concatenate([rendered, rendered[:, :, :1]], axis=-1)

    return Image.fromarray((rendered * 255).astype(np.uint8))


class TextImageEncoder:
    """
    Unified CLIP text+image encoder.

    Loading priority:
      1. open_clip local model (preferred — handles RemoteCLIP & standard CLIP)
      2. transformers AutoModel local (fallback for HF-format CLIP checkpoints)
      3. Statistical hash fallback (offline degraded mode)

    Models must be pre-staged. The encoder NEVER downloads weights at runtime.
    """

    def __init__(self, model_path: str | Path, dimension: int = 512, offline: bool = True):
        self.model_path = Path(model_path)
        self.dimension = dimension
        self.offline = offline
        self.backend = "local-statistical-fallback"
        self._model: Any | None = None
        self._processor: Any | None = None
        self._tokenizer: Any | None = None
        self._open_clip_preprocess: Any | None = None
        self._use_open_clip = False

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

        if self._try_open_clip():
            return

        self._try_transformers()

    def _try_open_clip(self) -> bool:
        """Attempt to load via open_clip from a local checkpoint directory."""
        try:
            import open_clip
            import torch

            ckpt_files = [
                p for p in self.model_path.rglob("*")
                if p.suffix in {".pt", ".pth", ".bin", ".safetensors"} and p.is_file()
            ]

            config_file = self.model_path / "open_clip_config.json"

            if config_file.exists():
                import json
                cfg = json.loads(config_file.read_text())
                model_name = cfg.get("model_cfg", {}).get("model_name", "ViT-B-32")
            else:
                model_name = "ViT-B-32"

            if ckpt_files:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=str(ckpt_files[0])
                )
                tokenizer = open_clip.get_tokenizer(model_name)
            else:
                return False

            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            self._model = model
            self._open_clip_preprocess = preprocess
            self._tokenizer = tokenizer
            self._use_open_clip = True
            self.backend = f"local-open-clip/{model_name}"
            return True
        except Exception:
            return False

    def _try_transformers(self) -> bool:
        """Attempt to load via HuggingFace transformers (CLIP or RemoteCLIP)."""
        try:
            from transformers import AutoModel, AutoProcessor

            model = AutoModel.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=True
            )
            model.eval()
            self._model = model
            self._processor = processor
            self._use_open_clip = False
            self.backend = "local-clip-transformers"
            return True
        except Exception:
            return False


    def encode_text(self, query: str) -> EmbeddingResult:
        if self._model is not None:
            try:
                return self._encode_text_model(query)
            except Exception:
                pass
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        expanded = tokens + [f"bigram:{l}:{r}" for l, r in zip(tokens, tokens[1:])]
        return EmbeddingResult(
            _hashed_vector(expanded or ["empty-query"], self.dimension),
            "local-text-hash-fallback",
        )

    def _encode_text_model(self, query: str) -> EmbeddingResult:
        import torch

        with torch.no_grad():
            if self._use_open_clip:
                tokens = self._tokenizer([query])
                if torch.cuda.is_available() and next(self._model.parameters()).is_cuda:
                    tokens = tokens.cuda()
                features = self._model.encode_text(tokens)
                vector = features[0].detach().cpu().float().numpy()
            else:
                batch = self._processor(
                    text=[query], return_tensors="pt", padding=True, truncation=True
                )
                output = self._model.get_text_features(**batch)
                vector = output[0].detach().cpu().numpy()

        return EmbeddingResult(_fit_dimension(vector, self.dimension), self.backend)

    def encode_image(self, image: np.ndarray) -> EmbeddingResult:
        if self._model is not None:
            try:
                return self._encode_image_model(image)
            except Exception:
                pass
        return EmbeddingResult(_image_statistics(image, self.dimension), "local-statistical-fallback")

    def _encode_image_model(self, image: np.ndarray) -> EmbeddingResult:
        import torch

        pil_image = _to_pil_rgb(image)

        with torch.no_grad():
            if self._use_open_clip:
                tensor = self._open_clip_preprocess(pil_image).unsqueeze(0)
                if torch.cuda.is_available() and next(self._model.parameters()).is_cuda:
                    tensor = tensor.cuda()
                features = self._model.encode_image(tensor)
                vector = features[0].detach().cpu().float().numpy()
            else:
                batch = self._processor(images=pil_image, return_tensors="pt")
                output = self._model.get_image_features(**batch)
                vector = output[0].detach().cpu().numpy()

        return EmbeddingResult(_fit_dimension(vector, self.dimension), self.backend)
