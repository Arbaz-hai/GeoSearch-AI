from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image


@dataclass(slots=True)
class SceneRaster:
    scene_id: str
    source_path: Path
    image: np.ndarray
    transform: tuple[float, float, float, float, float, float]
    crs: str
    bounds: tuple[float, float, float, float]
    acquisition_date: str
    sensor: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class TilePatch:
    tile_id: str
    scene_id: str
    image: np.ndarray
    grid_row: int
    grid_col: int
    transform: tuple[float, float, float, float, float, float]
    crs: str
    bounds: tuple[float, float, float, float]


def infer_sensor(path: str | Path, metadata: dict[str, Any] | None = None) -> str:
    token = f"{path} {json.dumps(metadata or {})}".lower()
    if "sentinel-2 rgb" in token:
        return "Sentinel-2 RGB"
    if "sentinel-2" in token or "sentinel2" in token or "s2" in token:
        return "Sentinel-2 L2A"
    if "sentinel-1" in token or "sentinel1" in token or "sar" in token:
        return "Sentinel-1"
    if "landsat" in token or re.search(r"\blc0?\d", token):
        return "Landsat Collection 2"
    if "bhuvan" in token:
        return "Bhuvan"
    return "Unknown"


def infer_acquisition_date(path: str | Path, metadata: dict[str, Any] | None = None) -> str:
    values = metadata or {}
    for key in ["acquisition_date", "datetime", "date", "TIFFTAG_DATETIME"]:
        if values.get(key):
            return str(values[key])[:10]
    match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", Path(path).name)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return datetime.fromtimestamp(Path(path).stat().st_mtime).date().isoformat()


def _scene_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}-{digest}"


def _read_with_rasterio(path: Path) -> SceneRaster:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        dataset = rasterio.open(path)
    with dataset:
        image = dataset.read().astype(np.float32)
        transform = tuple(dataset.transform)[:6]
        bounds = (dataset.bounds.left, dataset.bounds.bottom, dataset.bounds.right, dataset.bounds.top)
        tags = {str(key): str(value) for key, value in dataset.tags().items()}
        crs = str(dataset.crs or "EPSG:4326")
    return SceneRaster(
        scene_id=_scene_id(path),
        source_path=path,
        image=image,
        transform=transform,
        crs=crs,
        bounds=bounds,
        acquisition_date=infer_acquisition_date(path, tags),
        sensor=infer_sensor(path, tags),
        metadata=tags,
    )


def _read_with_pillow(path: Path) -> SceneRaster:
    image = np.asarray(Image.open(path)).astype(np.float32)
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    else:
        image = np.moveaxis(image, -1, 0)
    height, width = image.shape[-2:]
    sidecar_path = path.with_suffix(".geo.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
    transform = tuple(sidecar.get("transform", [1.0, 0.0, 0.0, 0.0, -1.0, float(height)]))
    bounds = tuple(sidecar.get("bounds", tile_bounds(transform, 0, 0, height, width)))
    metadata = dict(sidecar.get("metadata", {}))
    return SceneRaster(
        scene_id=_scene_id(path),
        source_path=path,
        image=image,
        transform=transform,
        crs=str(sidecar.get("crs", "EPSG:4326")),
        bounds=bounds,
        acquisition_date=infer_acquisition_date(path, metadata),
        sensor=infer_sensor(path, metadata),
        metadata=metadata,
    )


def read_scene(path: str | Path) -> SceneRaster:
    source_path = Path(path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    try:
        return _read_with_rasterio(source_path)
    except (ModuleNotFoundError, RuntimeError, ValueError):
        return _read_with_pillow(source_path)


def tile_bounds(transform: tuple[float, float, float, float, float, float], row: int, col: int, height: int, width: int) -> tuple[float, float, float, float]:
    a, b, c, d, e, f = transform
    points = [(col, row), (col + width, row), (col, row + height), (col + width, row + height)]
    coordinates = [(a * x + b * y + c, d * x + e * y + f) for x, y in points]
    xs, ys = zip(*coordinates)
    return min(xs), min(ys), max(xs), max(ys)


def to_wgs84_bounds(bounds: tuple[float, float, float, float], crs: str) -> tuple[float, float, float, float]:
    if crs.upper() in {"EPSG:4326", "OGC:CRS84"}:
        return bounds
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        points = [transformer.transform(x, y) for x in [bounds[0], bounds[2]] for y in [bounds[1], bounds[3]]]
        xs, ys = zip(*points)
        return min(xs), min(ys), max(xs), max(ys)
    except Exception:
        return bounds


def tile_scene(scene: SceneRaster, tile_size: int = 256, stride: int | None = None) -> Iterator[TilePatch]:
    step = stride or tile_size
    channels, height, width = scene.image.shape
    for row in range(0, height, step):
        for col in range(0, width, step):
            patch = scene.image[:, row : min(row + tile_size, height), col : min(col + tile_size, width)]
            if patch.shape[1] < tile_size or patch.shape[2] < tile_size:
                if scene.metadata.get("resample_to_tile_size"):
                    rows = np.rint(np.linspace(0, patch.shape[1] - 1, tile_size)).astype(int)
                    columns = np.rint(np.linspace(0, patch.shape[2] - 1, tile_size)).astype(int)
                    patch = patch[:, rows][:, :, columns]
                else:
                    padded = np.full((channels, tile_size, tile_size), np.nan, dtype=np.float32)
                    padded[:, : patch.shape[1], : patch.shape[2]] = patch
                    patch = padded
            tile_transform = (
                scene.transform[0],
                scene.transform[1],
                scene.transform[0] * col + scene.transform[1] * row + scene.transform[2],
                scene.transform[3],
                scene.transform[4],
                scene.transform[3] * col + scene.transform[4] * row + scene.transform[5],
            )
            tile_id = f"{scene.scene_id}/{row // step}_{col // step}"
            yield TilePatch(
                tile_id=tile_id,
                scene_id=scene.scene_id,
                image=patch,
                grid_row=row // step,
                grid_col=col // step,
                transform=tile_transform,
                crs=scene.crs,
                bounds=to_wgs84_bounds(tile_bounds(tile_transform, 0, 0, tile_size, tile_size), scene.crs) if scene.bounds else (),
            )


def save_tile(tile: TilePatch, directory: str | Path, preview_image: np.ndarray | None = None) -> tuple[Path, Path]:
    root = Path(directory) / tile.scene_id
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{tile.grid_row}_{tile.grid_col}"
    array_path = root / f"{stem}.npy"
    preview_path = root / f"{stem}.png"
    np.save(array_path, tile.image.astype(np.float32))
    source_preview = preview_image if preview_image is not None else tile.image
    preview = source_preview[: min(3, source_preview.shape[0])]
    if preview.shape[0] == 1:
        preview = np.repeat(preview, 3, axis=0)
    if preview.shape[0] == 2:
        preview = np.concatenate([preview, preview[:1]], axis=0)
    display = np.moveaxis(preview[:3], 0, -1)
    scaled = np.zeros_like(display, dtype=np.float32)
    for channel in range(display.shape[-1]):
        band = display[:, :, channel]
        finite = band[np.isfinite(band)]
        if finite.size:
            low, high = np.percentile(finite, [2, 98])
            if high <= low:
                low, high = float(finite.min()), float(finite.max())
            if high > low:
                scaled[:, :, channel] = np.clip((np.nan_to_num(band, nan=low) - low) / (high - low), 0, 1)
            else:
                scaled[:, :, channel] = 0.5
    Image.fromarray((np.nan_to_num(scaled) * 255).astype(np.uint8)).save(preview_path)
    return array_path, preview_path


def save_scene_preview(scene: SceneRaster, directory: str | Path) -> Path:
    """Save a display-only preview of the complete source scene."""
    root = Path(directory) / scene.scene_id
    root.mkdir(parents=True, exist_ok=True)
    preview_path = root / "scene.png"
    source = scene.image[: min(3, scene.image.shape[0])]
    if source.shape[0] == 1:
        source = np.repeat(source, 3, axis=0)
    if source.shape[0] == 2:
        source = np.concatenate([source, source[:1]], axis=0)
    display = np.moveaxis(source[:3], 0, -1)
    scaled = np.zeros_like(display, dtype=np.float32)
    for channel in range(display.shape[-1]):
        band = display[:, :, channel]
        finite = band[np.isfinite(band)]
        if finite.size:
            low, high = np.percentile(finite, [2, 98])
            if high > low:
                scaled[:, :, channel] = np.clip((np.nan_to_num(band, nan=low) - low) / (high - low), 0, 1)
            else:
                scaled[:, :, channel] = 0.5
    Image.fromarray((scaled * 255).astype(np.uint8)).save(preview_path)
    return preview_path
