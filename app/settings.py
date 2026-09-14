from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        key, separator, raw_value = raw_line.strip().partition(":")
        if not separator:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        value = raw_value.strip().strip("'\"")
        if value:
            current[key] = _parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
    return root


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded or {}
    except ModuleNotFoundError:
        return _simple_yaml(path.read_text(encoding="utf-8"))


@dataclass(slots=True)
class Settings:
    root_dir: Path
    project_name: str
    data_dir: Path
    raw_dir: Path
    tile_dir: Path
    model_dir: Path
    index_dir: Path
    qdrant_path: Path
    database_url: str
    tile_size: int = 256
    tile_stride: int = 256
    embedding_dimension: int = 256
    semantic_collection: str = "imagery_semantic"
    visual_collection: str = "imagery_visual"
    confidence_threshold: float = 0.68
    cloud_probability_threshold: float = 0.35
    minimum_valid_fraction: float = 0.7
    minimum_retrieval_score: float = 0.3
    offline_mode: bool = True
    default_aoi: dict[str, float] = field(default_factory=dict)
    default_date_start: str = "2023-01-01"
    default_date_end: str = "2024-01-01"
    model_paths: dict[str, str] = field(default_factory=dict)
    model_registry: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def database_path(self) -> Path:
        prefix = "sqlite:///"
        if self.database_url.startswith(prefix):
            return self.resolve_path(self.database_url.removeprefix(prefix))
        return self.index_dir / "metadata.db"

    @property
    def import_dir(self) -> Path:
        """Managed extraction location for ZIP imports supplied through the local UI/API."""
        return self.data_dir / "imports"

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root_dir / path

    def model_path(self, name: str) -> Path:
        return self.resolve_path(self.model_paths.get(name, str(self.model_dir / name)))

    def ensure_directories(self) -> None:
        for path in [self.data_dir, self.raw_dir, self.tile_dir, self.model_dir, self.index_dir, self.qdrant_path, self.import_dir]:
            path.mkdir(parents=True, exist_ok=True)


def load_settings(config_path: str | Path | None = None) -> Settings:
    root_dir = Path(__file__).resolve().parents[1]
    source = Path(config_path or os.getenv("SAT_CHANGE_CONFIG", root_dir / "config" / "config.yaml"))
    source = source if source.is_absolute() else root_dir / source
    config = _load_yaml(source)
    environment = {
        "data_dir": os.getenv("SAT_CHANGE_DATA_DIR"),
        "qdrant_path": os.getenv("SAT_CHANGE_QDRANT_PATH"),
        "database_url": os.getenv("SAT_CHANGE_DATABASE_URL"),
        "offline_mode": os.getenv("SAT_CHANGE_OFFLINE"),
    }
    config.update({key: _parse_scalar(value) for key, value in environment.items() if value is not None})
    data_dir = root_dir / str(config.get("data_dir", "data"))
    settings = Settings(
        root_dir=root_dir,
        project_name=str(config.get("project_name", "Satellite Imagery Search")),
        data_dir=data_dir,
        raw_dir=root_dir / str(config.get("raw_dir", "data/raw")),
        tile_dir=root_dir / str(config.get("tile_dir", "data/tiles")),
        model_dir=root_dir / str(config.get("model_dir", "data/models")),
        index_dir=root_dir / str(config.get("index_dir", "data/index")),
        qdrant_path=root_dir / str(config.get("qdrant_path", "data/index/qdrant")),
        database_url=str(config.get("database_url", "sqlite:///data/index/metadata.db")),
        tile_size=int(config.get("tile_size", 256)),
        tile_stride=int(config.get("tile_stride", 256)),
        embedding_dimension=int(config.get("embedding_dimension", 256)),
        semantic_collection=str(config.get("semantic_collection", "imagery_semantic")),
        visual_collection=str(config.get("visual_collection", "imagery_visual")),
        confidence_threshold=float(config.get("confidence_threshold", 0.68)),
        cloud_probability_threshold=float(config.get("cloud_probability_threshold", 0.35)),
        minimum_valid_fraction=float(config.get("minimum_valid_fraction", 0.7)),
        minimum_retrieval_score=float(config.get("minimum_retrieval_score", 0.3)),
        offline_mode=bool(config.get("offline_mode", True)),
        default_aoi=dict(config.get("default_aoi", {})),
        default_date_start=str(config.get("default_date_start", "2023-01-01")),
        default_date_end=str(config.get("default_date_end", "2024-01-01")),
        model_paths=dict(config.get("model_paths", {})),
        model_registry=dict(config.get("model_registry", {})),
    )
    settings.ensure_directories()
    return settings
