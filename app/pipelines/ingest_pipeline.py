from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Any, Callable
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from app.embeddings.text_image_encoder import TextImageEncoder
from app.embeddings.visual_encoder import VisualEncoder
from app.index.metadata_store import MetadataStore
from app.index.qdrant_store import QdrantStore
from app.ingestion.coregister import coregister
from app.ingestion.descriptions import describe_scene, describe_tile
from app.ingestion.normalize import histogram_match, normalize_reflectance
from app.ingestion.quality_mask import apply_quality_mask, quality_mask, valid_fraction
from app.ingestion.captioning import LocalImageCaptioner
from app.ingestion.tiler import TilePatch, read_scene, save_scene_preview, save_tile
from app.settings import Settings
from app.utils.provenance import build_provenance, processing_step


ProgressCallback = Callable[[str], None]
SUPPORTED_IMAGE_EXTENSIONS = {".tif", ".tiff", ".cog", ".png", ".jpg", ".jpeg", ".jp2", ".webp"}


class IngestPipeline:
    def __init__(
        self,
        settings: Settings,
        metadata: MetadataStore,
        vectors: QdrantStore,
        text_encoder: TextImageEncoder,
        visual_encoder: VisualEncoder,
    ):
        self.settings = settings
        self.metadata = metadata
        self.vectors = vectors
        self.text_encoder = text_encoder
        self.visual_encoder = visual_encoder
        self.captioner = LocalImageCaptioner(settings.model_path("blip"), settings.offline_mode)

    @staticmethod
    def _stable_id(value: str) -> str:
        return str(uuid5(NAMESPACE_URL, value))

    def ingest_scene(
        self,
        source_path: str | Path,
        source_id: str | None = None,
        progress: ProgressCallback | None = None,
        trigger_change: bool = True,
    ) -> dict[str, Any]:
        emit = progress or (lambda _: None)
        source = Path(source_path).resolve()
        emit(f"Reading {source.name}")
        scene = read_scene(source)
        scene_source_id = source_id or scene.metadata.get("source_id") or source.stem
        scene_description, semantic_tags = describe_scene(source, scene.acquisition_date, scene.sensor, scene.metadata)
        caption = self.captioner.caption(scene.image)
        if caption:
            scene_description = caption[0].upper() + caption[1:].rstrip(".") + "."
        scene_preview = save_scene_preview(scene, self.settings.tile_dir)
        scene.metadata = {**scene.metadata, "description": scene_description, "semantic_tags": semantic_tags}
        scene.metadata["preview_path"] = str(scene_preview)
        existing = self.metadata.scene_by_source(scene_source_id, scene.acquisition_date)
        if existing and Path(existing["source_path"]).resolve() != source:
            location_suffix = hashlib.sha1(str(source.parent).encode("utf-8")).hexdigest()[:8]
            scene_source_id = f"{scene_source_id}-{location_suffix}"
            existing = self.metadata.scene_by_source(scene_source_id, scene.acquisition_date)
        if existing:
            emit(f"Skipped duplicate {scene_source_id} acquired {scene.acquisition_date}")
            return {"status": "duplicate", "scene": existing, "tiles_indexed": 0, "change_events": []}
        scene_id = self._stable_id(f"scene:{scene_source_id}:{scene.acquisition_date}")
        scene_provenance = build_provenance(
            [scene_source_id],
            [processing_step("raster_read", details={"path": str(source)})],
        )
        stored_scene = self.metadata.add_scene(
            {
                "id": scene_id,
                "source_id": scene_source_id,
                "source_path": str(source),
                "sensor": scene.sensor,
                "acquisition_date": scene.acquisition_date,
                "crs": scene.crs,
                "bounds": scene.bounds,
                "transform": scene.transform,
                "metadata": scene.metadata,
                "provenance": scene_provenance,
            }
        )

        emit(f"Embedding {source.name} as single scene unit")
        channels, height, width = scene.image.shape
        if scene.metadata.get("skip_quality_mask"):
            mask = np.all(np.isfinite(scene.image), axis=0)
            processed = normalize_reflectance(scene.image)
        else:
            mask = quality_mask(scene.image, scene.sensor, self.settings.cloud_probability_threshold)
            processed = normalize_reflectance(apply_quality_mask(scene.image, mask))

        previous_tiles = self.metadata.tiles_at_grid(0, 0, end_date=scene.acquisition_date)
        previous_tiles = [t for t in previous_tiles if t["scene_id"] != scene_id]
        if previous_tiles:
            reference = np.load(previous_tiles[-1]["tile_path"])
            if reference.shape == processed.shape:
                registered = coregister(reference, processed)
                processed = histogram_match(registered.aligned, reference)

        tile = TilePatch(
            tile_id=f"{scene_id}/0_0",
            scene_id=scene_id,
            image=processed,
            grid_row=0,
            grid_col=0,
            transform=scene.transform,
            crs=scene.crs,
            bounds=scene.bounds,
        )
        array_path, preview_path = save_tile(tile, self.settings.tile_dir, preview_image=scene.image)
        tile_description = describe_tile(scene_description, processed, 0, 0)
        semantic = self.text_encoder.encode_image(processed)
        visual = self.visual_encoder.encode_image(processed)
        tile_id = self._stable_id(f"tile:{scene_id}:0:0")
        semantic_point_id = self._stable_id(f"semantic:{tile_id}")
        visual_point_id = self._stable_id(f"visual:{tile_id}")
        tile_provenance = build_provenance(
            [scene_source_id],
            [
                processing_step("quality_mask", details={"valid_fraction": valid_fraction(mask)}),
                processing_step("radiometric_normalization"),
                processing_step("coregistration" if previous_tiles else "coregistration_skipped"),
                processing_step("semantic_embedding", details={"backend": semantic.backend}),
                processing_step("visual_embedding", details={"backend": visual.backend}),
            ],
            {"semantic": semantic.backend, "visual": visual.backend},
        )
        payload = {
            "tile_id": tile_id,
            "scene_id": scene_id,
            "source_id": scene_source_id,
            "sensor": scene.sensor,
            "acquisition_date": scene.acquisition_date,
            "bounds": list(scene.bounds) if scene.bounds else [],
            "crs": scene.crs,
            "transform": list(scene.transform),
            "grid_row": 0,
            "grid_col": 0,
            "valid_fraction": valid_fraction(mask),
            "preview_path": str(scene_preview),
            "semantic_tags": semantic_tags,
            "description": tile_description,
        }
        self.metadata.add_tile(
            {
                "id": tile_id,
                "scene_id": scene_id,
                "grid_row": 0,
                "grid_col": 0,
                "tile_path": str(array_path),
                "preview_path": str(scene_preview),
                "crs": scene.crs,
                "bounds": scene.bounds,
                "transform": scene.transform,
                "valid_fraction": valid_fraction(mask),
                "semantic_point_id": semantic_point_id,
                "visual_point_id": visual_point_id,
                "description": tile_description,
                "provenance": tile_provenance,
            }
        )
        self.vectors.upsert_many(self.settings.semantic_collection, [(semantic_point_id, semantic.vector, payload)])
        self.vectors.upsert_many(self.settings.visual_collection, [(visual_point_id, visual.vector, payload)])

        change_events: list[dict[str, Any]] = []
        if trigger_change:
            emit("Running automatic change detection against prior coverage")
            from app.pipelines.change_pipeline import ChangePipeline
            change_events = ChangePipeline(self.settings, self.metadata, self.vectors).analyze_new_scene(scene_id)

        emit(f"Indexed {scene_source_id}")
        return {"status": "indexed", "scene": stored_scene, "tiles_indexed": 1, "change_events": change_events}

    @staticmethod
    def _supported_files(source: Path) -> list[Path]:
        if source.is_file():
            return [source] if source.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else []
        return [
            path for path in sorted(source.rglob("*"))
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

    def _extract_zip(self, archive: Path, emit: ProgressCallback) -> Path:
        digest = hashlib.sha256(
            f"{archive.resolve()}:{archive.stat().st_mtime_ns}:{archive.stat().st_size}".encode("utf-8")
        ).hexdigest()[:16]
        destination = self.settings.import_dir / f"{archive.stem}-{digest}"
        allowed = SUPPORTED_IMAGE_EXTENSIONS | {".json"}
        with zipfile.ZipFile(archive) as bundle:
            members = [
                item for item in bundle.infolist()
                if not item.is_dir() and Path(item.filename).suffix.lower() in allowed
            ]
            total_size = sum(item.file_size for item in members)
            if len(members) > 10_000 or total_size > 5 * 1024 ** 3:
                raise ValueError("ZIP import is too large. Use fewer than 10,000 files and less than 5 GB uncompressed.")
            if not any(Path(item.filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS for item in members):
                raise ValueError("The ZIP contains no supported imagery files.")
            emit(f"Extracting {len(members)} supported file(s) from {archive.name}")
            for item in members:
                relative = Path(item.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("ZIP contains an unsafe file path.")
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("ZIP contains a symbolic link, which is not supported.")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)
        return destination

    def ingest_path(
        self,
        source_path: str | Path,
        source_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        emit = progress or (lambda _: None)
        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Path does not exist: {source}")
        if source.is_file() and source.suffix.lower() == ".zip":
            source = self._extract_zip(source, emit)
        files = self._supported_files(source)
        if not files:
            return {
                "status": "empty",
                "files_found": 0,
                "files_indexed": 0,
                "duplicates": 0,
                "tiles_indexed": 0,
                "change_events": [],
                "errors": ["No supported image files were found. Use GeoTIFF, COG, PNG, JPG, JP2, or WEBP files."],
            }
        results: list[dict[str, Any]] = []
        errors: list[str] = []
        total = len(files)
        for index, file_path in enumerate(files, start=1):
            emit(f"Ingesting {index}/{total}: {file_path.name}")
            try:
                results.append(
                    self.ingest_scene(
                        file_path,
                        source_id=source_id if total == 1 else None,
                        progress=emit,
                    )
                )
            except Exception as error:
                errors.append(f"{file_path.name}: {error}")
        indexed = [r for r in results if r["status"] == "indexed"]
        duplicates = [r for r in results if r["status"] == "duplicate"]
        return {
            "status": "indexed" if indexed and not errors else "partial" if indexed or duplicates else "failed",
            "files_found": total,
            "files_indexed": len(indexed),
            "duplicates": len(duplicates),
            "tiles_indexed": sum(r["tiles_indexed"] for r in indexed),
            "change_events": [ev for r in indexed for ev in r["change_events"]],
            "errors": errors,
        }

    def ingest_directory(
        self,
        source_directory: str | Path,
        progress: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        source = Path(source_directory)
        return [self.ingest_scene(path, progress=progress) for path in self._supported_files(source)]
