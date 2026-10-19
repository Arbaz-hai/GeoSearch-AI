from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.conversation.query_parser import QueryParser
from app.embeddings.text_image_encoder import TextImageEncoder
from app.embeddings.visual_encoder import VisualEncoder
from app.index.metadata_store import MetadataStore
from app.index.qdrant_store import QdrantStore
from app.settings import Settings
from app.utils.provenance import build_provenance, processing_step


class SearchPipeline:
    """
    Semantic search pipeline integrating:
    - Qwen3 NLP query understanding
    - CLIP text & image embeddings
    - Qdrant vector search
    - Scene-level result deduplication
    - Multi-signal ranking (semantic + metadata + temporal + quality)
    - Feedback-driven reranking
    """

    def __init__(
        self,
        settings: Settings,
        metadata: MetadataStore,
        vectors: QdrantStore,
        text_encoder: TextImageEncoder,
        visual_encoder: VisualEncoder,
        query_parser: QueryParser | None = None,
    ):
        self.settings = settings
        self.metadata = metadata
        self.vectors = vectors
        self.text_encoder = text_encoder
        self.visual_encoder = visual_encoder
        self.query_parser = query_parser


    def search_text(self, query: str, limit: int = 20, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self.search_text_with_status(query, limit, filters)["results"]

    def search_text_with_status(self, query: str, limit: int = 20, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        parsed_query: dict[str, Any] = {}
        query_interpretation = ""
        if self.query_parser is not None:
            parsed_query = self.query_parser.parse(query)
            query_interpretation = self.query_parser.describe_parse(parsed_query)
            filters = dict(filters or {})
            temporal = parsed_query.get("temporal", {})
            if temporal.get("after") and not filters.get("date_start"):
                filters["date_start"] = temporal["after"] + "-01-01"
            if temporal.get("before") and not filters.get("date_end"):
                filters["date_end"] = temporal["before"] + "-12-31"
            if temporal.get("from") and not filters.get("date_start"):
                filters["date_start"] = temporal["from"] + "-01-01"
            if temporal.get("to") and not filters.get("date_end"):
                filters["date_end"] = temporal["to"] + "-12-31"

        embedding = self.text_encoder.encode_text(query)

        raw_matches = self.vectors.search(
            self.settings.semantic_collection,
            embedding.vector,
            max(limit * 6, 120),
            filters,
        )
        print(f"[SEARCH_DEBUG] query='{query}' raw_matches={len(raw_matches)} filters={filters}", flush=True)

        lexical = self._lexical_candidates(query, filters)
        print(f"[SEARCH_DEBUG] lexical={len(lexical)}", flush=True)

        candidates: dict[str, dict[str, Any]] = {}
        for match in raw_matches:
            tile_id = match["payload"].get("tile_id")
            if tile_id:
                candidates[tile_id] = match

        for tile, scene, lex_score in lexical:
            existing = candidates.get(tile["id"])
            if existing:
                existing["score"] += 0.15 * lex_score
            else:
                payload = {
                    "tile_id": tile["id"],
                    "description": tile.get("description", ""),
                }
                candidates[tile["id"]] = {"id": tile["semantic_point_id"], "score": 0.65 + 0.15 * lex_score, "payload": payload}

        sorted_matches = sorted(candidates.values(), key=lambda m: m["score"], reverse=True)
        print(f"[SEARCH_DEBUG] candidates={len(candidates)} calling _format_results_scene_level", flush=True)
        results = self._format_results_scene_level(sorted_matches, "text", embedding.backend, query)[:limit]
        print(f"[SEARCH_DEBUG] formatted results={len(results)}", flush=True)

        if results:
            status = f"Found **{len(results)}** result(s)."
            if query_interpretation:
                status = f"{query_interpretation}  \n{status}"
            return {"results": results, "status": status, "matched": True, "query_parsed": parsed_query}

        return {
            "results": [],
            "status": "No matching imagery is indexed yet. Ingest a file, folder, or ZIP then try again.",
            "matched": False,
            "query_parsed": parsed_query,
        }


    def search_image(
        self,
        image: str | Path | np.ndarray | Image.Image,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.search_image_with_status(image, limit, filters)["results"]

    def search_image_with_status(
        self,
        image: str | Path | np.ndarray | Image.Image,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        arr = self._image_array(image)
        embedding = self.text_encoder.encode_image(arr)
        raw_matches = self.vectors.search(
            self.settings.semantic_collection, embedding.vector, limit * 4, filters
        )
        filtered = [m for m in raw_matches if m["score"] >= self.settings.minimum_retrieval_score]
        results = self._format_results_scene_level(filtered, "image", embedding.backend, None)[:limit]

        if results:
            return {
                "results": results,
                "status": f"Found **{len(results)}** visually similar image(s).",
                "matched": True,
            }
        return {
            "results": [],
            "status": "No close visual match in the local archive. Try ingesting more imagery.",
            "matched": False,
        }


    def find_similar_to_scene(self, scene_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Find scenes visually similar to a given scene (excluding itself)."""
        tiles = self.metadata.list_tiles(scene_id)
        if not tiles:
            return []
        tile = tiles[0]
        point = self.vectors.get(self.settings.semantic_collection, tile["semantic_point_id"])
        if point is None:
            return []
        vector = np.asarray(point["vector"], dtype=np.float32)
        raw_matches = self.vectors.search(self.settings.semantic_collection, vector, limit * 4)
        results = self._format_results_scene_level(raw_matches, "seed", self.vectors.backend, None)
        return [r for r in results if r["scene_id"] != scene_id][:limit]

    def find_similar_tile(self, tile_id: str, limit: int = 20) -> list[dict[str, Any]]:
        tile = self.metadata.get_tile(tile_id)
        if tile is None:
            raise KeyError(tile_id)
        point = self.vectors.get(self.settings.visual_collection, tile["visual_point_id"])
        if point is None:
            return []
        vector = np.asarray(point["vector"], dtype=np.float32)
        raw_matches = self.vectors.search(self.settings.visual_collection, vector, limit + 1)
        results = self._format_results_scene_level(raw_matches, "seed", self.vectors.backend, None)
        return [r for r in results if r.get("tile_id") != tile_id][:limit]


    @staticmethod
    def _image_array(image: str | Path | np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(image, np.ndarray):
            values = image
        elif isinstance(image, Image.Image):
            values = np.asarray(image)
        else:
            values = np.asarray(Image.open(image))
        if values.ndim == 3:
            return np.moveaxis(values, -1, 0).astype(np.float32)
        return values[np.newaxis, :, :].astype(np.float32)

    def _format_results_scene_level(
        self,
        matches: list[dict[str, Any]],
        mode: str,
        backend: str,
        query: str | None,
    ) -> list[dict[str, Any]]:
        """
        Format and deduplicate results at the SCENE level.
        For each scene, keep only the highest-scoring tile as representative.
        """
        tile_cache: dict[str, dict[str, Any]] = {}
        scene_best: dict[str, dict[str, Any]] = {}

        for match in matches:
            tile_id = match["payload"].get("tile_id")
            if not tile_id or tile_id in tile_cache:
                continue
            tile_cache[tile_id] = match

            tile = self.metadata.get_tile(tile_id)
            if tile is None:
                print(f"[FORMAT_DEBUG] get_tile({tile_id}) is None!", flush=True)
                continue
            scene = self.metadata.get_scene(tile["scene_id"])
            if scene is None:
                print(f"[FORMAT_DEBUG] get_scene({tile['scene_id']}) is None!", flush=True)
                continue

            similarity = float(match["score"])
            feedback_bias = self.metadata.feedback_bias(tile_id)
            cloud_cover = float(scene["metadata"].get("cloud_cover", scene["metadata"].get("cloud_coverage", 0)) or 0)
            quality_score = max(0.0, 1.0 - cloud_cover / 100.0) * float(tile.get("valid_fraction", 1.0))
            combined_score = round(0.80 * similarity + 0.10 * quality_score + feedback_bias, 5)

            result = {
                "tile_id": tile["id"],
                "scene_id": scene["source_id"] if scene else tile["scene_id"],
                "scene_uuid": scene["id"] if scene else tile["scene_id"],
                "source_id": scene["source_id"],
                "source_name": Path(scene["source_path"]).name if scene.get("source_path") else scene["source_id"],
                "acquisition_date": scene["acquisition_date"],
                "acquisition_time": scene["metadata"].get("acquisition_time", ""),
                "sensor": scene["sensor"],
                "dataset": scene["metadata"].get("dataset", scene["sensor"]),
                "resolution": scene["metadata"].get("resolution", scene["metadata"].get("GSD", "")),
                "cloud_cover": cloud_cover,
                "crs": tile.get("crs", scene.get("crs", "EPSG:4326")),
                "bounds": tile["bounds"],
                "center_lat": _center(tile["bounds"], 1),
                "center_lon": _center(tile["bounds"], 0),
                "transform": tile.get("transform", []),
                "bands": scene["metadata"].get("bands", scene["metadata"].get("count", "")),
                "similarity": combined_score,
                "base_similarity": round(similarity, 5),
                "feedback_reranking_bias": round(feedback_bias, 5),
                "quality_score": round(quality_score, 3),
                "preview_path": scene["metadata"].get("preview_path") or tile["preview_path"],
                "tile_preview_path": tile["preview_path"],
                "description": (
                    tile.get("description")
                    or scene["metadata"].get("description")
                    or "No description available."
                ),
                "caption": scene["metadata"].get("caption", ""),
                "ingestion_time": scene["created_at"],
                "embedding_model": backend,
                "captioning_model": "BLIP" if scene["metadata"].get("caption") else "Not available",
                "source_path": scene.get("source_path", ""),
                "provenance": build_provenance(
                    [scene["source_id"]],
                    [processing_step("ann_search", details={"mode": mode})],
                    {"query_encoder": backend},
                    {"query_text": query},
                ),
            }

            existing = scene_best.get(scene["id"])
            if existing is None or combined_score > existing["similarity"]:
                scene_best[scene["id"]] = result

        ordered = sorted(scene_best.values(), key=lambda r: r["similarity"], reverse=True)
        return [{**r, "rank": rank} for rank, r in enumerate(ordered, start=1)]

    def _lexical_candidates(
        self, query: str, filters: dict[str, Any] | None
    ) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
        terms = [
            t for t in re.findall(r"[a-z0-9]+", query.lower())
            if len(t) > 2 and t not in {
                "image", "images", "imagery", "show", "find", "with", "from",
                "near", "the", "and", "for", "are", "that", "this",
            }
        ]
        if not terms:
            return []
        results: list[tuple[dict[str, Any], dict[str, Any], float]] = []
        for tile in self.metadata.list_tiles():
            scene = self.metadata.get_scene(tile["scene_id"])
            if scene is None or not self._matches_filters(tile, scene, filters or {}):
                continue
            haystack = " ".join([
                tile.get("description", ""),
                scene["source_id"],
                str(scene["metadata"].get("description", "")),
                str(scene["metadata"].get("caption", "")),
            ]).lower()
            matched = sum(1 for t in terms if t in haystack)
            if matched:
                results.append((tile, scene, matched / len(terms)))
        return sorted(results, key=lambda x: x[2], reverse=True)

    @staticmethod
    def _matches_filters(tile: dict[str, Any], scene: dict[str, Any], filters: dict[str, Any]) -> bool:
        if filters.get("sensor") and scene["sensor"] != filters["sensor"]:
            return False
        if filters.get("date_start") and scene["acquisition_date"] < str(filters["date_start"]):
            return False
        if filters.get("date_end") and scene["acquisition_date"] > str(filters["date_end"]):
            return False
        bbox = filters.get("bbox")
        bounds = tile.get("bounds", [])
        if bbox and len(bounds) == 4:
            if not (bounds[0] <= bbox[2] and bounds[2] >= bbox[0] and
                    bounds[1] <= bbox[3] and bounds[3] >= bbox[1]):
                return False
        return True


def _center(bounds: list[float], axis: int) -> float:
    """Return center longitude (axis=0) or latitude (axis=1) from bbox."""
    if not bounds or len(bounds) < 4:
        return 0.0
    return round((bounds[axis] + bounds[axis + 2]) / 2, 6)
