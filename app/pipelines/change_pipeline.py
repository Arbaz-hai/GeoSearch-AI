from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from app.embeddings.change_model import ChangeModel
from app.conversation.qwen_chat import LocalQwenChat
from app.index.metadata_store import MetadataStore
from app.index.qdrant_store import QdrantStore
from app.ingestion.coregister import coregister
from app.ingestion.quality_mask import quality_mask, valid_fraction
from app.ingestion.radiometric import histogram_match
from app.ingestion.shadow_mask import compute_shadow_mask, shadow_fraction
from app.ingestion.spectral_indices import SpectralIndices, compute_spectral_indices
from app.settings import Settings
from app.utils.provenance import build_provenance, processing_step


class ChangePipeline:
    def __init__(self, settings: Settings, metadata: MetadataStore, vectors: QdrantStore):
        self.settings = settings
        self.metadata = metadata
        self.vectors = vectors
        self.model = ChangeModel(settings.model_path("changeformer"), settings.offline_mode)
        self.chat = LocalQwenChat(settings.model_path("qwen3"), settings.offline_mode)
        self.mask_directory = settings.tile_dir / "change_masks"
        self.mask_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _overlaps(bounds: list[float], aoi: list[float] | None) -> bool:
        return aoi is None or (
            bounds[0] <= aoi[2] and bounds[2] >= aoi[0] and
            bounds[1] <= aoi[3] and bounds[3] >= aoi[1]
        )

    def _classify_change_type(
        self,
        before: SpectralIndices,
        after: SpectralIndices,
        fallback: str,
    ) -> str:
        ndvi_delta = round(after.ndvi_mean - before.ndvi_mean, 4)
        ndwi_delta = round(after.ndwi_mean - before.ndwi_mean, 4)
        ndbi_delta = round(after.ndbi_mean - before.ndbi_mean, 4)

        if ndbi_delta > 0.12 and ndvi_delta < -0.08:
            return "construction"
        if ndvi_delta < -0.20 and ndbi_delta < 0.05:
            return "clearance"
        if abs(ndwi_delta) > 0.15:
            return "water-extent"
        if ndvi_delta > 0.15:
            return "vegetation-expansion"
        return fallback

    def analyze_new_scene(self, scene_id: str) -> list[dict[str, Any]]:
        scene = self.metadata.get_scene(scene_id)
        if scene is None:
            raise KeyError(scene_id)
        results: list[dict[str, Any]] = []
        for after_tile in self.metadata.list_tiles(scene_id):
            history = self.metadata.tiles_at_grid(
                after_tile["grid_row"], after_tile["grid_col"],
                end_date=scene["acquisition_date"],
            )
            earlier = [t for t in history if t["scene_id"] != scene_id]
            if earlier:
                result = self._compare(earlier[-1], after_tile, history)
                if result:
                    results.append(result)
        return results

    def analyze(self, start_date: str, end_date: str, aoi: list[float] | None = None) -> list[dict[str, Any]]:
        groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for tile in self.metadata.list_tiles():
            scene = self.metadata.get_scene(tile["scene_id"])
            if (
                scene is None
                or not start_date <= scene["acquisition_date"] <= end_date
                or not self._overlaps(tile["bounds"], aoi)
            ):
                continue
            groups.setdefault((tile["grid_row"], tile["grid_col"]), []).append(tile)
        results: list[dict[str, Any]] = []
        for history in groups.values():
            history.sort(key=lambda t: (self.metadata.get_scene(t["scene_id"]) or {}).get("acquisition_date", ""))
            for before, after in zip(history, history[1:]):
                result = self._compare(before, after, history)
                if result:
                    results.append(result)
        return sorted(results, key=lambda e: e["confidence"], reverse=True)

    def _compare(
        self,
        before_tile: dict[str, Any],
        after_tile: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        before_path = Path(before_tile["tile_path"])
        after_path = Path(after_tile["tile_path"])
        if not before_path.exists() or not after_path.exists():
            return None

        before = np.load(before_path)
        after = np.load(after_path)

        # Harmonize spatial dimensions if slight resolution/cropping discrepancies exist
        min_h = min(before.shape[1], after.shape[1])
        min_w = min(before.shape[2], after.shape[2])
        before = before[:, :min_h, :min_w]
        after = after[:, :min_h, :min_w]

        before_scene = self.metadata.get_scene(before_tile["scene_id"])
        after_scene = self.metadata.get_scene(after_tile["scene_id"])
        if before_scene is None or after_scene is None:
            return None

        before_mask = quality_mask(before, before_scene["sensor"], self.settings.cloud_probability_threshold)
        after_mask = quality_mask(after, after_scene["sensor"], self.settings.cloud_probability_threshold)
        valid = before_mask & after_mask

        if valid_fraction(valid) < self.settings.minimum_valid_fraction:
            return None

        registration = coregister(before, after)
        aligned_after = registration.aligned
        if registration.quality >= 0.10:
            aligned_after = histogram_match(registration.aligned, before)

        prediction = self.model.predict(before, aligned_after, valid)

        before_point = self.vectors.get(self.settings.visual_collection, before_tile["visual_point_id"])
        after_point = self.vectors.get(self.settings.visual_collection, after_tile["visual_point_id"])
        embedding_distance = 0.0
        if before_point and after_point:
            left = np.asarray(before_point["vector"], dtype=np.float32)
            right = np.asarray(after_point["vector"], dtype=np.float32)
            denom = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-6)
            embedding_distance = float(np.clip((1 - np.dot(left, right) / denom) / 2, 0, 1))

        # Spectral Indices & Multi-Evidence Gate
        before_indices = compute_spectral_indices(before)
        after_indices = compute_spectral_indices(aligned_after)
        semantic_type = self._classify_change_type(before_indices, after_indices, prediction.change_type)

        # Shadow suppression gate
        after_shadow = compute_shadow_mask(after)
        shadow_frac = shadow_fraction(after_shadow)

        base_conf = float(np.clip(
            0.75 * prediction.confidence + 0.25 * (embedding_distance if embedding_distance > 0 else prediction.confidence),
            0,
            1
        ))
        if shadow_frac > 0.40:
            base_conf *= 0.85  # moderate downgrade

        if base_conf < 0.40:
            return None

        confidence = base_conf
        earliest_date = self._earliest_supported_date(history, before_tile, after_tile)

        key = f"{before_tile['id']}_{after_tile['id']}"
        mask_path = self.mask_directory / f"{key}.npy"
        np.save(mask_path, prediction.change_map)

        mask_preview_path = _render_mask_png(prediction.change_map, self.mask_directory / f"{key}_preview.png")

        mask_binary = prediction.change_map >= 0.55
        changed_area_pct = round(float(mask_binary.mean()) * 100, 2)

        provenance = build_provenance(
            [before_scene["source_id"], after_scene["source_id"]],
            [
                processing_step("quality_mask", details={"valid_fraction": valid_fraction(valid)}),
                processing_step("coregistration", details={
                    "shift_y": registration.shift_y,
                    "shift_x": registration.shift_x,
                    "quality": registration.quality,
                }),
                processing_step("spectral_verification", details={
                    "before_ndvi": before_indices.ndvi_mean,
                    "after_ndvi": after_indices.ndvi_mean,
                    "shadow_fraction": round(shadow_frac, 3),
                }),
                processing_step("change_detection", details={
                    "pixel_confidence": prediction.confidence,
                    "embedding_distance": embedding_distance,
                    "changed_area_pct": changed_area_pct,
                }),
            ],
            {"change_model": prediction.backend},
        )

        record = {
            "before_tile_id": before_tile["id"],
            "after_tile_id": after_tile["id"],
            "change_type": semantic_type,
            "confidence": confidence,
            "earliest_detected_date": earliest_date,
            "mask_path": str(mask_path),
            "mask_preview_path": str(mask_preview_path) if mask_preview_path else None,
            "changed_area_pct": changed_area_pct,
            "status": "pending",
            "provenance": provenance,
        }
        return self.metadata.add_change_event(record)

    def _earliest_supported_date(
        self,
        history: list[dict[str, Any]],
        baseline: dict[str, Any],
        latest: dict[str, Any],
    ) -> str:
        baseline_path = Path(baseline["tile_path"])
        if not baseline_path.exists():
            return (self.metadata.get_scene(latest["scene_id"]) or {}).get("acquisition_date", "")

        baseline_array = np.load(baseline_path)
        baseline_scene = self.metadata.get_scene(baseline["scene_id"])
        try:
            baseline_index = history.index(baseline)
        except ValueError:
            baseline_index = 0

        candidates = history[baseline_index + 1:]
        latest_scene = self.metadata.get_scene(latest["scene_id"])
        earliest = (latest_scene or {}).get("acquisition_date", "")

        for candidate in candidates:
            candidate_path = Path(candidate["tile_path"])
            if not candidate_path.exists():
                continue
            candidate_array = np.load(candidate_path)
            candidate_scene = self.metadata.get_scene(candidate["scene_id"])
            if candidate_scene is None or baseline_scene is None:
                continue

            mh = min(baseline_array.shape[1], candidate_array.shape[1])
            mw = min(baseline_array.shape[2], candidate_array.shape[2])
            b_arr = baseline_array[:, :mh, :mw]
            c_arr = candidate_array[:, :mh, :mw]

            valid = (
                quality_mask(b_arr, baseline_scene["sensor"], self.settings.cloud_probability_threshold)
                & quality_mask(c_arr, candidate_scene["sensor"], self.settings.cloud_probability_threshold)
            )
            if valid_fraction(valid) < 0.20:
                continue
            pred = self.model.predict(
                b_arr, coregister(b_arr, c_arr).aligned, valid
            )
            if pred.confidence >= 0.40:
                earliest = candidate_scene["acquisition_date"]
                break
        return earliest

    def review_queue(self, limit: int = 200) -> list[dict[str, Any]]:
        queue: list[dict[str, Any]] = []
        for event in self.metadata.pending_change_events(limit):
            before = self.metadata.get_tile(event["before_tile_id"])
            after = self.metadata.get_tile(event["after_tile_id"])
            before_scene = self.metadata.get_scene(before["scene_id"]) if before else None
            after_scene = self.metadata.get_scene(after["scene_id"]) if after else None

            mask_preview = event.get("mask_preview_path")
            if not mask_preview and event.get("mask_path"):
                preview_candidate = Path(event["mask_path"]).with_name(
                    Path(event["mask_path"]).stem + "_preview.png"
                )
                if preview_candidate.exists():
                    mask_preview = str(preview_candidate)

            queue.append({
                **event,
                "before_preview": before["preview_path"] if before else None,
                "after_preview": after["preview_path"] if after else None,
                "mask_preview": mask_preview,
                "changed_area_pct": event.get("changed_area_pct", 0),
                "location": after["bounds"] if after else [],
                "before_date": before_scene["acquisition_date"] if before_scene else "",
                "after_date": after_scene["acquisition_date"] if after_scene else "",
                "sensor": after_scene["sensor"] if after_scene else "",
                "before_sensor": before_scene["sensor"] if before_scene else "",
            })
        return queue

    def answer_question(self, question: str, events: list[dict[str, Any]]) -> str:
        """Answer only from currently calculated change evidence."""
        if not events:
            return (
                "No change candidates were found for this period. "
                "Try a wider date range or ingest another observation."
            )
        generated = self.chat.answer(question, events)
        if generated:
            return generated

        text = question.strip().lower()
        count = len(events)
        types = Counter(e["change_type"].replace("_", " ") for e in events)
        earliest = min(e["earliest_detected_date"] for e in events)
        strongest = max(events, key=lambda e: e["confidence"])
        type_summary = ", ".join(f"{n} {t}" for t, n in types.most_common(3))

        if any(w in text for w in {"how many", "count", "number"}):
            return f"The comparison found **{count}** candidate change(s): {type_summary}."
        if any(w in text for w in {"when", "date", "earliest", "time"}):
            return (
                f"The earliest supported change date is **{earliest}**. "
                f"The strongest candidate is **{strongest['change_type'].replace('_', ' ')}** "
                f"with **{strongest['confidence']:.0%}** confidence."
            )
        if any(w in text for w in {"confidence", "certain", "strongest", "reliable"}):
            return (
                f"The strongest candidate is **{strongest['change_type'].replace('_', ' ')}** "
                f"at **{strongest['confidence']:.0%}** confidence. "
                f"Detected types: {type_summary}."
            )
        return (
            f"Based on **{count}** detected candidate(s) — types: {type_summary}. "
            f"Earliest supported change date: **{earliest}**. "
            f"Strongest: **{strongest['change_type'].replace('_', ' ')}** "
            f"at **{strongest['confidence']:.0%}**."
        )


def _render_mask_png(change_map: np.ndarray, output_path: Path) -> Path | None:
    """Render a change probability map as a color-coded PNG (green=unchanged, red=changed)."""
    try:
        from PIL import Image

        arr = np.asarray(change_map, dtype=np.float32)
        arr = np.nan_to_num(np.clip(arr, 0, 1), nan=0.0)
        h, w = arr.shape

        r = (arr * 255).astype(np.uint8)
        g = ((1 - arr) * 204).astype(np.uint8)
        b = np.zeros((h, w), dtype=np.uint8)
        alpha = np.full((h, w), 180, dtype=np.uint8)

        rgba = np.stack([r, g, b, alpha], axis=-1)
        Image.fromarray(rgba, mode="RGBA").save(output_path)
        return output_path
    except Exception:
        return None
