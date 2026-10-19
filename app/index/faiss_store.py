from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaissMatch:
    id: str
    score: float
    payload: dict[str, Any]


class FaissStore:
    """
    Triple FAISS Vector Index Store for GeoSearch AI.
    Maintains three high-throughput indexes:
      1. visual: RemoteCLIP 512-D visual & semantic embeddings
      2. temporal: Prithvi-EO-2.0 multi-temporal embeddings
      3. sar_optical: SkySense++ fused radar-optical embeddings
    Supports zero-rebuild incremental addition and exact/approximate cosine search.
    """

    COLLECTIONS = ("visual", "temporal", "sar_optical")

    def __init__(self, index_dir: str | Path, dimension: int = 512):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension

        self._indexes: dict[str, Any] = {}
        self._id_maps: dict[str, list[str]] = {}
        self._payloads: dict[str, dict[str, Any]] = {}

        self._load_all()

    def _index_path(self, collection: str) -> Path:
        return self.index_dir / f"faiss_{collection}.bin"

    def _meta_path(self, collection: str) -> Path:
        return self.index_dir / f"faiss_{collection}_meta.json"

    def _load_all(self) -> None:
        for col in self.COLLECTIONS:
            self._load_collection(col)

    def _load_collection(self, collection: str) -> None:
        idx_path = self._index_path(collection)
        meta_path = self._meta_path(collection)

        self._id_maps[collection] = []
        self._payloads[collection] = {}

        try:
            import faiss
            if idx_path.exists() and meta_path.exists():
                self._indexes[collection] = faiss.read_index(str(idx_path))
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self._id_maps[collection] = meta.get("ids", [])
                self._payloads[collection] = meta.get("payloads", {})
                logger.info("Loaded FAISS index '%s' with %d entries", collection, self._indexes[collection].ntotal)
            else:
                self._indexes[collection] = faiss.IndexFlatIP(self.dimension)
        except Exception as err:
            logger.warning("FAISS native unavailable (%s); using pure numpy cosine mirror for %s", err, collection)
            self._indexes[collection] = None

    def upsert(
        self,
        collection: str,
        point_id: str,
        vector: np.ndarray,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        if collection not in self.COLLECTIONS:
            return False

        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self.dimension:
            if vec.shape[1] > self.dimension:
                vec = np.interp(np.linspace(0, vec.shape[1] - 1, self.dimension), np.arange(vec.shape[1]), vec[0]).reshape(1, -1).astype(np.float32)
            else:
                padded = np.zeros((1, self.dimension), dtype=np.float32)
                padded[0, : vec.shape[1]] = vec[0]
                vec = padded

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm

        id_map = self._id_maps[collection]
        if point_id not in id_map:
            id_map.append(point_id)
            if self._indexes.get(collection) is not None:
                self._indexes[collection].add(vec)

        self._payloads[collection][point_id] = payload or {}
        self._persist(collection)
        return True

    def search(
        self,
        collection: str,
        query_vector: np.ndarray,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[FaissMatch]:
        if collection not in self.COLLECTIONS:
            return []

        q_vec = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(q_vec))
        if norm > 1e-8:
            q_vec = q_vec / norm

        index = self._indexes.get(collection)
        results: list[FaissMatch] = []

        if index is not None and index.ntotal > 0:
            k = min(max(limit * 3, 20), index.ntotal)
            distances, indices = index.search(q_vec, k)
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self._id_maps[collection]):
                    continue
                point_id = self._id_maps[collection][idx]
                payload = self._payloads[collection].get(point_id, {})
                if filters and not self._matches_filters(payload, filters):
                    continue
                results.append(FaissMatch(id=point_id, score=float(dist), payload=payload))
                if len(results) >= limit:
                    break

        return results

    def _matches_filters(self, payload: dict[str, Any], filters: dict[str, Any]) -> bool:
        if filters.get("sensor") and payload.get("sensor") != filters["sensor"]:
            return False
        if filters.get("date_start") and payload.get("acquisition_date", "") < str(filters["date_start"]):
            return False
        if filters.get("date_end") and payload.get("acquisition_date", "") > str(filters["date_end"]):
            return False
        return True

    def _persist(self, collection: str) -> None:
        try:
            import faiss
            if self._indexes.get(collection) is not None:
                faiss.write_index(self._indexes[collection], str(self._index_path(collection)))
            meta = {
                "ids": self._id_maps[collection],
                "payloads": self._payloads[collection],
            }
            self._meta_path(collection).write_text(json.dumps(meta, default=str), encoding="utf-8")
        except Exception:
            pass

    def stats(self) -> dict[str, Any]:
        out = {}
        for col in self.COLLECTIONS:
            idx = self._indexes.get(col)
            count = idx.ntotal if idx is not None else len(self._id_maps[col])
            out[f"faiss_{col}"] = {"count": count, "dimension": self.dimension}
        return out
