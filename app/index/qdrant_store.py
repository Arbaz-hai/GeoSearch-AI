from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _overlaps(left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...]) -> bool:
    return len(left) == 4 and len(right) == 4 and left[0] <= right[2] and left[2] >= right[0] and left[1] <= right[3] and left[3] >= right[1]


class QdrantStore:
    def __init__(self, path: str | Path, semantic_collection: str, visual_collection: str, dimension: int):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.semantic_collection = semantic_collection
        self.visual_collection = visual_collection
        self.dimension = dimension
        self.lock = threading.RLock()
        self.mirror_path = self.path / "local_vector_mirror.json"
        self.mirror = self._load_mirror()
        self.client: Any | None = None
        self.backend = "local-persistent-vector-store"
        self._open_qdrant()

    def _load_mirror(self) -> dict[str, dict[str, dict[str, Any]]]:
        if self.mirror_path.exists():
            try:
                return json.loads(self.mirror_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {self.semantic_collection: {}, self.visual_collection: {}}

    def _persist_mirror(self) -> None:
        temporary = self.mirror_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.mirror, separators=(",", ":")), encoding="utf-8")
        temporary.replace(self.mirror_path)

    def _open_qdrant(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models

            self.client = QdrantClient(path=str(self.path / "native"))
            for collection in [self.semantic_collection, self.visual_collection]:
                if not self.client.collection_exists(collection):
                    self.client.create_collection(
                        collection_name=collection,
                        vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
                        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
                    )
            self.backend = "qdrant-local-hnsw"
        except Exception:
            self.client = None

    def upsert(self, collection: str, point_id: str, vector: np.ndarray, payload: dict[str, Any]) -> None:
        self.upsert_many(collection, [(point_id, vector, payload)])

    def upsert_many(self, collection: str, points: list[tuple[str, np.ndarray, dict[str, Any]]]) -> None:
        if not points:
            return
        records = []
        for point_id, vector, payload in points:
            vector_values = np.asarray(vector, dtype=np.float32).reshape(-1)
            if vector_values.size != self.dimension:
                vector_values = np.interp(
                    np.linspace(0, vector_values.size - 1, self.dimension),
                    np.arange(vector_values.size),
                    vector_values,
                ).astype(np.float32)
                v_norm = float(np.linalg.norm(vector_values))
                if v_norm:
                    vector_values = vector_values / v_norm
            records.append((point_id, vector_values, payload))

        with self.lock:
            collection_records = self.mirror.setdefault(collection, {})
            for point_id, vector_values, payload in records:
                collection_records[point_id] = {"vector": vector_values.tolist(), "payload": payload}
            self._persist_mirror()
            if self.client is not None:
                try:
                    from qdrant_client.http.models import PointStruct

                    self.client.upsert(
                        collection_name=collection,
                        points=[PointStruct(id=point_id, vector=vector.tolist(), payload=payload) for point_id, vector, payload in records],
                        wait=True,
                    )
                except Exception:
                    self.client = None
                    self.backend = "local-persistent-vector-store"

    def search(self, collection: str, vector: np.ndarray, limit: int = 20, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if self.client is not None:
            try:
                native = self._native_search(collection, query, max(limit * 5, limit))
                filtered = [item for item in native if self._matches(item["payload"], filters or {})]
                if len(filtered) >= limit or not filters:
                    return filtered[:limit]
            except Exception:
                self.client = None
                self.backend = "local-persistent-vector-store"

        records = self.mirror.get(collection, {})
        candidates = [(point_id, record) for point_id, record in records.items() if self._matches(record["payload"], filters or {})]
        if not candidates:
            return []

        matrix = np.asarray([record["vector"] for _, record in candidates], dtype=np.float32)
        if matrix.ndim == 2 and matrix.shape[1] != query.size:
            query = np.interp(
                np.linspace(0, query.size - 1, matrix.shape[1]),
                np.arange(query.size),
                query,
            ).astype(np.float32)
            q_norm = float(np.linalg.norm(query))
            if q_norm:
                query = query / q_norm

        norms = np.linalg.norm(matrix, axis=1) * max(float(np.linalg.norm(query)), 1e-6)
        scores = (matrix @ query) / np.maximum(norms, 1e-6)
        order = np.argsort(scores)[::-1][:limit]
        return [{"id": candidates[index][0], "score": float(scores[index]), "payload": candidates[index][1]["payload"]} for index in order]

    def _native_search(self, collection: str, vector: np.ndarray, limit: int) -> list[dict[str, Any]]:
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(collection_name=collection, query=vector.tolist(), limit=limit, with_payload=True)
            points = response.points
        else:
            points = self.client.search(collection_name=collection, query_vector=vector.tolist(), limit=limit, with_payload=True)
        return [{"id": str(point.id), "score": float(point.score), "payload": dict(point.payload or {})} for point in points]

    def get(self, collection: str, point_id: str) -> dict[str, Any] | None:
        return self.mirror.get(collection, {}).get(point_id)

    def all_points(self, collection: str) -> list[dict[str, Any]]:
        return [{"id": point_id, **record} for point_id, record in self.mirror.get(collection, {}).items()]

    def point_ids(self, collection: str) -> set[str]:
        return set(self.mirror.get(collection, {}))

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "semantic_vectors": len(self.mirror.get(self.semantic_collection, {})),
            "visual_vectors": len(self.mirror.get(self.visual_collection, {})),
        }

    @staticmethod
    def _matches(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
        sensor = filters.get("sensor")
        if sensor and payload.get("sensor") != sensor:
            return False
        date = str(payload.get("acquisition_date", ""))
        if filters.get("date_start") and date < str(filters["date_start"]):
            return False
        if filters.get("date_end") and date > str(filters["date_end"]):
            return False
        bbox = filters.get("bbox")
        if bbox and not _overlaps(payload.get("bounds", []), bbox):
            return False
        return True
