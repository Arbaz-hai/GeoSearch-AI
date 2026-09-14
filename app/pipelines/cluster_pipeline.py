from __future__ import annotations

from typing import Any
import numpy as np

from app.index.metadata_store import MetadataStore
from app.index.qdrant_store import QdrantStore
from app.pipelines.search_pipeline import SearchPipeline
from app.settings import Settings


class ClusterPipeline:
    """
    Unsupervised clustering over the CLIP embedding space using HDBSCAN, DBSCAN, or KMeans.
    Enables archive-wide semantic exploration and "Find me more like this".
    """

    def __init__(
        self,
        settings: Settings,
        metadata: MetadataStore,
        vectors: QdrantStore,
        search: SearchPipeline,
    ):
        self.settings = settings
        self.metadata = metadata
        self.vectors = vectors
        self.search = search

    def cluster_overview(self, max_clusters: int = 8, method: str = "kmeans") -> dict[str, Any]:
        points = self.vectors.all_points(self.settings.semantic_collection)
        if not points:
            points = self.vectors.all_points(self.settings.visual_collection)
        if not points:
            return {"points": [], "cluster_count": 0, "clusters": {}}

        vectors = np.asarray([p["vector"] for p in points], dtype=np.float32)
        labels = self._cluster_vectors(vectors, max_clusters, method=method)

        centered = vectors - vectors.mean(axis=0, keepdims=True)
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ basis[:2].T if basis.shape[0] >= 2 else np.pad(centered[:, :1], ((0, 0), (0, 1)))

        result = []
        cluster_map: dict[int, list[str]] = {}
        for point, coord, label in zip(points, coords, labels):
            lbl = int(label)
            payload = point["payload"]
            tile_id = payload.get("tile_id")
            scene_id = payload.get("scene_id") or payload.get("source_id")
            result.append({
                "tile_id": tile_id,
                "scene_id": scene_id,
                "x": float(coord[0]),
                "y": float(coord[1]),
                "cluster": lbl,
            })
            cluster_map.setdefault(lbl, []).append(scene_id)

        distinct_clusters = sorted(list(set(labels.tolist())))
        return {
            "points": result,
            "cluster_count": len(distinct_clusters),
            "clusters": cluster_map,
        }

    def find_similar(self, scene_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return self.search.find_similar_to_scene(scene_id, limit)

    @staticmethod
    def _cluster_vectors(vectors: np.ndarray, max_clusters: int, method: str = "kmeans") -> np.ndarray:
        n = len(vectors)
        if n <= 1:
            return np.zeros(n, dtype=int)

        k = min(max_clusters, max(2, int(np.sqrt(n))))

        if method == "hdbscan":
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, n // 10), metric="euclidean")
                labels = clusterer.fit_predict(vectors)
                if len(set(labels)) > 1:
                    return labels
            except Exception:
                pass

        if method == "dbscan":
            try:
                from sklearn.cluster import DBSCAN
                clusterer = DBSCAN(eps=0.45, min_samples=2, metric="cosine")
                labels = clusterer.fit_predict(vectors)
                if len(set(labels)) > 1:
                    return labels
            except Exception:
                pass

        try:
            from sklearn.cluster import KMeans
            return KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(vectors)
        except Exception:
            order = np.argsort(vectors[:, 0])
            labels = np.zeros(n, dtype=int)
            for rank, idx in enumerate(order):
                labels[idx] = min(k - 1, rank * k // n)
            return labels
