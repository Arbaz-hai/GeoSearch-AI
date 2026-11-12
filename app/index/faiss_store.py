from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

# Assuming logger is defined elsewhere in your file
logger = logging.getLogger(__name__)

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
                logger.info("Created new FAISS IndexFlatIP for '%s'", collection)
                
        except Exception as err:
            logger.warning("FAISS native unavailable or failed to load (%s); using fallback for %s", err, collection)
            self._indexes[collection] = None

    def upsert(
        self, 
        collection: str, 
        point_id: str, 
        vector: np.ndarray, 
        payload: dict[str, Any] | None = None,
    ) -> bool:
        if collection not in self.COLLECTIONS:
            logger.warning("Attempted to upsert into unknown collection: %s", collection)
            return False

        # Ensure vector is a 2D float32 array
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)

        # STRICT DIMENSION CHECK
        # Note: Interpolating or padding embedding vectors destroys their semantic meaning. 
        # If dimensions mismatch, the upstream embedding pipeline needs adjustment.
        if vec.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch in collection '{collection}'. "
                f"Expected {self.dimension}, got {vec.shape[1]}."
            )

        # L2 Normalize for Cosine Similarity (since we use IndexFlatIP)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        else:
            logger.warning("Zero vector provided for upsert on point_id: %s", point_id)

        # Update index and metadata
        if self._indexes.get(collection) is not None:
            self._indexes[collection].add(vec)
            
            # Update ID mapping
            if point_id not in self._id_maps[collection]:
                self._id_maps[collection].append(point_id)
                
            # Update payloads
            self._payloads[collection][point_id] = payload or {}
            
            # Persist changes to disk
            self._persist(collection)
            return True
            
        return False
