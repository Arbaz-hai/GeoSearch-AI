from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


_LABEL_ALIASES = {
    "AnnualCrop": {"annualcrop", "annual crop", "crop", "crops", "farmland", "agriculture", "agricultural"},
    "Forest": {"forest", "trees", "tree cover", "woodland"},
    "HerbaceousVegetation": {"herbaceousvegetation", "herbaceous vegetation", "grass", "grassland", "vegetation"},
    "Highway": {"highway", "road", "roads", "motorway"},
    "Industrial": {"industrial", "industry", "factory", "factories", "warehouse"},
    "Pasture": {"pasture", "grazing"},
    "PermanentCrop": {"permanentcrop", "permanent crop", "orchard", "vineyard"},
    "Residential": {"residential", "housing", "homes", "neighborhood", "buildings", "building"},
    "River": {"river", "rivers", "stream", "waterway"},
    "SeaLake": {"sealake", "sea lake", "sea", "lake", "water", "reservoir", "coast"},
}

_FEATURE_TITLES = [
    ({"river", "stream", "waterway"}, "River / Waterway — Satellite Scene"),
    ({"lake", "sealake", "sea", "reservoir", "ocean"}, "Water Body / Basin — Satellite Scene"),
    ({"forest", "woodland", "trees"}, "Forest & Dense Vegetation Canopy"),
    ({"urban", "residential", "housing", "building", "city"}, "Urban & Built-Up Infrastructure"),
    ({"industrial", "factory", "warehouse"}, "Industrial & Commercial Facility"),
    ({"highway", "road", "motorway"}, "Transportation & Road Corridor"),
    ({"crop", "annualcrop", "farmland", "agriculture"}, "Agricultural & Cropland Terrain"),
]


def _clean_name(path: Path) -> str:
    value = re.sub(r"[_-]+", " ", path.stem)
    value = re.sub(r"\b20\d{2}\b|\b[01]\d\b|\b[0-3]\d\b", "", value)
    return re.sub(r"\s+", " ", value).strip() or "Earth Observation Scene"


def _as_text(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value)
    return str(value or "")


def infer_semantic_tags(source: Path, metadata: dict[str, Any]) -> list[str]:
    supplied = metadata.get("semantic_tags", metadata.get("tags", []))
    if isinstance(supplied, str):
        supplied = re.split(r"[,;]", supplied)
    tags = [str(tag).strip() for tag in supplied if str(tag).strip()] if isinstance(supplied, (list, tuple, set)) else []
    searchable = " ".join(
        [
            _clean_name(source),
            _as_text(metadata.get("description")),
            _as_text(metadata.get("title")),
            " ".join(tags),
        ]
    ).lower()
    normalized = {tag.lower() for tag in tags}
    for label, aliases in _LABEL_ALIASES.items():
        if label.lower() in normalized or any(alias in searchable for alias in aliases):
            tags.append(label)
            normalized.add(label.lower())
    return list(dict.fromkeys(tags))


def generate_grounded_title(source: Path, tags: list[str], metadata: dict[str, Any]) -> str:
    """
    Generate an objective, strictly grounded title for the scene.
    
    IMPORTANT: Never hallucinate specific geographic names (e.g. 'Ganga River',
    'Main Road') unless authenticated by verified geospatial metadata.
    Visual classification is distinct from geographic identification.
    """
    verified_location = metadata.get("location_name") or metadata.get("geographic_name")
    
    explicit_title = metadata.get("title")
    if explicit_title and isinstance(explicit_title, str) and explicit_title.strip():
        clean_title = explicit_title.strip()
        if not re.search(r"sentinel|landsat|highlight|natural_color|scene|\.tif", clean_title, re.IGNORECASE):
            return clean_title

    tags_lower = {t.lower() for t in tags}
    for keywords, candidate_title in _FEATURE_TITLES:
        if any(k in tags_lower for k in keywords):
            if verified_location:
                return f"{candidate_title} ({verified_location})"
            return candidate_title

    clean = _clean_name(source)
    if "sentinel" in clean.lower() or "landsat" in clean.lower() or len(clean) > 35:
        return "Earth Observation Satellite Scene"
    return f"{clean.title()} — Satellite Scene"


def describe_scene(source: Path, acquisition_date: str, sensor: str, metadata: dict[str, Any]) -> tuple[str, list[str]]:
    """
    Return a concise, grounded description and searchable tags for an ingested scene.
    Separates visual classification from geographic identification.
    """
    description = str(metadata.get("description") or metadata.get("caption") or "").strip()
    tags = infer_semantic_tags(source, metadata)
    title = generate_grounded_title(source, tags, metadata)
    metadata["title"] = title

    label = _clean_name(source)
    if not description:
        description = f"{label.capitalize()} imagery"
        if acquisition_date:
            description += f" from {acquisition_date}"
        if sensor and sensor != "Unknown":
            description += f" ({sensor})"
        description += "."
    if tags:
        description = f"{description.rstrip('.')} Showing: {', '.join(tags)}."

    return description, tags


def describe_tile(scene_description: str, image: np.ndarray, grid_row: int, grid_col: int) -> str:
    """Create a grounded description for the image."""
    return scene_description
