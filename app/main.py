from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from app.auth import create_token, verify_token, optional_token
from app.services import ApplicationServices, create_services
from app.settings import load_settings
from app.training_data.schema import feedback_from_event
from app.ui.gradio_app import create_gradio_app, get_ui_options

logger = logging.getLogger(__name__)


class TextSearchRequest(BaseModel):
    query: str
    limit: int = Field(default=20, ge=1, le=100)
    sensor: str | None = None
    date_start: str | None = None
    date_end: str | None = None
    bbox: list[float] | None = None


class ImageSearchRequest(BaseModel):
    image_path: str
    limit: int = Field(default=20, ge=1, le=100)
    sensor: str | None = None
    date_start: str | None = None
    date_end: str | None = None
    bbox: list[float] | None = None


class SimilarSceneRequest(BaseModel):
    scene_id: str
    limit: int = Field(default=20, ge=1, le=100)


class ChangeRequest(BaseModel):
    start_date: str
    end_date: str
    bbox: list[float] | None = None


class IngestRequest(BaseModel):
    source_path: str
    source_id: str | None = None


class DecisionRequest(BaseModel):
    analyst_label: str
    analyst_id: str | None = None
    note: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str



def _filters(payload: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "sensor": payload.sensor,
            "date_start": payload.date_start,
            "date_end": payload.date_end,
            "bbox": payload.bbox,
        }.items()
        if value is not None
    }


settings = load_settings()

if settings.offline_mode:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

services: ApplicationServices = create_services(settings)

from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="GeoSearch AI",
    description="AI-powered satellite imagery semantic search and change detection platform.",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Mount local data directory for direct preview serving
if settings.data_dir.exists():
    app.mount("/data", StaticFiles(directory=str(settings.data_dir)), name="data")


def _to_web_path(local_path: str | None) -> str | None:
    if not local_path:
        return None
    # Normalize Windows backslashes
    norm = str(local_path).replace("\\", "/")
    # If path contains /data/, return from /data/
    if "/data/" in norm:
        return "/data/" + norm.split("/data/", 1)[1]
    return local_path



@app.post("/api/auth/register")
def auth_register(request: RegisterRequest) -> dict[str, Any]:
    try:
        account = services.metadata.create_user(request.username, request.password)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    token = create_token(account["id"], account["username"])
    return {"token": token, "username": account["username"], "user_id": account["id"]}


@app.post("/api/auth/login")
def auth_login(request: LoginRequest) -> dict[str, Any]:
    account = services.metadata.authenticate_user(request.username, request.password)
    if not account:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_token(account["id"], account["username"])
    return {"token": token, "username": account["username"], "user_id": account["id"]}



@app.get("/api/health")
def health() -> dict[str, Any]:
    model_status = _model_status()
    return {
        "status": "ok",
        "application": "GeoSearch AI",
        "offline_mode": settings.offline_mode,
        "metadata": services.metadata.statistics(),
        "vectors": services.vectors.stats(),
        "models": model_status,
    }


def _model_status() -> dict[str, Any]:
    names = ["clip", "blip", "qwen3", "remoteclip", "prithvi"]
    result = {}
    for name in names:
        path = settings.model_path(name)
        staged = path.exists() and path.is_dir() and any(path.iterdir())
        result[name] = {"staged": staged, "path": str(path)}
    return result



@app.post("/api/search/text")
def search_text(request: TextSearchRequest) -> list[dict[str, Any]]:
    f = _filters(request)
    res = services.search.search_text(request.query, request.limit, f)
    for r in res:
        if r.get("preview_path"):
            r["preview_path"] = _to_web_path(r["preview_path"])
        if r.get("tile_preview_path"):
            r["tile_preview_path"] = _to_web_path(r["tile_preview_path"])
    return res


@app.post("/api/search/text/detailed")
def search_text_detailed(request: TextSearchRequest) -> dict[str, Any]:
    f = _filters(request)
    outcome = services.search.search_text_with_status(request.query, request.limit, f)
    for r in outcome.get("results", []):
        if r.get("preview_path"):
            r["preview_path"] = _to_web_path(r["preview_path"])
        if r.get("tile_preview_path"):
            r["tile_preview_path"] = _to_web_path(r["tile_preview_path"])
    return outcome


@app.post("/api/search/image")
def search_image(request: ImageSearchRequest) -> list[dict[str, Any]]:
    try:
        return services.search.search_image(request.image_path, request.limit, _filters(request))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/search/similar")
def search_similar(request: SimilarSceneRequest) -> list[dict[str, Any]]:
    return services.search.find_similar_to_scene(request.scene_id, request.limit)



@app.get("/api/library")
def library(limit: int = 500) -> list[dict[str, Any]]:
    """Return scene-level metadata and thumbnails."""
    scenes = services.metadata.list_scenes()[:max(1, min(limit, 5000))]
    result = []
    for scene in scenes:
        tiles = services.metadata.list_tiles(scene["id"])
        preview = scene["metadata"].get("preview_path") or (tiles[0]["preview_path"] if tiles else None)
        result.append({
            "scene_id": scene["id"],
            "source_id": scene["source_id"],
            "sensor": scene["sensor"],
            "acquisition_date": scene["acquisition_date"],
            "crs": scene["crs"],
            "bounds": scene["bounds"],
            "preview_path": _to_web_path(preview),
            "tile_count": len(tiles),
            "description": scene["metadata"].get("description", ""),
            "caption": scene["metadata"].get("caption", ""),
            "cloud_cover": scene["metadata"].get("cloud_cover", None),
            "resolution": scene["metadata"].get("resolution", None),
        })
    return result


@app.get("/api/scene/{scene_id}")
def get_scene(scene_id: str) -> dict[str, Any]:
    scene = services.metadata.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    tiles = services.metadata.list_tiles(scene_id)
    return {**scene, "tile_count": len(tiles), "tiles": tiles[:10]}



@app.post("/api/change/analyze")
def analyze_change(request: ChangeRequest) -> list[dict[str, Any]]:
    return services.change.analyze(request.start_date, request.end_date, request.bbox)


@app.get("/api/review-queue")
def review_queue() -> list[dict[str, Any]]:
    queue = services.change.review_queue()
    for item in queue:
        if item.get("before_preview"):
            item["before_preview"] = _to_web_path(item["before_preview"])
        if item.get("after_preview"):
            item["after_preview"] = _to_web_path(item["after_preview"])
        if item.get("mask_preview"):
            item["mask_preview"] = _to_web_path(item["mask_preview"])
    return queue


@app.post("/api/review/{event_id}/decision")
def review_decision(
    event_id: str,
    request: DecisionRequest,
    _user: dict[str, Any] | None = Depends(optional_token),
) -> dict[str, Any]:
    event = services.metadata.get_change_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Change event not found")
    try:
        decision = services.metadata.record_decision(
            event_id, request.analyst_label, request.analyst_id, request.note
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    after_tile = services.metadata.get_tile(event["after_tile_id"])
    services.feedback.append(
        feedback_from_event(
            {**event, "location": after_tile["bounds"] if after_tile else []},
            request.analyst_label,
            request.analyst_id,
        )
    )
    return decision


@app.get("/api/audit/{entity_id}")
def audit_log(entity_id: str) -> list[dict[str, Any]]:
    return services.metadata.audit_log(entity_id)


@app.get("/api/audit")
def audit_log_all(limit: int = 100) -> list[dict[str, Any]]:
    return services.metadata.audit_log()[-limit:]



@app.post("/ingest/scene")
@app.post("/api/ingest/scene")
def ingest_scene(request: IngestRequest) -> dict[str, Any]:
    try:
        return services.ingest.ingest_path(request.source_path, request.source_id)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        logger.exception("Ingestion error for %s", request.source_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {error}") from error



@app.get("/api/admin/stats")
def admin_stats() -> dict[str, Any]:
    return {
        **services.metadata.statistics(),
        **services.vectors.stats(),
        "models": _model_status(),
    }


@app.get("/api/vectors/{collection}")
def vector_records(collection: str, limit: int = 500) -> list[dict[str, Any]]:
    if collection not in {settings.semantic_collection, settings.visual_collection}:
        raise HTTPException(status_code=404, detail="Unknown vector collection")
    return services.vectors.all_points(collection)[: max(1, min(limit, 5000))]


@app.post("/api/export/change-events")
def export_change_events() -> dict[str, str]:
    return services.export.export_change_events()



import gradio as gr

ui_theme, ui_css, ui_js = get_ui_options()
demo = create_gradio_app(services, theme=ui_theme, css=ui_css, js=ui_js)
app = gr.mount_gradio_app(app, demo, path="/gradio")

