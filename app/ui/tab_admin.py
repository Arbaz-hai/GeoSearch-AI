from __future__ import annotations

from pathlib import Path
from typing import Any


def build_admin_tab(gr: Any, services: Any) -> None:
    gr.Markdown(
        """
        ### Satellite Imagery Ingestion — GeoTIFF, COG, ZIP, Folder
        Add satellite imagery to the local offline archive. Files are indexed with CLIP semantic embeddings, BLIP captions, and spatial tiles.
        Upload files directly, or paste a local file/folder/ZIP path. **All imagery stays on this machine.**
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            uploads = gr.File(
                label="Upload Imagery (GeoTIFF, COG, PNG, JPG, JP2, WEBP, ZIP)",
                file_count="multiple",
                type="filepath",
            )
            source_path = gr.Textbox(
                label="Or paste a local file, folder, or ZIP path",
                placeholder=r"E:\Imagery\sentinel2\observations.zip",
            )
            with gr.Row():
                ingest_button = gr.Button("Ingest & Index", variant="primary", scale=2)
                clear_button = gr.Button("Clear", variant="secondary", scale=1)
        
        with gr.Column(scale=2):
            gr.Markdown("#### Offline Model Staging Status")
            model_status_display = gr.Markdown(_generate_model_status(services))
            gr.Markdown("---")
            gr.Markdown("#### Archive Statistics")
            archive_stats_display = gr.Markdown(_generate_archive_stats(services))
            refresh_stats_btn = gr.Button("Refresh Stats", variant="secondary")

    ingestion_status = gr.Markdown("Ready. No imagery has been ingested yet in this session.")

    gr.Markdown("---")
    gr.Markdown(
        """
        #### CLI Ingestion (Recommended for Large Archives)
        For ingesting hundreds of scenes, use the offline CLI with resume support:
        ```powershell
        # Ingest a folder
        python -m app.ingestion.ingest --input ./data/imagery/sentinel2

        # Ingest a ZIP archive
        python -m app.ingestion.ingest --input ./data/downloads/observations.zip

        # Force re-index existing scenes
        python -m app.ingestion.ingest --input ./data/imagery --force

        # Stage ML model weights first (requires internet)
        python scripts/stage_models.py
        ```
        """
    )

    def do_ingest(path: str | None, uploaded: list[str] | None, progress=gr.Progress()):
        values = [str(item) for item in (uploaded or []) if item]
        if not values:
            v = (path or "").strip().strip('"')
            if v:
                values = [v]
        if not values:
            return "Upload imagery files or enter a local path to get started."

        messages: list[str] = []
        progress(0.05, desc="Starting GeoSearch AI ingestion pipeline...")

        def update(msg: str) -> None:
            messages.append(msg)
            progress(None, desc=msg)

        try:
            results = [services.ingest.ingest_path(Path(v), progress=update) for v in values]
        except (FileNotFoundError, ValueError) as err:
            return f"Error: Could not access path — {err}"
        except Exception as err:
            return f"Ingestion failed: {err}"

        progress(1.0, desc="Ingestion complete")

        if all(r.get("status") == "empty" for r in results):
            return f"No supported imagery was found at the specified path.\n{results[0].get('errors', [''])[0]}"

        files_indexed = sum(r.get("files_indexed", 0) for r in results)
        tiles_indexed = sum(r.get("tiles_indexed", 0) for r in results)
        duplicates = sum(r.get("duplicates", 0) for r in results)
        changes = sum(len(r.get("change_events", [])) for r in results)
        errors = [e for r in results for e in r.get("errors", [])]

        summary = f"""
#### Ingestion Complete

| Metric | Count |
|--------|-------|
| Images Ingested | **{files_indexed}** |
| Spatial Tiles Indexed | **{tiles_indexed}** |
| Duplicate Scenes Skipped | {duplicates} |
| Change Candidates Detected | {changes} |
| Errors | {len(errors)} |
"""
        if changes:
            summary += f"\n> **{changes}** new multi-temporal change candidates flagged for analyst review in the Compare tab."
        if errors:
            summary += "\n\n**Errors encountered:**\n" + "\n".join(f"- {e}" for e in errors[:5])
        if messages:
            summary += "\n\n**Processing Log:**\n" + "\n".join(f"- {m}" for m in messages[-6:])

        return summary

    def refresh_system_stats():
        return _generate_archive_stats(services)

    ingest_button.click(do_ingest, [source_path, uploads], ingestion_status)
    clear_button.click(lambda: (None, "", "Ready."), outputs=[uploads, source_path, ingestion_status])
    refresh_stats_btn.click(refresh_system_stats, outputs=archive_stats_display)


def _generate_model_status(services: Any) -> str:
    models = [
        ("CLIP Embedder", "clip", "openai/clip-vit-base-patch32"),
        ("BLIP Captioner", "blip", "Salesforce/blip-image-captioning-base"),
        ("Qwen3 NLP", "qwen3", "Qwen/Qwen3-0.6B"),
        ("RemoteCLIP", "remoteclip", "chendelong/RemoteCLIP"),
        ("Prithvi-100M", "prithvi", "ibm-nasa-geospatial/Prithvi-100M"),
    ]
    rows = []
    for label, key, repo in models:
        p = services.settings.model_path(key)
        staged = p.exists() and p.is_dir() and any(p.iterdir())
        status = "[Staged]" if staged else "[Fallback]"
        rows.append(f"| {label} | {status} |")

    header = "| Model | Status |\n|-------|--------|\n"
    return header + "\n".join(rows) + "\n\n*Run `python scripts/stage_models.py` to download missing models.*"


def _generate_archive_stats(services: Any) -> str:
    try:
        db = services.metadata.statistics()
        vs = services.vectors.stats()
        return f"""
| Metric | Value |
|--------|-------|
| Indexed Scenes | **{db.get('scene_count', 0)}** |
| Spatial Tiles | **{db.get('tile_count', 0)}** |
| Semantic Vectors | **{vs.get('semantic_vectors', 0)}** |
| Visual Vectors | **{vs.get('visual_vectors', 0)}** |
| Pending Reviews | **{db.get('pending_change_count', 0)}** |
| Vector Backend | {vs.get('backend', 'Qdrant Local')} |
"""
    except Exception:
        return "*Stats unavailable — click Refresh Stats.*"
