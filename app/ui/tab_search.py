from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gradio import SelectData


def build_search_tab(gr: Any, services: Any) -> None:
    gr.Markdown(
        """
        ### Earth Observation Semantic & Visual Search
        Search the local imagery archive using natural language queries (interpreted by Qwen3 + CLIP embeddings) or upload a reference satellite image.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_query = gr.Textbox(
                label="Natural Language Query (Qwen3 NLP & Semantic Matching)",
                placeholder="e.g. 'Find images containing rivers', 'Show roads around urban areas after 2023', 'Find construction activity'",
                lines=2,
            )
            with gr.Row():
                example_1 = gr.Button("Rivers & Waterways", size="sm", variant="secondary")
                example_2 = gr.Button("Urban & Roads", size="sm", variant="secondary")
                example_3 = gr.Button("Farmland & Agriculture", size="sm", variant="secondary")
                example_4 = gr.Button("Construction Activity", size="sm", variant="secondary")
            
            image_query = gr.Image(
                label="Reference Satellite Image (Visual Similarity Search via CLIP)",
                type="pil",
                height=180,
            )

        with gr.Column(scale=2):
            sensor = gr.Dropdown(
                label="Sensor / Platform",
                choices=["All Sensors", "Sentinel-2 L2A", "Sentinel-2 RGB", "Sentinel-1", "Landsat Collection 2", "Bhuvan"],
                value="All Sensors",
            )
            with gr.Row():
                date_start = gr.Textbox(label="Acquisition Start", placeholder="YYYY-MM-DD", value="")
                date_end = gr.Textbox(label="Acquisition End", placeholder="YYYY-MM-DD", value="")

            with gr.Accordion("Geographic AOI Bounding Box (WGS84)", open=False):
                with gr.Row():
                    min_lon = gr.Number(label="West Lon", value=None)
                    min_lat = gr.Number(label="South Lat", value=None)
                    max_lon = gr.Number(label="East Lon", value=None)
                    max_lat = gr.Number(label="North Lat", value=None)
                use_aoi = gr.Checkbox(label="Filter by Bounding Box", value=False)

            with gr.Row():
                search_button = gr.Button("Search Archive", variant="primary", scale=2)
                clear_button = gr.Button("Clear", variant="secondary", scale=1)

    match_status = gr.Markdown("Ready. Enter a natural language query or upload a reference image to begin.")
    
    gr.Markdown("#### Ranked Imagery Results (Scene Assets)")
    gallery = gr.Gallery(
        label="Matching Imagery Scenes (Select an image to inspect full telemetry & metadata)",
        columns=4,
        rows=2,
        object_fit="contain",
        height="auto",
    )
    result_state = gr.State([])

    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=1):
            selected_image = gr.Image(label="Inspected Scene Preview", interactive=False, height=360)
            find_more_button = gr.Button("Find More Like This", variant="primary")
            selected_scene_id_state = gr.State(None)

        with gr.Column(scale=2):
            selected_card = gr.Markdown(
                """
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1e293b; border-radius: 12px; padding: 20px;">
                  <h4 style="color: #38bdf8; margin-top: 0;">GeoSearch AI Result Card</h4>
                  <p style="color: #94a3b8;">Select any result above to view its validated telemetry, sensor resolution, bounding box coordinates, and model provenance.</p>
                </div>
                """
            )

    def execute_search(
        query: str,
        image: Any,
        sensor_val: str,
        start: str,
        end: str,
        west: float | None,
        south: float | None,
        east: float | None,
        north: float | None,
        apply_aoi: bool,
    ):
        filters: dict[str, Any] = {}
        if sensor_val and sensor_val != "All Sensors":
            filters["sensor"] = sensor_val
        if start and start.strip():
            filters["date_start"] = start.strip()
        if end and end.strip():
            filters["date_end"] = end.strip()
        if apply_aoi and None not in {west, south, east, north}:
            filters["bbox"] = [west, south, east, north]

        if image is not None:
            outcome = services.search.search_image_with_status(image, filters=filters)
        elif query and query.strip():
            outcome = services.search.search_text_with_status(query.strip(), filters=filters)
        else:
            return [], "Please enter a descriptive search query or upload a reference image.", [], None, _empty_card()

        found = outcome["results"]
        gallery_items = []
        for item in found:
            date_str = f" · {item['acquisition_date']}" if item.get('acquisition_date') else ""
            sim_str = f" · {item['similarity']:.0%} match" if item.get('similarity') else ""
            label = f"#{item['rank']} {item['source_name']}{date_str}{sim_str}"
            gallery_items.append((item["preview_path"], label))

        first_img = found[0]["preview_path"] if found else None
        first_card = _format_card(found[0]) if found else _empty_card()
        first_scene_id = found[0]["scene_id"] if found else None

        return gallery_items, outcome["status"], found, first_img, first_card, first_scene_id

    def show_selected(found: list[dict[str, Any]], evt: SelectData):
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if not isinstance(idx, int) or idx < 0 or idx >= len(found):
            return None, _empty_card(), None
        item = found[idx]
        return item["preview_path"], _format_card(item), item["scene_id"]

    def find_more_like_this(scene_id: str | None, current_results: list[dict[str, Any]]):
        if not scene_id:
            return [], "Select a result first before clicking 'Find More Like This'.", current_results, None, _empty_card(), None
        similar = services.search.find_similar_to_scene(scene_id)
        if not similar:
            return [], f"No additional similar scenes found for scene `{scene_id}`.", current_results, None, _empty_card(), None

        gallery_items = []
        for item in similar:
            date_str = f" · {item['acquisition_date']}" if item.get('acquisition_date') else ""
            label = f"#{item['rank']} {item['source_name']}{date_str} · {item['similarity']:.0%} similar"
            gallery_items.append((item["preview_path"], label))

        first_img = similar[0]["preview_path"] if similar else None
        first_card = _format_card(similar[0]) if similar else _empty_card()
        first_scene_id = similar[0]["scene_id"] if similar else None

        status = f"Displaying **{len(similar)}** scenes most similar to `{scene_id}` (evaluated via CLIP embeddings)."
        return gallery_items, status, similar, first_img, first_card, first_scene_id

    search_button.click(
        execute_search,
        [text_query, image_query, sensor, date_start, date_end, min_lon, min_lat, max_lon, max_lat, use_aoi],
        [gallery, match_status, result_state, selected_image, selected_card, selected_scene_id_state],
    )
    gallery.select(
        show_selected,
        result_state,
        [selected_image, selected_card, selected_scene_id_state],
    )
    find_more_button.click(
        find_more_like_this,
        [selected_scene_id_state, result_state],
        [gallery, match_status, result_state, selected_image, selected_card, selected_scene_id_state],
    )

    example_1.click(lambda: "Find imagery containing rivers and natural waterways", outputs=text_query)
    example_2.click(lambda: "Show roads around urban infrastructure and residential areas", outputs=text_query)
    example_3.click(lambda: "Find agricultural cropland and vegetation terrain", outputs=text_query)
    example_4.click(lambda: "Find active construction and ground clearance", outputs=text_query)
    clear_button.click(
        lambda: ("", None, "All Sensors", "", "", None, None, None, None, False, [], "Cleared.", [], None, _empty_card(), None),
        outputs=[text_query, image_query, sensor, date_start, date_end, min_lon, min_lat, max_lon, max_lat, use_aoi, gallery, match_status, result_state, selected_image, selected_card, selected_scene_id_state],
    )


def _empty_card() -> str:
    return """
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1e293b; border-radius: 12px; padding: 20px;">
      <h4 style="color: #38bdf8; margin-top: 0;">GeoSearch AI Result Card</h4>
      <p style="color: #94a3b8;">Select any result above to view its validated telemetry, sensor resolution, bounding box coordinates, and model provenance.</p>
    </div>
    """


def _format_card(item: dict[str, Any]) -> str:
    """
    Format image result card strictly complying with Requirement 6:
    Only display fields that actually exist. Never fabricate metadata.
    """
    lines = [
        '<div style="background: #0d1527; border: 1px solid #1e3a5f; border-radius: 12px; padding: 18px; font-family: monospace, sans-serif;">',
        f'<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e3a5f; padding-bottom: 10px; margin-bottom: 14px;">',
        f'  <span style="color: #38bdf8; font-size: 16px; font-weight: 700;">GeoSearch AI — Image Card</span>',
        f'  <span style="background: #1e3a5f; color: #7dd3fc; padding: 3px 8px; border-radius: 6px; font-size: 12px;">Match: {item.get("similarity", 0):.1%}</span>',
        f'</div>',
    ]

    acq_date = item.get("acquisition_date", "Unknown Date")
    sensor_val = item.get("sensor") or "Satellite Sensor"
    title = f"{sensor_val} Imagery ({acq_date})"
    lines.append(f'<div style="margin-bottom: 8px;"><strong style="color: #94a3b8;">Title:</strong> <span style="color: #f8fafc; font-weight: 600;">{title}</span></div>')

    import re
    desc = item.get("description")
    if desc:
        desc = re.sub(r"\s*Section \(\d+, \d+\):.*", "", desc)
        lines.append(f'<div style="margin-bottom: 8px;"><strong style="color: #94a3b8;">Description:</strong> <span style="color: #cbd5e1;">{desc}</span></div>')

    acq_date = item.get("acquisition_date")
    if acq_date:
        lines.append(f'<div style="margin-bottom: 6px;"><strong style="color: #94a3b8;">Acquisition Date:</strong> <span style="color: #f8fafc;">{acq_date}</span></div>')

    acq_time = item.get("acquisition_time")
    if acq_time:
        lines.append(f'<div style="margin-bottom: 6px;"><strong style="color: #94a3b8;">Acquisition Time:</strong> <span style="color: #f8fafc;">{acq_time}</span></div>')

    dataset = item.get("dataset")
    if dataset:
        lines.append(f'<div style="margin-bottom: 6px;"><strong style="color: #94a3b8;">Dataset:</strong> <span style="color: #f8fafc;">{dataset}</span></div>')

    sensor = item.get("sensor")
    if sensor and sensor != "Unknown":
        lines.append(f'<div style="margin-bottom: 6px;"><strong style="color: #94a3b8;">Sensor:</strong> <span style="color: #f8fafc;">{sensor}</span></div>')

    res = item.get("resolution")
    if res:
        lines.append(f'<div style="margin-bottom: 6px;"><strong style="color: #94a3b8;">Resolution:</strong> <span style="color: #f8fafc;">{res}</span></div>')

    cloud = item.get("cloud_cover")
    if cloud is not None and cloud != "":
        lines.append(f'<div style="margin-bottom: 6px;"><strong style="color: #94a3b8;">Cloud Coverage:</strong> <span style="color: #f8fafc;">{cloud}%</span></div>')

    lat = item.get("center_lat")
    lon = item.get("center_lon")
    if lat and lon:
        lines.append(f'<div style="margin-bottom: 6px;"><strong style="color: #94a3b8;">Center Coordinates:</strong> <span style="color: #38bdf8;">{lat:.5f}° N, {lon:.5f}° E</span></div>')

    bbox = item.get("bounds")
    if bbox and len(bbox) == 4:
        lines.append(f'<div style="margin-bottom: 6px;"><strong style="color: #94a3b8;">Bounding Box:</strong> <code style="background: #111d35; color: #a5f3fc; padding: 2px 5px; border-radius: 4px;">[{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]</code></div>')

    crs = item.get("crs")
    if crs:
        lines.append(f'<div style="margin-bottom: 6px;"><strong style="color: #94a3b8;">CRS:</strong> <span style="color: #f8fafc;">{crs}</span></div>')

    source = item.get("source_path") or item.get("source_name")
    if source:
        lines.append(f'<div style="margin-bottom: 6px;"><strong style="color: #94a3b8;">Source:</strong> <span style="color: #94a3b8;">{source}</span></div>')

    ingest_time = item.get("ingestion_time")
    if ingest_time:
        lines.append(f'<div style="margin-bottom: 6px;"><strong style="color: #94a3b8;">Ingestion Time:</strong> <span style="color: #94a3b8;">{ingest_time}</span></div>')

    image_id = item.get("scene_id") or item.get("tile_id")
    if image_id:
        lines.append(f'<div style="margin-bottom: 6px;"><strong style="color: #94a3b8;">Scene ID:</strong> <code style="color: #fcd34d;">{image_id}</code></div>')

    lines.append('</div>')
    return "\n".join(lines)
