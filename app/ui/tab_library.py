from __future__ import annotations

from typing import Any
from gradio import SelectData


def build_library_tab(gr: Any, services: Any) -> None:
    gr.Markdown(
        """
        ### Satellite Imagery Archive & Semantic Clustering
        Explore the complete local scene archive. The system automatically groups imagery into visual clusters using unsupervised clustering (K-Means, HDBSCAN, DBSCAN) in the CLIP embedding space - no manual labels required.
        """
    )

    with gr.Row():
        cluster_method = gr.Radio(
            label="Unsupervised Clustering Algorithm",
            choices=["K-Means", "HDBSCAN", "DBSCAN"],
            value="K-Means",
            scale=2,
        )
        refresh_button = gr.Button("Recluster & Refresh Archive", variant="primary", scale=1)

    status_markdown = gr.Markdown("Click 'Recluster & Refresh Archive' to load and cluster local imagery scenes.")

    gallery = gr.Gallery(
        label="Indexed Satellite Scenes (Complete Scene Assets)",
        columns=5,
        object_fit="contain",
        height="auto",
    )
    library_state = gr.State([])

    gr.Markdown("---")
    gr.Markdown("#### Inspected Scene Telemetry & Cluster Neighbors")
    with gr.Row():
        with gr.Column(scale=1):
            selected_image = gr.Image(label="Scene Preview", interactive=False, height=360)
            similar_btn = gr.Button("Find Nearest Neighbors in Archive", variant="secondary")
            current_scene_id = gr.State(None)

        with gr.Column(scale=2):
            selected_details = gr.Markdown(
                """
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1e293b; border-radius: 12px; padding: 20px;">
                  <p style="color: #94a3b8; margin: 0;">Click on any scene above to view its validated telemetry, cluster assignment, and nearest visual neighbors.</p>
                </div>
                """
            )

    def load_archive(method_name: str):
        method_key = method_name.lower().replace("-", "")
        overview = services.cluster.cluster_overview(max_clusters=8, method=method_key)
        
        scene_clusters: dict[str, int] = {}
        for pt in overview.get("points", []):
            sc_id = pt.get("scene_id")
            if sc_id and sc_id not in scene_clusters:
                scene_clusters[sc_id] = pt.get("cluster", 0)

        scenes = services.metadata.list_scenes()
        cards = []
        records = []

        for sc in scenes:
            sc_id = sc["id"]
            tiles = services.metadata.list_tiles(sc_id)
            preview = sc.get("metadata", {}).get("preview_path") or (tiles[0]["preview_path"] if tiles else None)
            cluster_num = scene_clusters.get(sc_id, scene_clusters.get(sc["source_id"], 0))

            title = sc.get("metadata", {}).get("title") or sc["source_id"]
            label = f"{title} · Cluster {cluster_num}"
            cards.append((preview, label))

            records.append({
                "preview_path": preview,
                "title": title,
                "scene": sc,
                "tile_count": len(tiles),
                "cluster": cluster_num,
                "bounds": sc.get("bounds", []),
                "crs": sc.get("crs", "EPSG:4326"),
                "source_path": sc.get("source_path", ""),
                "description": sc.get("metadata", {}).get("description") or "Satellite scene asset",
            })

        first_prev = records[0]["preview_path"] if records else None
        first_detail = _format_scene_detail(records[0]) if records else "Archive is currently empty."
        first_id = records[0]["scene"]["id"] if records else None

        summary = (
            f"Archived: **{len(records)}** complete scene(s) organized into **{overview['cluster_count']}** semantic clusters via **{method_name}**."
        )
        return cards, records, summary, first_prev, first_detail, first_id

    def show_scene_detail(records: list[dict[str, Any]], evt: SelectData):
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if not isinstance(idx, int) or idx < 0 or idx >= len(records):
            return None, "Select a scene.", None
        item = records[idx]
        return item["preview_path"], _format_scene_detail(item), item["scene"]["id"]

    def find_scene_neighbors(scene_id: str | None):
        if not scene_id:
            return "Select a scene first."
        neighbors = services.cluster.find_similar(scene_id, limit=6)
        if not neighbors:
            return f"No similar neighbors found for scene `{scene_id}`."

        items = []
        for nb in neighbors:
            items.append(f"- **{nb['source_name']}** ({nb['similarity']:.1%} match) — {nb.get('description', '')[:80]}...")
        return "#### Nearest Semantic Neighbors\n" + "\n".join(items)

    refresh_button.click(
        load_archive,
        inputs=cluster_method,
        outputs=[gallery, library_state, status_markdown, selected_image, selected_details, current_scene_id],
    )
    gallery.select(
        show_scene_detail,
        inputs=library_state,
        outputs=[selected_image, selected_details, current_scene_id],
    )
    similar_btn.click(
        find_scene_neighbors,
        inputs=current_scene_id,
        outputs=selected_details,
    )


def _format_scene_detail(item: dict[str, Any]) -> str:
    sc = item.get("scene") or {}
    meta = sc.get("metadata") or {}

    return f"""
    <div style="background: #0d1527; border: 1px solid #1e3a5f; border-radius: 12px; padding: 18px; font-family: monospace, sans-serif;">
      <h3 style="color: #38bdf8; margin-top: 0;">{item.get('title')}</h3>
      <p style="color: #cbd5e1;">{item.get('description')}</p>
      
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 14px;">
        <div><strong style="color: #94a3b8;">Acquisition Date:</strong> {sc.get('acquisition_date', 'Unknown')}</div>
        <div><strong style="color: #94a3b8;">Sensor Platform:</strong> {sc.get('sensor', 'Unknown')}</div>
        <div><strong style="color: #94a3b8;">Semantic Cluster:</strong> <span style="color: #34d399; font-weight: 700;">Cluster {item.get('cluster', 0)}</span></div>
        <div><strong style="color: #94a3b8;">Spatial Tiles:</strong> {item.get('tile_count', 1)} indexed</div>
        <div><strong style="color: #94a3b8;">CRS:</strong> {item.get('crs', 'EPSG:4326')}</div>
        <div><strong style="color: #94a3b8;">Resolution:</strong> {meta.get('resolution', meta.get('GSD', '10 m'))}</div>
      </div>

      <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #1e293b;">
        <strong style="color: #94a3b8;">Bounds (WGS84):</strong><br>
        <code style="color: #a5f3fc; font-size: 11px;">{item.get('bounds')}</code>
      </div>
      <div style="margin-top: 8px;">
        <strong style="color: #94a3b8;">Local File Source:</strong><br>
        <code style="color: #94a3b8; font-size: 11px;">{item.get('source_path')}</code>
      </div>
    </div>
    """
