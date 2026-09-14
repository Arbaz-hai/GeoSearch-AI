from __future__ import annotations

from typing import Any

from app.ui.tab_account import build_account_tab
from app.ui.tab_admin import build_admin_tab
from app.ui.tab_change import build_change_tab
from app.ui.tab_library import build_library_tab
from app.ui.tab_search import build_search_tab


UI_CSS = """
    :root {
      --ink: #f1f5f9;
      --ink-soft: #cbd5e1;
      --muted: #64748b;
      --line: #1e3a5f;
      --panel: #0d1527;
      --canvas: #060a14;
      --accent: #00d2ff;
      --accent-dark: #0284c7;
      --accent-soft: #0c4a6e;
      --glow: rgba(0, 210, 255, 0.15);
    }
    
    /* Background: Space & Satellite Orbit Theme */
    body, .gradio-container {
      background-color: var(--canvas) !important;
      background-image: 
        radial-gradient(circle at 10% 15%, rgba(14, 165, 233, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(99, 102, 241, 0.06) 0%, transparent 45%),
        radial-gradient(ellipse at 50% 0%, rgba(30, 58, 95, 0.25) 0%, transparent 60%),
        linear-gradient(180deg, #070d1a 0%, #04070d 100%) !important;
      background-attachment: fixed !important;
      color: var(--ink) !important;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    .gradio-container {
      max-width: 1540px !important;
      padding: 24px 36px 48px !important;
    }
    
    /* Hero Header */
    .hero-panel {
      background: linear-gradient(135deg, rgba(13, 21, 39, 0.95) 0%, rgba(17, 34, 64, 0.90) 100%);
      border: 1px solid #1e3a5f;
      border-radius: 16px;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
      margin-bottom: 20px;
      padding: 22px 28px;
      position: relative;
      overflow: hidden;
    }
    
    .hero-panel::after {
      content: "";
      position: absolute;
      top: 0;
      right: 0;
      width: 450px;
      height: 100%;
      background: radial-gradient(circle at 80% 50%, rgba(0, 210, 255, 0.12) 0%, transparent 70%);
      pointer-events: none;
    }
    
    .hero-eyebrow {
      color: var(--accent) !important;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.15em;
      margin-bottom: 6px;
      text-transform: uppercase;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .pulse-dot {
      width: 8px;
      height: 8px;
      background-color: #22c55e;
      border-radius: 50%;
      box-shadow: 0 0 8px #22c55e;
      display: inline-block;
    }
    
    .hero-title {
      color: #ffffff !important;
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin: 0 0 6px 0;
    }
    
    .hero-subtitle {
      color: var(--ink-soft) !important;
      font-size: 14px;
      font-weight: 400;
      margin: 0;
      max-width: 860px;
      line-height: 1.5;
    }

    .telemetry-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid rgba(30, 58, 95, 0.6);
      font-size: 12px;
    }

    .telemetry-pill {
      background: rgba(11, 23, 44, 0.8);
      border: 1px solid #1e3a5f;
      border-radius: 6px;
      padding: 4px 10px;
      color: #94a3b8;
    }

    .telemetry-pill strong {
      color: #38bdf8;
    }

    /* Blocks & Forms */
    .gradio-container .block, .gradio-container .form, .gradio-container .panel {
      background-color: rgba(13, 21, 39, 0.75) !important;
      border-color: var(--line) !important;
      border-radius: 12px !important;
    }

    .gradio-container .tabs {
      border-bottom: 1px solid var(--line) !important;
      margin-bottom: 20px !important;
    }

    .gradio-container .tab-nav button {
      color: var(--ink-soft) !important;
      font-weight: 650 !important;
      font-size: 14px !important;
      border-radius: 8px 8px 0 0 !important;
      padding: 10px 18px !important;
    }

    .gradio-container .tab-nav button.selected {
      color: var(--accent) !important;
      border-color: var(--accent) !important;
      background: rgba(0, 210, 255, 0.05) !important;
    }

    .gradio-container button.primary {
      background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
      border: 1px solid #38bdf8 !important;
      color: #ffffff !important;
      font-weight: 700 !important;
      border-radius: 8px !important;
      box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25) !important;
    }

    .gradio-container button.primary:hover {
      background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%) !important;
      box-shadow: 0 6px 20px rgba(0, 210, 255, 0.35) !important;
    }

    .gradio-container button.secondary {
      background: #111d35 !important;
      border-color: #1e3a5f !important;
      color: var(--ink) !important;
      font-weight: 600 !important;
      border-radius: 8px !important;
    }

    .gradio-container input, .gradio-container textarea, .gradio-container select {
      background: #090f1d !important;
      border-color: #1e3a5f !important;
      color: var(--ink) !important;
      border-radius: 8px !important;
    }

    .gradio-container input:focus, .gradio-container textarea:focus {
      border-color: var(--accent) !important;
      box-shadow: 0 0 0 2px var(--glow) !important;
    }

    .gradio-container .gallery {
      background: #090f1d !important;
      border: 1px solid var(--line) !important;
      border-radius: 12px !important;
      padding: 12px !important;
    }
"""


def get_ui_options() -> tuple[Any, str, str]:
    import gradio as gr

    theme = gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_md,
    ).set(
        body_background_fill="#060a14",
        body_background_fill_dark="#060a14",
        body_text_color="#f1f5f9",
        body_text_color_dark="#f1f5f9",
        body_text_color_subdued="#94a3b8",
        body_text_color_subdued_dark="#94a3b8",
        background_fill_primary="#0d1527",
        background_fill_primary_dark="#0d1527",
        background_fill_secondary="#090f1d",
        background_fill_secondary_dark="#090f1d",
        block_background_fill="#0d1527",
        block_background_fill_dark="#0d1527",
        block_border_color="#1e3a5f",
        block_border_color_dark="#1e3a5f",
        input_background_fill="#090f1d",
        input_background_fill_dark="#090f1d",
        input_border_color="#1e3a5f",
        input_border_color_dark="#1e3a5f",
        button_primary_background_fill="#0284c7",
        button_primary_background_fill_dark="#0284c7",
        button_primary_border_color="#38bdf8",
        button_primary_border_color_dark="#38bdf8",
        panel_background_fill="#0d1527",
        panel_background_fill_dark="#0d1527",
        panel_border_color="#1e3a5f",
        panel_border_color_dark="#1e3a5f",
    )
    return theme, UI_CSS, ""


def create_gradio_app(
    services: Any,
    theme: Any = None,
    css: str | None = None,
    js: str | None = None,
) -> Any:
    import gradio as gr

    with gr.Blocks(
        title="GeoSearch AI | Earth Observation Intelligence",
        theme=theme,
        css=css,
        js=js,
    ) as demo:
        gr.HTML(
            """
            <section class="hero-panel">
              <div class="hero-eyebrow">
                <span class="pulse-dot"></span>
                <span>GEOSEARCH AI — EARTH OBSERVATION PLATFORM</span>
              </div>
              <h1 class="hero-title">GeoSearch AI</h1>
              <p class="hero-subtitle">
                AI-powered satellite & aerial imagery semantic search, multi-temporal change detection, and analyst verification. Operating in fully offline, local on-prem mode.
              </p>
              <div class="telemetry-bar">
                <span class="telemetry-pill"><strong>System Mode:</strong> Fully Offline / Local</span>
                <span class="telemetry-pill"><strong>NLP Layer:</strong> Qwen3 Query Understanding</span>
                <span class="telemetry-pill"><strong>Vision Models:</strong> CLIP + BLIP</span>
                <span class="telemetry-pill"><strong>Vector Store:</strong> Qdrant Embedded HNSW</span>
                <span class="telemetry-pill"><strong>Raster Support:</strong> GeoTIFF / Cloud Optimized GeoTIFF</span>
              </div>
            </section>
            """
        )

        with gr.Tabs():
            with gr.Tab("Semantic Search"):
                build_search_tab(gr, services)
            with gr.Tab("Multi-Temporal Change Detection"):
                build_change_tab(gr, services)
            with gr.Tab("Imagery Archive & Clusters"):
                build_library_tab(gr, services)
            with gr.Tab("Ingest Imagery (GeoTIFF/COG)"):
                build_admin_tab(gr, services)
            with gr.Tab("Security & System Health"):
                build_account_tab(gr, services)

    return demo
