from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.training_data.schema import feedback_from_event


def _blend_overlay(base_path: str | None, mask_path: str | None, opacity: float) -> Any:
    """Blend the change mask over the satellite observation with given opacity."""
    if not base_path or not Path(base_path).exists():
        return None
    try:
        base = Image.open(base_path).convert("RGBA")
        if not mask_path or not Path(mask_path).exists() or opacity <= 0:
            return base.convert("RGB")

        mask = Image.open(mask_path).convert("RGBA")
        mask = mask.resize(base.size, Image.Resampling.BILINEAR)

        r, g, b, a = mask.split()
        a = a.point(lambda p: int(p * opacity))
        mask.putalpha(a)

        blended = Image.alpha_composite(base, mask)
        return blended.convert("RGB")
    except Exception:
        return base_path


def build_change_tab(gr: Any, services: Any) -> None:
    gr.Markdown(
        """
        ### Multi-Temporal Change Detection & Analyst Review
        Compare two satellite observations of the same scene over time to identify structural, environmental, and infrastructure changes (appearance, clearance, expansion, construction, water bodies).
        """
    )

    with gr.Tab("Compare Observations"):
        with gr.Row():
            before_upload = gr.File(
                label="Earlier Observation (Before)",
                file_count="single",
                type="filepath",
            )
            after_upload = gr.File(
                label="Later Observation (After)",
                file_count="single",
                type="filepath",
            )
        
        with gr.Row():
            run_button = gr.Button("Analyze Multi-Temporal Changes", variant="primary", scale=2)
            opacity_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.65,
                step=0.05,
                label="Change Mask Overlay Opacity",
                scale=1,
            )

        comparison_status = gr.Markdown("Select or upload both observations to start multi-temporal alignment and change classification.")

        gr.Markdown("#### Visual Change Evidence")
        with gr.Row():
            disp_before = gr.Image(label="Earlier Observation (T1)", interactive=False)
            disp_after = gr.Image(label="Later Observation (T2)", interactive=False)
            disp_mask = gr.Image(label="Change Detection Mask (Red = Changed Area)", interactive=False)

        change_metrics_card = gr.Markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1e293b; border-radius: 12px; padding: 16px;">
              <p style="color: #94a3b8; margin: 0;">Run comparison to display change classification, confidence metrics, and estimated area affected.</p>
            </div>
            """
        )

        event_state = gr.State([])
        cached_before_path = gr.State(None)
        cached_after_path = gr.State(None)
        cached_mask_path = gr.State(None)

        gr.Markdown("#### Ask Questions About These Changes")
        with gr.Row():
            question = gr.Textbox(
                label="Question on Detection Evidence (Qwen3 Grounded Answer)",
                placeholder="e.g. 'When was the earliest change detected?', 'What is the confidence of the detected change?', 'How much area was changed?'",
                scale=4,
            )
            ask_button = gr.Button("Get Answer", variant="secondary", scale=1)
        answer = gr.Markdown("Answers are grounded strictly in the computed multi-temporal evidence.")

    with gr.Tab("Analyst Review Queue"):
        gr.Markdown(
            """
            #### Analyst Review & False-Alarm Suppression
            Review flagged change candidates. Confirm valid changes or reject false alarms (clouds, shadows, registration noise). Every decision updates the persistent audit trail and drives local reranking.
            """
        )
        with gr.Row():
            refresh_queue = gr.Button("Refresh Queue", variant="secondary", scale=1)
            review_event = gr.Dropdown(label="Pending Change Candidates", choices=[], interactive=True, scale=3)
            analyst_id = gr.Textbox(label="Analyst Identifier", placeholder="analyst_01", value="analyst_admin", scale=2)

        review_state = gr.State([])
        with gr.Row():
            review_before = gr.Image(label="Earlier Observation (T1)", interactive=False)
            review_after = gr.Image(label="Later Observation (T2)", interactive=False)
            review_mask = gr.Image(label="Detected Change Mask", interactive=False)

        review_detail = gr.Markdown("Select a candidate from the review queue.")
        review_note = gr.Textbox(
            label="Review Note / Rationale",
            placeholder="e.g. 'Confirmed new road construction widening' or 'Rejected: seasonal vegetation fluctuation'",
        )

        with gr.Row():
            accept_button = gr.Button("Accept (Confirm Change)", variant="primary")
            reject_button = gr.Button("Reject (False Alarm)", variant="secondary")

        review_status = gr.Markdown()

        gr.Markdown("#### Persistent Audit Trail Log")
        audit_table = gr.DataFrame(
            headers=["Timestamp", "Entity", "Action", "Actor", "Details"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
        )


    def execute_comparison(before_upload_path: str | None, after_upload_path: str | None, opacity: float, progress=gr.Progress()):
        if not before_upload_path or not after_upload_path:
            return None, None, None, "Please upload both earlier and later imagery to compare.", _empty_metrics(), [], None, None, None

        try:
            progress(0.15, desc="Ingesting and indexing T1 observation")
            before_res = services.ingest.ingest_scene(Path(before_upload_path), trigger_change=False)
            progress(0.40, desc="Ingesting and indexing T2 observation")
            after_res = services.ingest.ingest_scene(Path(after_upload_path), trigger_change=False)
        except Exception as err:
            return None, None, None, f"Ingestion error: {err}", _empty_metrics(), [], None, None, None

        progress(0.65, desc="Co-registering and computing change classification")
        before_tiles = services.metadata.list_tiles(before_res["scene"]["id"])
        after_tiles = services.metadata.list_tiles(after_res["scene"]["id"])
        after_by_grid = {(t["grid_row"], t["grid_col"]): t for t in after_tiles}

        events = []
        for b in before_tiles:
            a = after_by_grid.get((b["grid_row"], b["grid_col"]))
            if a:
                ev = services.change._compare(b, a, [b, a])
                if ev:
                    events.append(ev)

        progress(1.0, desc="Comparison complete")

        if not events:
            before_preview = (before_res["scene"].get("metadata") or {}).get("preview_path")
            after_preview = (after_res["scene"].get("metadata") or {}).get("preview_path")
            status = "Comparison completed: No significant changes exceeded the confidence threshold (false alarms suppressed)."
            return before_preview, after_preview, None, status, _empty_metrics(), [], before_preview, after_preview, None

        top_event = events[0]
        before_tile = services.metadata.get_tile(top_event["before_tile_id"])
        after_tile = services.metadata.get_tile(top_event["after_tile_id"])

        before_img = before_tile["preview_path"] if before_tile else None
        after_img = after_tile["preview_path"] if after_tile else None
        
        mask_img = top_event.get("mask_preview_path")
        if not mask_img and top_event.get("mask_path"):
            cand = Path(top_event["mask_path"]).with_name(Path(top_event["mask_path"]).stem + "_preview.png")
            if cand.exists():
                mask_img = str(cand)

        blended_after = _blend_overlay(after_img, mask_img, opacity)

        metrics_html = _format_metrics(top_event, before_res["scene"], after_res["scene"])
        summary = f"Detected **{len(events)}** change candidate(s). Top category: **{top_event['change_type'].replace('_', ' ').title()}** with **{top_event['confidence']:.0%}** confidence."

        return before_img, blended_after, mask_img, summary, metrics_html, events, before_img, after_img, mask_img

    def update_opacity(opacity: float, after_img_path: str | None, mask_img_path: str | None):
        if not after_img_path:
            return None
        return _blend_overlay(after_img_path, mask_img_path, opacity)

    def ask_change_question(q: str, evs: list[dict[str, Any]]):
        if not q or not q.strip():
            return "Please type a question regarding the detected changes."
        return services.change.answer_question(q, evs)


    def load_queue_data():
        queue = services.change.review_queue()
        choices = [
            (
                f"{item.get('after_date', 'Undated')} · {item['change_type'].replace('_', ' ').title()} ({item['confidence']:.0%})",
                item["id"],
            )
            for item in queue
        ]
        val = queue[0]["id"] if queue else None
        status = (
            "No pending candidates. All items have been reviewed."
            if not queue
            else f"Loaded **{len(queue)}** pending candidate(s) awaiting analyst verification."
        )

        b_img, a_img, m_img, detail = (None, None, None, "No pending items.")
        if queue:
            b_img, a_img, m_img, detail = _format_review_item(queue[0])

        audit_data = _fetch_audit_records()
        return (
            gr.update(choices=choices, value=val),
            queue,
            status,
            b_img,
            a_img,
            m_img,
            detail,
            audit_data,
        )

    def select_review_candidate(event_id: str | None, queue: list[dict[str, Any]]):
        item = next((c for c in queue if c["id"] == event_id), None)
        if not item:
            return None, None, None, "Select a pending candidate."
        return _format_review_item(item)

    def process_decision(event_id: str | None, analyst: str, note: str, label: str):
        if not event_id:
            return "Select a candidate first.", gr.update(), [], None, None, None, "", []

        analyst_name = analyst.strip() or "analyst_admin"
        try:
            services.metadata.record_decision(
                event_id,
                label,
                analyst_name,
                note.strip() or f"{label.title()} via analyst review",
            )
        except Exception as err:
            return f"Failed to record decision: {err}", gr.update(), [], None, None, None, "", []

        new_queue = services.change.review_queue()
        choices = [
            (
                f"{item.get('after_date', 'Undated')} · {item['change_type'].replace('_', ' ').title()} ({item['confidence']:.0%})",
                item["id"],
            )
            for item in new_queue
        ]
        next_val = new_queue[0]["id"] if new_queue else None

        b_img, a_img, m_img, detail = (None, None, None, "No further pending items.")
        if new_queue:
            b_img, a_img, m_img, detail = _format_review_item(new_queue[0])

        status_msg = f"Recorded **{label.upper()}** for change `{event_id[:8]}...` by **{analyst_name}**. Queue updated in real-time."
        audit_data = _fetch_audit_records()

        return (
            status_msg,
            gr.update(choices=choices, value=next_val),
            new_queue,
            b_img,
            a_img,
            m_img,
            detail,
            audit_data,
        )

    def _fetch_audit_records() -> list[list[str]]:
        logs = services.metadata.audit_log()[-20:]
        rows = []
        for entry in reversed(logs):
            payload_str = ", ".join(f"{k}: {v}" for k, v in (entry.get("payload") or {}).items())
            rows.append([
                entry.get("created_at", "")[:19].replace("T", " "),
                entry.get("entity_type", ""),
                entry.get("action", ""),
                entry.get("actor") or "System",
                payload_str,
            ])
        return rows

    run_button.click(
        execute_comparison,
        [before_upload, after_upload, opacity_slider],
        [disp_before, disp_after, disp_mask, comparison_status, change_metrics_card, event_state, cached_before_path, cached_after_path, cached_mask_path],
    )
    opacity_slider.change(
        update_opacity,
        [opacity_slider, cached_after_path, cached_mask_path],
        outputs=disp_after,
    )
    ask_button.click(ask_change_question, [question, event_state], answer)

    refresh_queue.click(
        load_queue_data,
        outputs=[review_event, review_state, review_status, review_before, review_after, review_mask, review_detail, audit_table],
    )
    review_event.change(
        select_review_candidate,
        [review_event, review_state],
        [review_before, review_after, review_mask, review_detail],
    )
    accept_button.click(
        lambda e_id, a_id, note: process_decision(e_id, a_id, note, "confirm"),
        [review_event, analyst_id, review_note],
        [review_status, review_event, review_state, review_before, review_after, review_mask, review_detail, audit_table],
    )
    reject_button.click(
        lambda e_id, a_id, note: process_decision(e_id, a_id, note, "reject"),
        [review_event, analyst_id, review_note],
        [review_status, review_event, review_state, review_before, review_after, review_mask, review_detail, audit_table],
    )


def _empty_metrics() -> str:
    return """
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1e293b; border-radius: 12px; padding: 16px;">
      <p style="color: #94a3b8; margin: 0;">Run comparison to display change classification, confidence metrics, and estimated area affected.</p>
    </div>
    """


def _format_metrics(event: dict[str, Any], before_scene: dict[str, Any], after_scene: dict[str, Any]) -> str:
    category = event.get("change_type", "unknown").replace("_", " ").title()
    conf = event.get("confidence", 0)
    area = event.get("changed_area_pct", 0)
    earliest = event.get("earliest_detected_date", "Not determined")
    before_date = before_scene.get("acquisition_date", "T1")
    after_date = after_scene.get("acquisition_date", "T2")

    return f"""
    <div style="background: #0d1527; border: 1px solid #1e3a5f; border-radius: 12px; padding: 18px; font-family: monospace, sans-serif;">
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px;">
        <div style="border-left: 3px solid #38bdf8; padding-left: 10px;">
          <span style="color: #94a3b8; font-size: 12px;">DETECTED CHANGE CATEGORY</span><br>
          <strong style="color: #38bdf8; font-size: 18px;">{category}</strong>
        </div>
        <div style="border-left: 3px solid #34d399; padding-left: 10px;">
          <span style="color: #94a3b8; font-size: 12px;">DETECTION CONFIDENCE</span><br>
          <strong style="color: #34d399; font-size: 18px;">{conf:.1%}</strong>
        </div>
        <div style="border-left: 3px solid #f59e0b; padding-left: 10px;">
          <span style="color: #94a3b8; font-size: 12px;">CHANGED AREA FRACTION</span><br>
          <strong style="color: #f59e0b; font-size: 18px;">{area:.2f}%</strong>
        </div>
        <div style="border-left: 3px solid #a78bfa; padding-left: 10px;">
          <span style="color: #94a3b8; font-size: 12px;">EARLIEST CONFIRMED DATE</span><br>
          <strong style="color: #a78bfa; font-size: 18px;">{earliest}</strong>
        </div>
      </div>
      <div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid #1e293b; color: #94a3b8; font-size: 13px;">
        Observation Baseline: <strong style="color: #f8fafc;">{before_date}</strong> ➔ Evaluation: <strong style="color: #f8fafc;">{after_date}</strong>
      </div>
    </div>
    """


def _format_review_item(item: dict[str, Any]) -> tuple[Any, Any, Any, str]:
    b_img = item.get("before_preview")
    a_img = item.get("after_preview")
    m_img = item.get("mask_preview")

    detail = f"""
    <div style="background: #0d1527; border: 1px solid #1e3a5f; border-radius: 10px; padding: 14px; font-family: monospace;">
      <h4 style="color: #38bdf8; margin: 0 0 8px 0;">Candidate: {item.get('change_type', '').replace('_', ' ').title()}</h4>
      <div><strong style="color: #94a3b8;">Event ID:</strong> <code style="color: #fcd34d;">{item.get('id')}</code></div>
      <div><strong style="color: #94a3b8;">Confidence:</strong> <span style="color: #34d399;">{item.get('confidence', 0):.1%}</span></div>
      <div><strong style="color: #94a3b8;">Baseline Date:</strong> {item.get('before_date', 'Unknown')}</div>
      <div><strong style="color: #94a3b8;">Evaluation Date:</strong> {item.get('after_date', 'Unknown')}</div>
      <div><strong style="color: #94a3b8;">Earliest Supported Appearance:</strong> <span style="color: #a78bfa;">{item.get('earliest_detected_date', 'Unknown')}</span></div>
      <div><strong style="color: #94a3b8;">Sensor Platform:</strong> {item.get('sensor', 'Unknown')}</div>
      <div><strong style="color: #94a3b8;">Coordinates / Bounds:</strong> <code style="color: #cbd5e1;">{item.get('location')}</code></div>
    </div>
    """
    return b_img, a_img, m_img, detail
