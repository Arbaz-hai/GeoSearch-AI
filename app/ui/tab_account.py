from __future__ import annotations

from typing import Any
from app.auth import create_token


def build_account_tab(gr: Any, services: Any) -> None:
    gr.Markdown(
        """
        ### Authentication, Security & Local System Telemetry
        Manage local analyst accounts, inspect offline tokens, and review model weights and database storage state.
        Passwords are encrypted locally using salted `scrypt` hashing (no external auth services).
        """
    )

    session_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Sign In")
            login_name = gr.Textbox(label="Analyst Username", placeholder="e.g. analyst_01")
            login_password = gr.Textbox(label="Password", type="password")
            with gr.Row():
                login_btn = gr.Button("Sign In", variant="primary")
                logout_btn = gr.Button("Sign Out", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("#### Create Local Analyst Account")
            signup_name = gr.Textbox(label="New Username", placeholder="3–64 characters")
            signup_password = gr.Textbox(label="New Password", type="password")
            signup_btn = gr.Button("Create Account", variant="secondary")

    auth_status = gr.Markdown("Status: **Not signed in** (Guest Session)")

    gr.Markdown("---")
    gr.Markdown("#### Active Session & API Token")
    with gr.Row():
        token_display = gr.Textbox(
            label="Local HMAC Bearer Token (for authenticated REST API requests)",
            interactive=False,
            placeholder="Sign in to generate a bearer token",
            lines=2,
            scale=3,
        )
        refresh_sys_btn = gr.Button("Refresh Telemetry", variant="secondary", scale=1)

    gr.Markdown("---")
    gr.Markdown("#### Offline Model Staging & Storage Status")
    system_status_card = gr.Markdown()

    def do_login(username: str, password: str):
        if not username or not password:
            return "Please enter both username and password.", None, ""
        account = services.metadata.authenticate_user(username, password)
        if not account:
            return "[Error] Authentication failed: Invalid username or password.", None, ""
        token = create_token(account["id"], account["username"])
        msg = f"[Success] Signed in as **{account['username']}** (User ID: `{account['id']}`). Active local session authenticated."
        return msg, account, token

    def do_signup(username: str, password: str):
        if not username or not password:
            return "Please specify both a username and password.", None, ""
        try:
            account = services.metadata.create_user(username, password)
        except ValueError as err:
            return f"[Error] Registration failed: {err}", None, ""
        token = create_token(account["id"], account["username"])
        msg = f"[Success] Account created and authenticated for **{account['username']}**."
        return msg, account, token

    def do_logout():
        return "Signed out. Session invalidated.", None, ""

    def load_system_telemetry():
        models = [
            ("CLIP Text & Image Embedder", services.settings.model_path("clip")),
            ("BLIP Image Captioner", services.settings.model_path("blip")),
            ("Qwen3 Query NLP & Q&A", services.settings.model_path("qwen3")),
            ("RemoteCLIP (Domain CLIP)", services.settings.model_path("remoteclip")),
            ("Prithvi Multispectral Backbone", services.settings.model_path("prithvi")),
        ]
        
        m_rows = []
        for name, p in models:
            staged = p.exists() and p.is_dir() and any(p.iterdir())
            staged_html = "<span style='color: #34d399; font-weight: 700;'>STAGED</span>"
            not_staged_html = "<span style='color: #f87171; font-weight: 700;'>NOT STAGED</span>"
            st_badge = staged_html if staged else not_staged_html
            m_rows.append(f"<tr><td style='padding: 6px 12px;'>{name}</td><td style='padding: 6px 12px;'>{st_badge}</td><td style='padding: 6px 12px;'><code>{p}</code></td></tr>")

        db_stats = services.metadata.statistics()
        vec_stats = services.vectors.stats()

        return f"""
        <div style="background: #0d1527; border: 1px solid #1e3a5f; border-radius: 12px; padding: 18px; font-family: monospace, sans-serif;">
          <h4 style="color: #38bdf8; margin-top: 0;">Local Model Staging Registry</h4>
          <table style="width: 100%; border-collapse: collapse; font-size: 13px; color: #f8fafc;">
            <thead>
              <tr style="border-bottom: 1px solid #1e3a5f; text-align: left; color: #94a3b8;">
                <th style="padding: 6px 12px;">Component</th>
                <th style="padding: 6px 12px;">Offline State</th>
                <th style="padding: 6px 12px;">Local Storage Target</th>
              </tr>
            </thead>
            <tbody>
              {''.join(m_rows)}
            </tbody>
          </table>

          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 16px; border-top: 1px solid #1e3a5f; padding-top: 14px;">
            <div><strong style="color: #94a3b8;">Indexed Scenes:</strong> <span style="color: #38bdf8;">{db_stats.get('scene_count', 0)}</span></div>
            <div><strong style="color: #94a3b8;">Spatial Tiles:</strong> <span style="color: #38bdf8;">{db_stats.get('tile_count', 0)}</span></div>
            <div><strong style="color: #94a3b8;">Pending Reviews:</strong> <span style="color: #f59e0b;">{db_stats.get('pending_change_count', 0)}</span></div>
            <div><strong style="color: #94a3b8;">Vector Engine:</strong> <span style="color: #34d399;">{vec_stats.get('backend', 'Qdrant Local')}</span></div>
            <div><strong style="color: #94a3b8;">Semantic Vectors:</strong> <span style="color: #a78bfa;">{vec_stats.get('semantic_vectors', 0)}</span></div>
          </div>
        </div>
        """

    login_btn.click(do_login, [login_name, login_password], [auth_status, session_state, token_display])
    signup_btn.click(do_signup, [signup_name, signup_password], [auth_status, session_state, token_display])
    logout_btn.click(do_logout, outputs=[auth_status, session_state, token_display])
    refresh_sys_btn.click(load_system_telemetry, outputs=system_status_card)


