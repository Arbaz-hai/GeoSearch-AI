from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import uvicorn
from app.settings import load_settings


def print_banner(settings) -> None:
    db_stats = settings.database_path
    print("=" * 70)
    print("      GeoSearch AI — Earth Observation Semantic Search & Change AI  ")
    print("=" * 70)
    print(f"  Mode           : OFFLINE-FIRST (Network access disabled)")
    print(f"  Local SQLite   : {db_stats}")
    print(f"  Qdrant Path    : {settings.qdrant_path}")
    print(f"  Model Staging  : {settings.model_dir}")
    
    print("\n  Offline Model Inventory:")
    for model_name in ["clip", "blip", "qwen3", "remoteclip", "prithvi"]:
        p = settings.model_path(model_name)
        staged = p.exists() and p.is_dir() and any(p.iterdir())
        st_label = "[READY]" if staged else "[FALLBACK MODE]"
        print(f"   • {model_name:<14} : {st_label:<16} ({p})")
    
    print("\n  Web Interface  : http://127.0.0.1:7860")
    print("  REST API Docs  : http://127.0.0.1:7860/api/docs")
    print("  Health Check   : http://127.0.0.1:7860/api/health")
    print("=" * 70)
    print("  Starting server...\n")


def main() -> None:
    settings = load_settings()
    print_banner(settings)

    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "127.0.0.1")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
