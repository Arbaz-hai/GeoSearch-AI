from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import create_services
from app.settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally ingest one scene without rebuilding existing vectors")
    parser.add_argument("scene", type=Path)
    parser.add_argument("--source-id")
    arguments = parser.parse_args()
    services = create_services(load_settings())
    result = services.ingest.ingest_scene(arguments.scene, arguments.source_id, progress=print)
    print(result)


if __name__ == "__main__":
    main()
