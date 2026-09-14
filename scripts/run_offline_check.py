from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from app.services import create_services
from app.settings import load_settings
from app.utils.licenses import model_staging_status


def main() -> None:
    settings = load_settings()
    settings.offline_mode = True
    services = create_services(settings)
    search_results = services.search.search_text("local offline verification", limit=3)
    change_results = services.change.analyze(settings.default_date_start, settings.default_date_end)
    report = {
        "offline_environment": {"HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"], "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"]},
        "model_staging": model_staging_status(settings),
        "metadata": services.metadata.statistics(),
        "vectors": services.vectors.stats(),
        "search_smoke_result_count": len(search_results),
        "change_smoke_result_count": len(change_results),
        "status": "passed",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
