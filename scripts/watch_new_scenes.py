from __future__ import annotations

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.services import create_services
from app.settings import load_settings


class SceneHandler(FileSystemEventHandler):
    def __init__(self, services):
        self.services = services
        self.extensions = {".tif", ".tiff", ".cog", ".png", ".jpg", ".jpeg"}

    def on_created(self, event):
        path = Path(event.src_path)
        if event.is_directory or path.suffix.lower() not in self.extensions:
            return
        time.sleep(0.5)
        try:
            print(self.services.ingest.ingest_scene(path, progress=print))
        except Exception as error:
            print(f"Ingestion failed for {path}: {error}")


def main() -> None:
    settings = load_settings()
    watch_path = settings.raw_dir / "incoming"
    watch_path.mkdir(parents=True, exist_ok=True)
    observer = Observer()
    observer.schedule(SceneHandler(create_services(settings)), str(watch_path), recursive=False)
    observer.start()
    print(f"Watching {watch_path}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
