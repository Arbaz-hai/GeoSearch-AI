from __future__ import annotations

import json
import hashlib
import hmac
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

try:
    from sqlalchemy import Float, ForeignKey, String, Text
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

    class SqlAlchemyBase(DeclarativeBase):
        pass

    class Scene(SqlAlchemyBase):
        __tablename__ = "scenes"
        id: Mapped[str] = mapped_column(String, primary_key=True)
        source_id: Mapped[str] = mapped_column(String, index=True)
        source_path: Mapped[str] = mapped_column(Text)
        sensor: Mapped[str] = mapped_column(String)
        acquisition_date: Mapped[str] = mapped_column(String, index=True)
        crs: Mapped[str] = mapped_column(String)
        bounds_json: Mapped[str] = mapped_column(Text)
        transform_json: Mapped[str] = mapped_column(Text)
        metadata_json: Mapped[str] = mapped_column(Text)
        provenance_json: Mapped[str] = mapped_column(Text)
        created_at: Mapped[str] = mapped_column(String)

    class Tile(SqlAlchemyBase):
        __tablename__ = "tiles"
        id: Mapped[str] = mapped_column(String, primary_key=True)
        scene_id: Mapped[str] = mapped_column(ForeignKey("scenes.id"), index=True)
        grid_row: Mapped[int] = mapped_column()
        grid_col: Mapped[int] = mapped_column()
        tile_path: Mapped[str] = mapped_column(Text)
        preview_path: Mapped[str] = mapped_column(Text)
        crs: Mapped[str] = mapped_column(String)
        bounds_json: Mapped[str] = mapped_column(Text)
        transform_json: Mapped[str] = mapped_column(Text)
        valid_fraction: Mapped[float] = mapped_column(Float)
        semantic_point_id: Mapped[str] = mapped_column(String)
        visual_point_id: Mapped[str] = mapped_column(String)
        description: Mapped[str] = mapped_column(Text, default="")
        provenance_json: Mapped[str] = mapped_column(Text)
        created_at: Mapped[str] = mapped_column(String)

    class ChangeEvent(SqlAlchemyBase):
        __tablename__ = "change_events"
        id: Mapped[str] = mapped_column(String, primary_key=True)
        before_tile_id: Mapped[str] = mapped_column(ForeignKey("tiles.id"))
        after_tile_id: Mapped[str] = mapped_column(ForeignKey("tiles.id"))
        change_type: Mapped[str] = mapped_column(String)
        confidence: Mapped[float] = mapped_column(Float)
        earliest_detected_date: Mapped[str] = mapped_column(String)
        mask_path: Mapped[str | None] = mapped_column(Text, nullable=True)
        status: Mapped[str] = mapped_column(String, index=True)
        provenance_json: Mapped[str] = mapped_column(Text)
        created_at: Mapped[str] = mapped_column(String)
        updated_at: Mapped[str] = mapped_column(String)

    class AuditLog(SqlAlchemyBase):
        __tablename__ = "audit_logs"
        id: Mapped[str] = mapped_column(String, primary_key=True)
        entity_type: Mapped[str] = mapped_column(String)
        entity_id: Mapped[str] = mapped_column(String, index=True)
        action: Mapped[str] = mapped_column(String)
        actor: Mapped[str | None] = mapped_column(String, nullable=True)
        payload_json: Mapped[str] = mapped_column(Text)
        created_at: Mapped[str] = mapped_column(String)

    class AnalystDecision(SqlAlchemyBase):
        __tablename__ = "analyst_decisions"
        id: Mapped[str] = mapped_column(String, primary_key=True)
        event_id: Mapped[str] = mapped_column(ForeignKey("change_events.id"), index=True)
        analyst_label: Mapped[str] = mapped_column(String)
        analyst_id: Mapped[str | None] = mapped_column(String, nullable=True)
        note: Mapped[str | None] = mapped_column(Text, nullable=True)
        created_at: Mapped[str] = mapped_column(String)
except ModuleNotFoundError:
    SqlAlchemyBase = object

    class _FallbackScene:
        pass

    class _FallbackTile:
        pass

    class _FallbackChangeEvent:
        pass

    class _FallbackAuditLog:
        pass

    class _FallbackAnalystDecision:
        pass

    
    Scene = _FallbackScene
    Tile = _FallbackTile
    ChangeEvent = _FallbackChangeEvent
    AuditLog = _FallbackAuditLog
    AnalystDecision = _FallbackAnalystDecision


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _load(value: str | None, default: Any) -> Any:
    return json.loads(value) if value else default


class MetadataStore:
    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS scenes (
          id TEXT PRIMARY KEY,
          source_id TEXT NOT NULL,
          source_path TEXT NOT NULL,
          sensor TEXT NOT NULL,
          acquisition_date TEXT NOT NULL,
          crs TEXT NOT NULL,
          bounds_json TEXT NOT NULL,
          transform_json TEXT NOT NULL,
          metadata_json TEXT NOT NULL,
          provenance_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          UNIQUE(source_id, acquisition_date)
        );
        CREATE TABLE IF NOT EXISTS tiles (
          id TEXT PRIMARY KEY,
          scene_id TEXT NOT NULL REFERENCES scenes(id),
          grid_row INTEGER NOT NULL,
          grid_col INTEGER NOT NULL,
          tile_path TEXT NOT NULL,
          preview_path TEXT NOT NULL,
          crs TEXT NOT NULL,
          bounds_json TEXT NOT NULL,
          transform_json TEXT NOT NULL,
          valid_fraction REAL NOT NULL,
          semantic_point_id TEXT NOT NULL,
          visual_point_id TEXT NOT NULL,
          description TEXT NOT NULL DEFAULT '',
          provenance_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          UNIQUE(scene_id, grid_row, grid_col)
        );
        CREATE TABLE IF NOT EXISTS change_events (
          id TEXT PRIMARY KEY,
          before_tile_id TEXT NOT NULL REFERENCES tiles(id),
          after_tile_id TEXT NOT NULL REFERENCES tiles(id),
          change_type TEXT NOT NULL,
          confidence REAL NOT NULL,
          earliest_detected_date TEXT NOT NULL,
          mask_path TEXT,
          status TEXT NOT NULL,
          provenance_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(before_tile_id, after_tile_id)
        );
        CREATE TABLE IF NOT EXISTS analyst_decisions (
          id TEXT PRIMARY KEY,
          event_id TEXT NOT NULL REFERENCES change_events(id),
          analyst_label TEXT NOT NULL,
          analyst_id TEXT,
          note TEXT,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_logs (
          id TEXT PRIMARY KEY,
          entity_type TEXT NOT NULL,
          entity_id TEXT NOT NULL,
          action TEXT NOT NULL,
          actor TEXT,
          payload_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS users (
          id TEXT PRIMARY KEY,
          username TEXT NOT NULL COLLATE NOCASE UNIQUE,
          password_salt TEXT NOT NULL,
          password_hash TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tiles_scene_grid ON tiles(scene_id, grid_row, grid_col);
        CREATE INDEX IF NOT EXISTS idx_scenes_date ON scenes(acquisition_date);
        CREATE INDEX IF NOT EXISTS idx_changes_status_confidence ON change_events(status, confidence DESC);
        """
        with self.lock, self._connect() as connection:
            connection.executescript(schema)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(tiles)").fetchall()}
            if "description" not in columns:
                connection.execute("ALTER TABLE tiles ADD COLUMN description TEXT NOT NULL DEFAULT ''")

    def _audit(self, connection: sqlite3.Connection, entity_type: str, entity_id: str, action: str, actor: str | None, payload: dict[str, Any]) -> None:
        connection.execute(
            "INSERT INTO audit_logs VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid4()), entity_type, entity_id, action, actor, _dump(payload), _now()),
        )

    def scene_by_source(self, source_id: str, acquisition_date: str) -> dict[str, Any] | None:
        with self.lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM scenes WHERE source_id=? AND acquisition_date=?", (source_id, acquisition_date)).fetchone()
        return self._scene_record(row) if row else None

    def add_scene(self, record: dict[str, Any]) -> dict[str, Any]:
        existing = self.scene_by_source(record["source_id"], record["acquisition_date"])
        if existing:
            return existing
        scene_id = record.get("id") or str(uuid4())
        created_at = _now()
        values = (
            scene_id,
            record["source_id"],
            record["source_path"],
            record["sensor"],
            record["acquisition_date"],
            record.get("crs", "EPSG:4326"),
            _dump(record.get("bounds", [])),
            _dump(record.get("transform", [])),
            _dump(record.get("metadata", {})),
            _dump(record.get("provenance", {})),
            created_at,
        )
        with self.lock, self._connect() as connection:
            connection.execute("INSERT INTO scenes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
            self._audit(connection, "scene", scene_id, "ingested", None, {"source_id": record["source_id"]})
        return self.get_scene(scene_id) or {}

    def get_scene(self, scene_id: str) -> dict[str, Any] | None:
        with self.lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM scenes WHERE id=? OR source_id=? LIMIT 1", (scene_id, scene_id)).fetchone()
        return self._scene_record(row) if row else None

    def list_scenes(self, start_date: str | None = None, end_date: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM scenes WHERE 1=1"
        values: list[Any] = []
        if start_date:
            query += " AND acquisition_date>=?"
            values.append(start_date)
        if end_date:
            query += " AND acquisition_date<=?"
            values.append(end_date)
        query += " ORDER BY acquisition_date"
        with self.lock, self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._scene_record(row) for row in rows]

    def add_tile(self, record: dict[str, Any]) -> dict[str, Any]:
        tile_id = record.get("id") or str(uuid4())
        values = (
            tile_id,
            record["scene_id"],
            int(record["grid_row"]),
            int(record["grid_col"]),
            record["tile_path"],
            record["preview_path"],
            record.get("crs", "EPSG:4326"),
            _dump(record.get("bounds", [])),
            _dump(record.get("transform", [])),
            float(record.get("valid_fraction", 0)),
            record["semantic_point_id"],
            record["visual_point_id"],
            record.get("description", ""),
            _dump(record.get("provenance", {})),
            _now(),
        )
        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO tiles (
                  id, scene_id, grid_row, grid_col, tile_path, preview_path, crs,
                  bounds_json, transform_json, valid_fraction, semantic_point_id,
                  visual_point_id, description, provenance_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            self._audit(connection, "tile", tile_id, "indexed", None, {"scene_id": record["scene_id"]})
        return self.get_tile(tile_id) or {}

    @staticmethod
    def _user_record(row: sqlite3.Row) -> dict[str, Any]:
        return {"id": row["id"], "username": row["username"], "created_at": row["created_at"]}

    @staticmethod
    def _password_hash(password: str, salt: bytes) -> str:
        return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1).hex()

    def create_user(self, username: str, password: str) -> dict[str, Any]:
        name = username.strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]{3,64}", name):
            raise ValueError("Username must be 3–64 letters, numbers, dots, underscores, or hyphens.")
        if len(password) < 8:
            raise ValueError("Password must contain at least 8 characters.")
        salt = os.urandom(16)
        user_id = str(uuid4())
        with self.lock, self._connect() as connection:
            try:
                connection.execute(
                    "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                    (user_id, name, salt.hex(), self._password_hash(password, salt), _now()),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("That username is already in use.") from error
            self._audit(connection, "user", user_id, "created", user_id, {"username": name})
            row = connection.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return self._user_record(row)

    def authenticate_user(self, username: str, password: str) -> dict[str, Any] | None:
        with self.lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE username=?", (username.strip(),)).fetchone()
        if not row:
            return None
        candidate = self._password_hash(password, bytes.fromhex(row["password_salt"]))
        return self._user_record(row) if hmac.compare_digest(candidate, row["password_hash"]) else None

    def get_tile(self, tile_id: str) -> dict[str, Any] | None:
        with self.lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM tiles WHERE id=?", (tile_id,)).fetchone()
        return self._tile_record(row) if row else None

    def list_tiles(self, scene_id: str | None = None) -> list[dict[str, Any]]:
        if not scene_id:
            query = "SELECT * FROM tiles ORDER BY created_at"
            values: tuple[Any, ...] = ()
        else:
            query = """
            SELECT tiles.* FROM tiles
            LEFT JOIN scenes ON scenes.id = tiles.scene_id
            WHERE tiles.scene_id = ? OR scenes.source_id = ?
            ORDER BY tiles.created_at
            """
            values = (scene_id, scene_id)
        with self.lock, self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._tile_record(row) for row in rows]

    def tiles_at_grid(self, grid_row: int, grid_col: int, start_date: str | None = None, end_date: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT tiles.* FROM tiles JOIN scenes ON scenes.id=tiles.scene_id WHERE grid_row=? AND grid_col=?"
        values: list[Any] = [grid_row, grid_col]
        if start_date:
            query += " AND scenes.acquisition_date>=?"
            values.append(start_date)
        if end_date:
            query += " AND scenes.acquisition_date<=?"
            values.append(end_date)
        query += " ORDER BY scenes.acquisition_date"
        with self.lock, self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._tile_record(row) for row in rows]

    def add_change_event(self, record: dict[str, Any]) -> dict[str, Any]:
        existing = self.find_change_event(record["before_tile_id"], record["after_tile_id"])
        event_id = existing["id"] if existing else record.get("id") or str(uuid4())
        timestamp = _now()
        values = (
            event_id,
            record["before_tile_id"],
            record["after_tile_id"],
            record["change_type"],
            float(record["confidence"]),
            record["earliest_detected_date"],
            record.get("mask_path"),
            record.get("status", "pending"),
            _dump(record.get("provenance", {})),
            existing["created_at"] if existing else timestamp,
            timestamp,
        )
        with self.lock, self._connect() as connection:
            connection.execute("INSERT OR REPLACE INTO change_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
            self._audit(connection, "change_event", event_id, "detected", None, {"confidence": record["confidence"]})
        return self.get_change_event(event_id) or {}

    def find_change_event(self, before_tile_id: str, after_tile_id: str) -> dict[str, Any] | None:
        with self.lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM change_events WHERE before_tile_id=? AND after_tile_id=?", (before_tile_id, after_tile_id)).fetchone()
        return self._change_record(row) if row else None

    def get_change_event(self, event_id: str) -> dict[str, Any] | None:
        with self.lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM change_events WHERE id=?", (event_id,)).fetchone()
        return self._change_record(row) if row else None

    def pending_change_events(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.lock, self._connect() as connection:
            rows = connection.execute("SELECT * FROM change_events WHERE status='pending' ORDER BY confidence DESC LIMIT ?", (limit,)).fetchall()
        return [self._change_record(row) for row in rows]

    def record_decision(self, event_id: str, analyst_label: str, analyst_id: str | None = None, note: str | None = None) -> dict[str, Any]:
        if analyst_label not in {"confirm", "reject"} and not analyst_label.startswith("relabel:"):
            raise ValueError("analyst_label must be confirm, reject, or relabel:<type>")
        event = self.get_change_event(event_id)
        if event is None:
            raise KeyError(event_id)
        with self.lock, self._connect() as connection:
            decision_id = str(uuid4())
            connection.execute("INSERT INTO analyst_decisions VALUES (?, ?, ?, ?, ?, ?)", (decision_id, event_id, analyst_label, analyst_id, note, _now()))
            status = "confirmed" if analyst_label == "confirm" else "rejected" if analyst_label == "reject" else "relabelled"
            connection.execute("UPDATE change_events SET status=?, updated_at=? WHERE id=?", (status, _now(), event_id))
            self._audit(connection, "change_event", event_id, "analyst_decision", analyst_id, {"label": analyst_label, "note": note})
        return {"id": decision_id, "event_id": event_id, "analyst_label": analyst_label, "analyst_id": analyst_id, "note": note, "timestamp": _now()}

    def audit_log(self, entity_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM audit_logs"
        values: tuple[Any, ...] = ()
        if entity_id:
            query += " WHERE entity_id=?"
            values = (entity_id,)
        query += " ORDER BY created_at"
        with self.lock, self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [{**dict(row), "payload": _load(row["payload_json"], {})} for row in rows]

    def feedback_bias(self, tile_id: str) -> float:
        query = """
        SELECT analyst_decisions.analyst_label
        FROM analyst_decisions
        JOIN change_events ON analyst_decisions.event_id=change_events.id
        WHERE change_events.before_tile_id=? OR change_events.after_tile_id=?
        """
        with self.lock, self._connect() as connection:
            labels = [row[0] for row in connection.execute(query, (tile_id, tile_id)).fetchall()]
        if not labels:
            return 0.0
        values = [1.0 if label == "confirm" else -1.0 if label == "reject" else 0.2 for label in labels]
        return 0.05 * sum(values) / len(values)

    def statistics(self) -> dict[str, Any]:
        with self.lock, self._connect() as connection:
            scene_count = connection.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
            tile_count = connection.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
            pending_count = connection.execute("SELECT COUNT(*) FROM change_events WHERE status='pending'").fetchone()[0]
            last_ingested = connection.execute("SELECT MAX(created_at) FROM scenes").fetchone()[0]
        return {"scene_count": scene_count, "tile_count": tile_count, "pending_change_count": pending_count, "last_ingested": last_ingested}

    @staticmethod
    def _scene_record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["bounds"] = _load(record.pop("bounds_json"), [])
        record["transform"] = _load(record.pop("transform_json"), [])
        record["metadata"] = _load(record.pop("metadata_json"), {})
        record["provenance"] = _load(record.pop("provenance_json"), {})
        return record

    @staticmethod
    def _tile_record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["bounds"] = _load(record.pop("bounds_json"), [])
        record["transform"] = _load(record.pop("transform_json"), [])
        record["provenance"] = _load(record.pop("provenance_json"), {})
        return record

    @staticmethod
    def _change_record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["provenance"] = _load(record.pop("provenance_json"), {})
        return record
