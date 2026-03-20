"""
Thin SQLite wrapper for persisting job records across server restarts.
Uses Python's built-in sqlite3 — no extra dependency needed.
"""
from __future__ import annotations

import datetime
import json
import sqlite3
import threading
from typing import Any


_CREATE = """
CREATE TABLE IF NOT EXISTS jobs (
    id                  TEXT PRIMARY KEY,
    status              TEXT NOT NULL DEFAULT 'queued',
    progress            REAL DEFAULT 0.0,
    message             TEXT,
    video_path          TEXT NOT NULL,
    result_path         TEXT NOT NULL,
    csv_perframe_path   TEXT NOT NULL,
    csv_summary_path    TEXT NOT NULL,
    engine_json         TEXT NOT NULL,
    meta_json           TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    experiment_id       TEXT DEFAULT '',
    video_filename      TEXT DEFAULT ''
)
"""

# Columns added in later versions — applied via ALTER TABLE if missing
_EXTRA_COLUMNS: list[tuple[str, str]] = [
    ("experiment_id", "TEXT DEFAULT ''"),
    ("video_filename", "TEXT DEFAULT ''"),
]


class JobDatabase:
    """Thread-safe SQLite job store."""

    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(_CREATE)
            # Migrate: add columns that didn't exist in older DB versions
            existing = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
            for col_name, col_def in _EXTRA_COLUMNS:
                if col_name not in existing:
                    try:
                        conn.execute(f"ALTER TABLE jobs ADD COLUMN {col_name} {col_def}")
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def insert(
        self,
        *,
        job_id: str,
        video_path: str,
        result_path: str,
        csv_perframe: str,
        csv_summary: str,
        engine_json: str,
        meta_json: str,
        video_filename: str = "",
    ) -> None:
        created = datetime.datetime.utcnow().isoformat()
        try:
            meta = json.loads(meta_json)
        except Exception:
            meta = {}
        experiment_id = meta.get("experiment_id", "")
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO jobs
                  (id, status, progress, message, video_path, result_path,
                   csv_perframe_path, csv_summary_path, engine_json, meta_json,
                   created_at, experiment_id, video_filename)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job_id, "queued", 0.0, None,
                    video_path, result_path, csv_perframe, csv_summary,
                    engine_json, meta_json,
                    created, experiment_id, video_filename,
                ),
            )

    def update(self, job_id: str, status: str, progress: float, message: str | None) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, progress=?, message=? WHERE id=?",
                (status, progress, message, job_id),
            )

    def delete(self, job_id: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def all_jobs(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return dict(row) if row else None
