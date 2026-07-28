"""SQLite storage for the three-layer ETL design.

Layer 1 (raw):        one table per source, immutable, stores raw JSON.
Layer 2 (standardized): the single ``meeting`` table, one canonical schema.
Layer 3 (research):   ``canonical_meeting`` + ``meeting_analysis``.

The design principle: raw data is never modified. Standardized and research
tables are always derived and can be dropped and rebuilt at any time.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .models import DEDUPER_VERSION, Meeting, RawRecord

SCHEMA = """
-- ---------- Layer 1: RAW (immutable, one table per source) ----------
CREATE TABLE IF NOT EXISTS raw_meetingbank (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id      TEXT NOT NULL UNIQUE,
    raw_json       TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    imported_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_localview (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id      TEXT NOT NULL UNIQUE,
    raw_json       TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    imported_at    TEXT NOT NULL
);

-- ---------- Layer 2: STANDARDIZED (single canonical schema) ----------
CREATE TABLE IF NOT EXISTS meeting (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    source                 TEXT NOT NULL,
    source_id              TEXT NOT NULL,
    meeting_date           TEXT,
    municipality           TEXT,
    state                  TEXT,
    state_fips             TEXT,
    county_fips            TEXT,
    municipality_fips      TEXT,
    fips_code              TEXT,
    meeting_name           TEXT,
    agenda                 TEXT,
    minutes                TEXT,
    transcript             TEXT,
    video_url              TEXT,
    audio_url              TEXT,
    latitude               REAL,
    longitude              REAL,
    canonical_id           INTEGER,
    standardization_version TEXT NOT NULL,
    import_timestamp       TEXT NOT NULL,
    UNIQUE (source, source_id),
    FOREIGN KEY (canonical_id) REFERENCES canonical_meeting (canonical_id)
);

CREATE INDEX IF NOT EXISTS idx_meeting_canonical ON meeting (canonical_id);
-- Bucket key for dedup: identical FIPS + close dates go in the same bucket,
-- so pairwise comparisons stay bounded even on large datasets.
CREATE INDEX IF NOT EXISTS idx_meeting_bucket
    ON meeting (fips_code, meeting_date);

-- ---------- Layer 3: RESEARCH ----------
CREATE TABLE IF NOT EXISTS canonical_meeting (
    canonical_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    primary_meeting_id INTEGER NOT NULL,
    confidence        REAL NOT NULL,
    method            TEXT NOT NULL,
    decision          TEXT NOT NULL,
    evidence          TEXT,
    deduper_version   TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    FOREIGN KEY (primary_meeting_id) REFERENCES meeting (id)
);

CREATE TABLE IF NOT EXISTS meeting_analysis (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id       INTEGER NOT NULL,
    classifier_version TEXT,
    prompt_version   TEXT,
    model            TEXT,
    topic            TEXT,
    confidence       REAL,
    evidence         TEXT,
    review_status    TEXT DEFAULT 'unreviewed',
    reviewed_by      TEXT,
    review_timestamp TEXT,
    created_at       TEXT NOT NULL,
    FOREIGN KEY (meeting_id) REFERENCES meeting (id)
);
"""

# Maps a source name to its Layer 1 raw table. Extend this when adding sources.
RAW_TABLES = {
    "meetingbank": "raw_meetingbank",
    "localview": "raw_localview",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Thin wrapper around a SQLite connection with the pipeline schema."""

    def __init__(self, path: str | Path = "pipeline.db"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Bring pre-existing DBs up to the current schema.

        SQLite's ``CREATE TABLE IF NOT EXISTS`` won't add new columns to a
        table that already exists, so we ALTER-add anything missing. This
        keeps upgrades of pre-FIPS databases transparent — the derived
        standardized layer can then be rebuilt to populate the new columns.
        """
        existing = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(meeting)")
        }
        for column, ddl in (
            ("state_fips", "ALTER TABLE meeting ADD COLUMN state_fips TEXT"),
            ("county_fips", "ALTER TABLE meeting ADD COLUMN county_fips TEXT"),
            ("municipality_fips",
             "ALTER TABLE meeting ADD COLUMN municipality_fips TEXT"),
            ("fips_code", "ALTER TABLE meeting ADD COLUMN fips_code TEXT"),
        ):
            if column not in existing:
                self.conn.execute(ddl)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ---------- Layer 1: raw writes (idempotent) ----------

    def raw_table_for(self, source: str) -> str:
        try:
            return RAW_TABLES[source]
        except KeyError:
            raise ValueError(
                f"Unknown source {source!r}; register it in database.RAW_TABLES"
            ) from None

    def insert_raw(self, source: str, record: RawRecord) -> bool:
        """Insert one raw record. Returns True if inserted, False if it existed.

        Idempotent on ``source_id``: re-running an importer never duplicates
        or mutates existing raw rows.
        """
        table = self.raw_table_for(source)
        with self.transaction() as conn:
            cur = conn.execute(
                f"INSERT OR IGNORE INTO {table} "
                "(source_id, raw_json, dataset_version, imported_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    record.source_id,
                    record.payload_json(),
                    record.dataset_version,
                    _utcnow(),
                ),
            )
            return cur.rowcount > 0

    def iter_raw(self, source: str) -> Iterator[dict[str, Any]]:
        table = self.raw_table_for(source)
        cur = self.conn.execute(
            f"SELECT source_id, raw_json, dataset_version FROM {table}"
        )
        for row in cur:
            yield {
                "source_id": row["source_id"],
                "payload": json.loads(row["raw_json"]),
                "dataset_version": row["dataset_version"],
            }

    def count_raw(self, source: str) -> int:
        table = self.raw_table_for(source)
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # ---------- Layer 2: standardized writes ----------

    def clear_standardized(self, source: Optional[str] = None) -> None:
        """Drop standardized rows so they can be rebuilt from raw."""
        with self.transaction() as conn:
            if source:
                conn.execute("DELETE FROM meeting WHERE source = ?", (source,))
            else:
                conn.execute("DELETE FROM meeting")

    def upsert_meeting(self, meeting: Meeting) -> int:
        """Insert or replace a standardized meeting. Returns its row id."""
        row = meeting.as_row()
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO meeting (
                    source, source_id, meeting_date, municipality, state,
                    state_fips, county_fips, municipality_fips, fips_code,
                    meeting_name, agenda, minutes, transcript, video_url,
                    audio_url, latitude, longitude, standardization_version,
                    import_timestamp
                ) VALUES (
                    :source, :source_id, :meeting_date, :municipality, :state,
                    :state_fips, :county_fips, :municipality_fips, :fips_code,
                    :meeting_name, :agenda, :minutes, :transcript, :video_url,
                    :audio_url, :latitude, :longitude, :standardization_version,
                    :import_timestamp
                )
                ON CONFLICT (source, source_id) DO UPDATE SET
                    meeting_date = excluded.meeting_date,
                    municipality = excluded.municipality,
                    state = excluded.state,
                    state_fips = excluded.state_fips,
                    county_fips = excluded.county_fips,
                    municipality_fips = excluded.municipality_fips,
                    fips_code = excluded.fips_code,
                    meeting_name = excluded.meeting_name,
                    agenda = excluded.agenda,
                    minutes = excluded.minutes,
                    transcript = excluded.transcript,
                    video_url = excluded.video_url,
                    audio_url = excluded.audio_url,
                    latitude = excluded.latitude,
                    longitude = excluded.longitude,
                    standardization_version = excluded.standardization_version,
                    import_timestamp = excluded.import_timestamp
                """,
                {**row, "import_timestamp": _utcnow()},
            )
            if cur.lastrowid:
                return cur.lastrowid
            existing = conn.execute(
                "SELECT id FROM meeting WHERE source = ? AND source_id = ?",
                (meeting.source, meeting.source_id),
            ).fetchone()
            return existing["id"]

    def iter_meetings(
        self, only_unresolved: bool = False
    ) -> Iterator[sqlite3.Row]:
        sql = "SELECT * FROM meeting"
        if only_unresolved:
            sql += " WHERE canonical_id IS NULL"
        sql += " ORDER BY fips_code, municipality, meeting_date"
        yield from self.conn.execute(sql)

    def count_meetings(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM meeting").fetchone()[0]

    # ---------- Layer 3: dedup writes ----------

    def clear_dedup(self) -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE meeting SET canonical_id = NULL")
            conn.execute("DELETE FROM canonical_meeting")

    def record_canonical(
        self,
        meeting_ids: Iterable[int],
        primary_meeting_id: int,
        confidence: float,
        method: str,
        decision: str,
        evidence: dict[str, Any],
    ) -> int:
        """Create a canonical entity and link its member meetings to it.

        Additive: original meeting rows are never deleted, only tagged with a
        ``canonical_id``.
        """
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO canonical_meeting (
                    primary_meeting_id, confidence, method, decision,
                    evidence, deduper_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    primary_meeting_id,
                    confidence,
                    method,
                    decision,
                    json.dumps(evidence, ensure_ascii=False),
                    DEDUPER_VERSION,
                    _utcnow(),
                ),
            )
            canonical_id = cur.lastrowid
            conn.executemany(
                "UPDATE meeting SET canonical_id = ? WHERE id = ?",
                [(canonical_id, mid) for mid in meeting_ids],
            )
            return canonical_id
