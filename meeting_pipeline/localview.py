"""LocalView importer + standardizer.

LocalView (https://localview.net / the associated research dataset) provides
local government meeting videos with transcripts and place metadata. As with
MeetingBank, we accept either a JSON/JSONL file or in-memory records and store
them verbatim in the raw layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

from .importer import BaseImporter
from .models import Meeting, RawRecord
from .standardize import BaseStandardizer, clean_text, parse_date


class LocalViewImporter(BaseImporter):
    source = "localview"

    def __init__(
        self,
        json_path: Optional[str | Path] = None,
        records: Optional[list[dict[str, Any]]] = None,
        dataset_version: str = "localview-v1",
    ):
        super().__init__(dataset_version=dataset_version)
        self.json_path = Path(json_path) if json_path else None
        self._records = records

    def fetch(self) -> Iterator[RawRecord]:
        if self._records is not None:
            source = self._records
        elif self.json_path is not None:
            source = self._read(self.json_path)
        else:
            raise ValueError("LocalViewImporter needs either json_path or records")
        for obj in source:
            yield RawRecord(source_id=self._extract_id(obj), payload=obj)

    @staticmethod
    def _read(path: Path) -> Iterator[dict[str, Any]]:
        text = path.read_text(encoding="utf-8")
        # Support both a JSON array and JSONL.
        stripped = text.lstrip()
        if stripped.startswith("["):
            yield from json.loads(text)
        else:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def _extract_id(obj: dict[str, Any]) -> str:
        for key in ("id", "video_id", "vid_id", "meeting_id"):
            if obj.get(key):
                return str(obj[key])
        place = obj.get("place_name") or obj.get("place") or "unknown"
        date = obj.get("date") or obj.get("meeting_date") or "undated"
        return f"{place}:{date}"


class LocalViewStandardizer(BaseStandardizer):
    source = "localview"

    def standardize(self, source_id: str, payload: dict[str, Any]) -> Meeting:
        return Meeting(
            source=self.source,
            source_id=source_id,
            meeting_date=parse_date(
                payload.get("date") or payload.get("meeting_date")
            ),
            municipality=clean_text(
                payload.get("place_name") or payload.get("place")
            ),
            state=clean_text(payload.get("state") or payload.get("state_name")),
            meeting_name=clean_text(
                payload.get("title") or payload.get("caption")
            ),
            agenda=clean_text(payload.get("agenda")),
            minutes=clean_text(payload.get("minutes")),
            transcript=clean_text(
                payload.get("caption_text_pipe")
                or payload.get("transcript")
                or payload.get("text")
            ),
            video_url=clean_text(payload.get("video_url") or payload.get("url")),
            audio_url=clean_text(payload.get("audio_url")),
            latitude=_as_float(payload.get("lat") or payload.get("latitude")),
            longitude=_as_float(payload.get("lon") or payload.get("longitude")),
        )


def _as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
