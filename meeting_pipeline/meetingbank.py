"""MeetingBank importer + standardizer.

MeetingBank (https://meetingbank.github.io/) is a corpus of U.S. city council
meetings with transcripts and metadata. Records may arrive from a JSONL file
or the HuggingFace ``datasets`` library; both paths produce identical raw
records so the standardizer never needs to care which was used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

from .importer import BaseImporter
from .models import Meeting, RawRecord
from .standardize import BaseStandardizer, clean_text, parse_date


class MeetingBankImporter(BaseImporter):
    source = "meetingbank"

    def __init__(
        self,
        jsonl_path: Optional[str | Path] = None,
        records: Optional[list[dict[str, Any]]] = None,
        dataset_version: str = "meetingbank-v1",
    ):
        super().__init__(dataset_version=dataset_version)
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._records = records

    def fetch(self) -> Iterator[RawRecord]:
        if self._records is not None:
            source = self._records
        elif self.jsonl_path is not None:
            source = self._read_jsonl(self.jsonl_path)
        else:
            raise ValueError(
                "MeetingBankImporter needs either jsonl_path or records"
            )
        for obj in source:
            source_id = self._extract_id(obj)
            yield RawRecord(source_id=source_id, payload=obj)

    @staticmethod
    def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def _extract_id(obj: dict[str, Any]) -> str:
        for key in ("id", "uid", "meeting_id", "guid"):
            if obj.get(key):
                return str(obj[key])
        # Fall back to a stable composite key so re-imports stay idempotent.
        city = obj.get("city") or obj.get("municipality") or "unknown"
        date = obj.get("date") or obj.get("meeting_date") or "undated"
        return f"{city}:{date}"


class MeetingBankStandardizer(BaseStandardizer):
    source = "meetingbank"

    def standardize(self, source_id: str, payload: dict[str, Any]) -> Meeting:
        return Meeting(
            source=self.source,
            source_id=source_id,
            meeting_date=parse_date(
                payload.get("date") or payload.get("meeting_date")
            ),
            municipality=clean_text(
                payload.get("city") or payload.get("municipality")
            ),
            state=clean_text(payload.get("state")),
            meeting_name=clean_text(
                payload.get("title")
                or payload.get("meeting_name")
                or payload.get("name")
            ),
            agenda=clean_text(payload.get("agenda")),
            minutes=clean_text(payload.get("minutes")),
            transcript=clean_text(
                payload.get("transcript") or payload.get("text")
            ),
            video_url=clean_text(payload.get("video_url") or payload.get("video")),
            audio_url=clean_text(payload.get("audio_url") or payload.get("audio")),
        )
