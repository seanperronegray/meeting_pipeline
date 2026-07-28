"""LocalView importer + standardizer.

LocalView (https://localview.net; paper: Barari & Simko, *Scientific Data*
2023, doi:10.1038/s41597-023-02044-y) is a corpus of ~140k U.S. local
government meeting videos, sourced from YouTube, with generated captions
serving as transcripts. Records are typically flat rows with a video id,
place / state identifiers, an upload date, a title, and a caption string.
Field naming has shifted a bit across releases, so this importer accepts
several common aliases::

    id / video_id / vid_id / meeting_id
    place_name / place
    state / state_name           (name or two-letter abbrev)
    state_fips                    (raw 2-digit numeric FIPS code)
    date / meeting_date / date_uploaded / upload_date
    title / caption
    caption_text_pipe / transcript / text
    video_url / url / youtube_url
    lat / latitude, lon / longitude

Whatever isn't recognised is preserved verbatim in the raw layer, so a future
standardizer version can pull it out without a re-import. As with MeetingBank
we accept either a JSON/JSONL file or an in-memory list.
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
        # LocalView releases have used various field names for the same
        # data — try each alias in order and fall through to ``None``.
        state = (
            payload.get("state")
            or payload.get("state_name")
            or payload.get("state_abbr")
        )
        if not state and payload.get("state_fips") is not None:
            # Numeric FIPS as string / int. ``_normalize_state`` inside the
            # resolver understands both forms.
            state = str(payload["state_fips"]).zfill(2)

        return Meeting(
            source=self.source,
            source_id=source_id,
            meeting_date=parse_date(
                payload.get("date")
                or payload.get("meeting_date")
                or payload.get("date_uploaded")
                or payload.get("upload_date")
                or payload.get("published_at")
            ),
            municipality=clean_text(
                payload.get("place_name")
                or payload.get("place")
                or payload.get("locality")
                or payload.get("city")
            ),
            state=clean_text(state),
            meeting_name=clean_text(
                payload.get("title")
                or payload.get("meeting_name")
                or payload.get("caption")
            ),
            agenda=clean_text(payload.get("agenda")),
            minutes=clean_text(payload.get("minutes")),
            transcript=clean_text(
                payload.get("caption_text_pipe")
                or payload.get("transcript")
                or payload.get("captions")
                or payload.get("text")
            ),
            video_url=clean_text(
                payload.get("video_url")
                or payload.get("youtube_url")
                or payload.get("url")
            ),
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
