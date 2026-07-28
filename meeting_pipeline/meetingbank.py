"""MeetingBank importer + standardizer.

MeetingBank (https://meetingbank.github.io/) is a corpus of U.S. city council
meetings with transcripts and human-written summaries. The canonical release
is hosted on Hugging Face as ``huuuyeah/meetingbank`` (paper: arXiv 2305.17529)
covering six municipalities: Alameda, Boston, Denver, King County, Long Beach,
and Seattle. Records look like::

    {
      "id":         0,
      "uid":        "SeattleCityCouncil_06132016_Res 31669",
      "summary":    "A RESOLUTION encouraging as a best practice ...",
      "transcript": "The report of the Civil Rights, Utilities, Economic ..."
    }

Note that the payload does **not** include explicit ``city``, ``state``, or
``date`` fields — those live encoded inside the ``uid``. The standardizer
below parses them out (see ``_parse_uid``). Records may arrive from a JSONL
file or an in-memory list; both paths produce identical raw records so the
standardizer never needs to care which was used.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
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
        # ``uid`` is preferred: it's the human-readable, globally stable id in
        # the HuggingFace release. The bare ``id`` field is a per-split index
        # (0..N-1) that would collide across splits and shift on re-shuffles.
        for key in ("uid", "meeting_id", "guid", "id"):
            if obj.get(key) is not None and obj.get(key) != "":
                return str(obj[key])
        # Fall back to a stable composite key so re-imports stay idempotent.
        city = obj.get("city") or obj.get("municipality") or "unknown"
        date = obj.get("date") or obj.get("meeting_date") or "undated"
        return f"{city}:{date}"


class MeetingBankStandardizer(BaseStandardizer):
    source = "meetingbank"

    # UID pattern:  {AlphaPlacePrefix}_{MMDDYYYY}[_{ItemId}]
    _UID_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*)_(\d{8})(?:_(.+))?$")

    # The six known MeetingBank municipalities. The prefix portion of the uid
    # is not free-text — it's one of these bracket-cased identifiers. Keeping
    # them explicit means we never mistranscribe (e.g. "LongBeachCC" -> "Long
    # Beach", not "Longbeach Cc").
    _PLACE_ALIASES: dict[str, tuple[str, str]] = {
        "seattlecitycouncil": ("Seattle", "WA"),
        "seattlecc": ("Seattle", "WA"),
        "seattle": ("Seattle", "WA"),
        "longbeachcc": ("Long Beach", "CA"),
        "longbeachcitycouncil": ("Long Beach", "CA"),
        "denvercitycouncil": ("Denver", "CO"),
        "denvercc": ("Denver", "CO"),
        "denver": ("Denver", "CO"),
        "bostoncitycouncil": ("Boston", "MA"),
        "bostoncc": ("Boston", "MA"),
        "boston": ("Boston", "MA"),
        "alamedacitycouncil": ("Alameda", "CA"),
        "alamedacc": ("Alameda", "CA"),
        "alameda": ("Alameda", "CA"),
        "kingcountycouncil": ("King County", "WA"),
        "kingcounty": ("King County", "WA"),
    }

    def standardize(self, source_id: str, payload: dict[str, Any]) -> Meeting:
        # Real MeetingBank rows carry city/state/date only inside the uid.
        # Parse it once, then let explicit fields (rare, but possible in
        # derived/repacked copies of the dataset) override.
        uid = payload.get("uid") or source_id or ""
        muni, state, uid_date, item_id = self._parse_uid(uid)

        return Meeting(
            source=self.source,
            source_id=source_id,
            meeting_date=(
                parse_date(payload.get("date") or payload.get("meeting_date"))
                or uid_date
            ),
            municipality=(
                clean_text(payload.get("city") or payload.get("municipality"))
                or muni
            ),
            state=clean_text(payload.get("state")) or state,
            meeting_name=clean_text(
                payload.get("title")
                or payload.get("meeting_name")
                or payload.get("name")
                or item_id
            ),
            agenda=clean_text(payload.get("agenda")),
            # ``summary`` in the HuggingFace release is a human-written
            # summary distilled from the meeting minutes — it's the closest
            # match to the canonical ``minutes`` field. Explicit ``minutes``
            # (in repackaged copies) still wins.
            minutes=clean_text(payload.get("minutes") or payload.get("summary")),
            transcript=clean_text(
                payload.get("transcript") or payload.get("text")
            ),
            video_url=clean_text(payload.get("video_url") or payload.get("video")),
            audio_url=clean_text(payload.get("audio_url") or payload.get("audio")),
        )

    @classmethod
    def _parse_uid(
        cls, uid: str
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Split a MeetingBank uid into ``(municipality, state, date, item_id)``.

        Returns ``None`` for any component we couldn't confidently derive:
        - an unknown place prefix leaves municipality/state blank (the raw
          uid still identifies the record, and the FIPS resolver will just
          skip location matching for it),
        - an invalid or non-MMDDYYYY date leaves ``meeting_date`` blank.
        """
        if not uid:
            return (None, None, None, None)
        m = cls._UID_RE.match(uid)
        if not m:
            return (None, None, None, None)
        prefix, datestr, item = m.groups()

        # MMDDYYYY -> ISO. Validate via strptime so a bogus 13/32/etc.
        # collapses to None rather than pretending to be a real date.
        mm, dd, yyyy = datestr[:2], datestr[2:4], datestr[4:]
        try:
            datetime.strptime(f"{yyyy}-{mm}-{dd}", "%Y-%m-%d")
            date_iso: Optional[str] = f"{yyyy}-{mm}-{dd}"
        except ValueError:
            date_iso = None

        muni, state = cls._PLACE_ALIASES.get(prefix.lower(), (None, None))
        item_id = item.strip() if item else None
        return (muni, state, date_iso, item_id)
