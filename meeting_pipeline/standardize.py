"""Standardization layer (Layer 1 -> Layer 2).

Standardizers are deterministic, pure functions of a raw payload. Each source
has one standardizer that maps its raw shape onto the single ``Meeting``
schema. Because they read from the immutable raw layer, standardized tables
can be dropped and rebuilt at any time without re-importing.

Standardizers only need to set free-text ``state`` and ``municipality`` — the
``Meeting`` dataclass then resolves those to a FIPS composite in its
``__post_init__``. That composite is what downstream stages (in particular the
dedup bucketing / location gate) key on, so **standardization must run before
dedup**: without a populated ``fips_code`` column, dedup would fall back to
free-text municipality matching and lose the state disambiguation.
"""

from __future__ import annotations

import abc
import re
from datetime import datetime
from typing import Any, Optional

from .database import Database
from .models import Meeting


def clean_text(value: Any) -> Optional[str]:
    """Collapse whitespace and normalize empty values to None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return re.sub(r"\s+", " ", text)


# Date formats seen across sources, tried in order.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)


def parse_date(value: Any) -> Optional[str]:
    """Parse a variety of date formats into an ISO 8601 date (YYYY-MM-DD)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Assume a Unix timestamp.
        try:
            return datetime.utcfromtimestamp(float(value)).strftime("%Y-%m-%d")
        except (ValueError, OSError, OverflowError):
            return None
    text = str(value).strip()
    if not text:
        return None
    # Fast path: already an ISO date/datetime.
    iso_match = re.match(r"(\d{4}-\d{2}-\d{2})", text)
    if iso_match:
        return iso_match.group(1)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


class BaseStandardizer(abc.ABC):
    """Maps raw records for one source into standardized ``Meeting`` rows."""

    source: str = ""

    def __init__(self) -> None:
        if not self.source:
            raise ValueError(f"{type(self).__name__} must set a 'source' name")

    @abc.abstractmethod
    def standardize(self, source_id: str, payload: dict[str, Any]) -> Meeting:
        raise NotImplementedError

    def run(self, db: Database) -> int:
        """Rebuild standardized rows for this source from the raw layer."""
        db.clear_standardized(self.source)
        count = 0
        for raw in db.iter_raw(self.source):
            meeting = self.standardize(raw["source_id"], raw["payload"])
            db.upsert_meeting(meeting)
            count += 1
        return count
