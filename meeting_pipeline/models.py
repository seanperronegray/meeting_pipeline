"""Canonical data models shared across pipeline layers.

The ``Meeting`` dataclass is the *standardized* representation. Every
importer produces raw, source-shaped rows; every standardizer maps those
rows into this single schema so that downstream stages (dedup, classification,
export) never need to know which source a record came from.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .fips import resolve_fips

# Bump these when the corresponding stage's logic changes. Every derived row
# records the version that produced it, so researchers can filter by method.
# 1.1.0: added FIPS location resolution (state_fips / county_fips /
# municipality_fips / composite fips_code) so downstream buckets and matching
# operate on structured location, not free-text municipality names.
# 1.2.0: composite fips_code no longer pads missing lower levels — its length
# now encodes the resolution level (2 = state, 5 = county, 7 = municipal).
STANDARDIZER_VERSION = "1.2.0"
DEDUPER_VERSION = "1.2.0"


@dataclass
class Meeting:
    """A standardized meeting record (Layer 2).

    Contains no source-specific fields. ``source`` + ``source_id`` together
    uniquely identify the originating raw row, which preserves provenance
    without leaking source schema into the canonical layer.

    Location is captured twice on purpose: free-text ``municipality`` / ``state``
    for display and provenance, and structured FIPS fields
    (``state_fips`` / ``county_fips`` / ``municipality_fips`` / composite
    ``fips_code``) for reliable bucketing and matching in the dedup pass.
    """

    source: str
    source_id: str
    meeting_date: Optional[str] = None  # ISO 8601 date (YYYY-MM-DD)
    municipality: Optional[str] = None
    state: Optional[str] = None
    state_fips: Optional[str] = None            # 2-digit Census state FIPS
    county_fips: Optional[str] = None           # 3-digit county FIPS (None = state-level or unknown)
    municipality_fips: Optional[str] = None     # 2-digit muni code (None = county/state-level or unknown)
    fips_code: Optional[str] = None             # composite: SS (2) | SSCCC (5) | SSCCCMM (7)
    meeting_name: Optional[str] = None
    agenda: Optional[str] = None
    minutes: Optional[str] = None
    transcript: Optional[str] = None
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    standardization_version: str = STANDARDIZER_VERSION

    def __post_init__(self) -> None:
        # Only resolve when the caller didn't already supply FIPS values —
        # e.g. when rehydrating a Meeting from a stored row we don't want to
        # overwrite persisted codes with a fresh lookup that may differ if
        # the PLACE_INDEX has changed since import.
        if self.fips_code is None and (self.state or self.municipality):
            resolution = resolve_fips(self.state, self.municipality)
            self.state_fips = self.state_fips or resolution.state_fips
            self.county_fips = self.county_fips or resolution.county_fips
            self.municipality_fips = (
                self.municipality_fips or resolution.municipality_fips
            )
            self.fips_code = resolution.code

    def as_row(self) -> dict[str, Any]:
        """Return a dict suitable for insertion into the ``meeting`` table."""
        return asdict(self)


@dataclass
class RawRecord:
    """A raw imported record (Layer 1). ``payload`` is stored verbatim as JSON."""

    source_id: str
    payload: dict[str, Any]
    dataset_version: str = "unknown"

    def payload_json(self) -> str:
        return json.dumps(self.payload, ensure_ascii=False, sort_keys=True)


@dataclass
class DedupDecision:
    """A single deduplication decision linking meetings to a canonical entity."""

    meeting_ids: list[int]
    primary_meeting_id: int
    confidence: float
    method: str  # deterministic | embedding | llm
    decision: str  # matched | needs_review
    evidence: dict[str, Any] = field(default_factory=dict)
