"""FIPS (Federal Information Processing Standards) location code resolution.

Maps a meeting's ``(state, municipality)`` to a hierarchical composite code
whose *length* carries the level:

    SS         (2 digits) state-level meeting
    SS CCC     (5 digits) county-level meeting
    SS CCC MM  (7 digits) municipal meeting

    │  │   └── municipality within county (2 digits, omitted when unknown /
    │  │       when the meeting is not municipal)
    │  └────── county within state (3 digits, standard Census county FIPS,
    │          omitted for state-level meetings)
    └──────── state (2 digits, standard Census Bureau state FIPS)

Missing lower levels are represented by *absence* — we do not pad with ``"00"``
or ``"000"``. That means the code's length alone tells you the resolution
level, and a state-level record can never accidentally collide with a
county- or municipal-level record in the same state.

State FIPS come from the Census Bureau. County FIPS come from the Census
Bureau (within-state). There is no standard 2-digit "place-within-county"
code — Census "place" FIPS is 5 digits and orthogonal to county — so the
``municipality_fips`` layer is left ``None`` in the shipped ``PLACE_INDEX``
until a real 2-digit scheme is sourced. Downstream stages treat a
missing composite level as "unknown at that resolution" and fall back to
coarser matching or flag for human review.

Extend ``PLACE_INDEX`` (or wire in a Census gazetteer loader) to grow coverage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Full 50 states plus DC and Puerto Rico. Standard Census Bureau state FIPS.
STATE_FIPS: dict[str, str] = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "FL": "12", "GA": "13",
    "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19",
    "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29",
    "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34",
    "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45",
    "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50",
    "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56",
    "DC": "11", "PR": "72",
}

# Full state / territory names to their two-letter abbreviation, so records
# that carry ``state: "Washington"`` resolve the same as ``state: "WA"``.
STATE_NAME_TO_ABBREV: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
    "puerto rico": "PR",
}

# Curated place index: state_abbrev -> normalized city name -> county_fips.
# County FIPS are the standard 3-digit Census values. Municipality-level FIPS
# is intentionally *not* recorded here — we don't have a valid 2-digit
# scheme, so we leave it null for known cities rather than fabricate codes.
# Extend this dict (and add a companion muni-code source) to widen coverage.
PLACE_INDEX: dict[str, dict[str, str]] = {
    "WA": {
        "seattle": "033",     # King County
        "tacoma": "053",      # Pierce County
        "spokane": "063",     # Spokane County
        "bellevue": "033",    # King County
    },
    "OR": {
        "portland": "051",    # Multnomah County
        "salem": "047",       # Marion County
        "eugene": "039",      # Lane County
    },
    "CA": {
        "los angeles": "037",    # Los Angeles County
        "san francisco": "075",  # San Francisco County
        "san diego": "073",      # San Diego County
        "san jose": "085",       # Santa Clara County
        "oakland": "001",        # Alameda County
        "sacramento": "067",     # Sacramento County
    },
    "CO": {
        "denver": "031",      # Denver County
        "boulder": "013",     # Boulder County
        "aurora": "005",      # Arapahoe County
    },
    "TX": {
        "austin": "453",      # Travis County
        "houston": "201",     # Harris County
        "dallas": "113",      # Dallas County
        "san antonio": "029", # Bexar County
    },
    "NY": {
        "new york": "061",    # New York County (Manhattan)
        "buffalo": "029",     # Erie County
        "albany": "001",      # Albany County
    },
    "IL": {
        "chicago": "031",     # Cook County
    },
    "MA": {
        "boston": "025",      # Suffolk County
        "cambridge": "017",   # Middlesex County
    },
}


# A meeting_name / place_name field can carry decorators that are semantically
# irrelevant for lookup. Strip common ones so ``"City of Seattle"`` resolves
# the same as ``"Seattle"``.
_CITY_DECORATOR_RE = re.compile(
    r"^(city|town|village|borough|township|county)\s+of\s+", re.IGNORECASE,
)
_TRAILING_STATE_RE = re.compile(r",\s*[A-Za-z .]+$")


def _normalize_city(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    text = name.strip()
    # "Seattle, WA" -> "Seattle"
    text = _TRAILING_STATE_RE.sub("", text)
    # "City of Seattle" -> "Seattle"
    text = _CITY_DECORATOR_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text or None


def _normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    text = state.strip()
    if not text:
        return None
    # Two-letter code?
    if len(text) == 2 and text.upper() in STATE_FIPS:
        return text.upper()
    # Full name?
    abbrev = STATE_NAME_TO_ABBREV.get(text.lower())
    return abbrev


@dataclass(frozen=True)
class FipsResolution:
    """Result of a location -> FIPS lookup.

    Each layer is ``None`` when it couldn't be resolved. The composite
    ``code`` concatenates only the layers that *were* resolved, so its length
    tells you the level (2 = state, 5 = state+county, 7 = state+county+muni).
    """

    state_fips: Optional[str] = None
    county_fips: Optional[str] = None
    municipality_fips: Optional[str] = None
    code: Optional[str] = None


def _compose(
    state_fips: Optional[str],
    county_fips: Optional[str],
    municipality_fips: Optional[str],
) -> Optional[str]:
    """Concatenate whatever levels are known — no padding for missing ones."""
    if not state_fips:
        return None
    parts = [state_fips]
    if county_fips:
        parts.append(county_fips)
        if municipality_fips:
            parts.append(municipality_fips)
    return "".join(parts)


def _lookup_place(
    state_abbrev: Optional[str], city: Optional[str]
) -> Optional[tuple[str, str]]:
    """Return ``(state_abbrev, county_fips)`` when uniquely resolvable."""
    if not city:
        return None
    if state_abbrev:
        county_fips = PLACE_INDEX.get(state_abbrev, {}).get(city)
        if county_fips:
            return (state_abbrev, county_fips)
        return None
    # No state given — see if the city appears in exactly one state.
    matches = [
        (abbr, cty)
        for abbr, cities in PLACE_INDEX.items()
        for name, cty in cities.items()
        if name == city
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_fips(
    state: Optional[str], municipality: Optional[str]
) -> FipsResolution:
    """Resolve a ``(state, municipality)`` pair to a FIPS composite.

    Emits only the levels we could resolve:
    - State-level (state name/abbrev known, city not in the index): 2-digit
      ``SS``.
    - County-level (state + city with a county lookup, no valid muni code):
      5-digit ``SSCCC``.
    - Municipal (all three known — currently never emitted, since the shipped
      ``PLACE_INDEX`` doesn't carry municipality codes): 7-digit ``SSCCCMM``.

    Unknown levels are ``None`` on the returned object and *omitted* (not
    padded with zeros) in the composite ``code``.
    """
    state_abbrev = _normalize_state(state)
    city = _normalize_city(municipality)
    place = _lookup_place(state_abbrev, city)

    if place is not None:
        resolved_state, county_fips = place
        state_fips = STATE_FIPS[resolved_state]
        # municipality_fips is intentionally None until a 2-digit muni source
        # is wired in — see module docstring.
        return FipsResolution(
            state_fips=state_fips,
            county_fips=county_fips,
            municipality_fips=None,
            code=_compose(state_fips, county_fips, None),
        )

    # No county-level match. Fall back to state-only when we can identify one.
    state_fips = STATE_FIPS.get(state_abbrev) if state_abbrev else None
    if state_fips:
        return FipsResolution(
            state_fips=state_fips,
            code=_compose(state_fips, None, None),
        )

    return FipsResolution()
