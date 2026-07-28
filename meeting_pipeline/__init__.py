"""meeting_pipeline — a reproducible ETL pipeline for municipal meeting data.

Three immutable-raw / derived layers:

    Layer 1  RAW           immutable, one table per source (raw_*).
    Layer 2  STANDARDIZED  single ``meeting`` schema, deterministic, FIPS-tagged.
    Layer 3  RESEARCH      canonical_meeting + analysis_prompt/run/finding.

Public API re-exports the pieces most callers need.
"""

from .analysis import (
    AnalysisBackend,
    AnalysisFinding,
    Exemplar,
    KeywordAnalyzer,
    MeetingAnalyzer,
    RunSummary,
)
from .database import (
    REVIEW_CONFIRMED,
    REVIEW_INCORRECT,
    REVIEW_STATES,
    REVIEW_UNREVIEWED,
    Database,
)
from .dedup import DedupConfig, Deduplicator
from .embeddings import HashingEmbedding, cosine_similarity
from .fips import FipsResolution, resolve_fips
from .importer import BaseImporter, ImportResult
from .localview import LocalViewImporter, LocalViewStandardizer
from .meetingbank import MeetingBankImporter, MeetingBankStandardizer
from .models import DedupDecision, Meeting, RawRecord
from .standardize import BaseStandardizer

__version__ = "0.1.0"

__all__ = [
    "AnalysisBackend",
    "AnalysisFinding",
    "BaseImporter",
    "BaseStandardizer",
    "cosine_similarity",
    "Database",
    "DedupConfig",
    "DedupDecision",
    "Deduplicator",
    "Exemplar",
    "FipsResolution",
    "HashingEmbedding",
    "ImportResult",
    "KeywordAnalyzer",
    "LocalViewImporter",
    "LocalViewStandardizer",
    "Meeting",
    "MeetingAnalyzer",
    "MeetingBankImporter",
    "MeetingBankStandardizer",
    "RawRecord",
    "resolve_fips",
    "REVIEW_CONFIRMED",
    "REVIEW_INCORRECT",
    "REVIEW_STATES",
    "REVIEW_UNREVIEWED",
    "RunSummary",
]
