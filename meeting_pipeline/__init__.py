"""meeting_pipeline — a reproducible ETL pipeline for municipal meeting data.

Three immutable-raw / derived layers:

    Layer 1  RAW           immutable, one table per source (raw_*).
    Layer 2  STANDARDIZED  single ``meeting`` schema, deterministic.
    Layer 3  RESEARCH      canonical_meeting + meeting_analysis.

Public API re-exports the pieces most callers need.
"""

from .database import Database
from .dedup import DedupConfig, Deduplicator
from .embeddings import HashingEmbedding, cosine_similarity
from .importer import BaseImporter, ImportResult
from .localview import LocalViewImporter, LocalViewStandardizer
from .meetingbank import MeetingBankImporter, MeetingBankStandardizer
from .models import DedupDecision, Meeting, RawRecord
from .standardize import BaseStandardizer

__version__ = "0.1.0"

__all__ = [
    "Database",
    "DedupConfig",
    "Deduplicator",
    "HashingEmbedding",
    "cosine_similarity",
    "BaseImporter",
    "ImportResult",
    "LocalViewImporter",
    "LocalViewStandardizer",
    "MeetingBankImporter",
    "MeetingBankStandardizer",
    "DedupDecision",
    "Meeting",
    "RawRecord",
    "BaseStandardizer",
]
