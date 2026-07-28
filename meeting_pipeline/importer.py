"""Reusable importer framework (Layer 1).

Importers are deliberately *stupid*: they download/read a source, yield raw
records, and store them verbatim. No parsing beyond identifying a stable
``source_id``, no normalization, no dedup, no AI. This keeps them idempotent
and makes raw data reproducible.

To add a source:
    1. Subclass ``BaseImporter``, set ``source``, implement ``fetch()``.
    2. Register a raw table in ``database.RAW_TABLES``.
    3. Add a standardizer in ``standardizers`` (see standardize.py).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterator

from .database import Database
from .models import RawRecord


@dataclass
class ImportResult:
    source: str
    seen: int = 0
    inserted: int = 0
    skipped: int = 0  # already present (idempotent re-run)

    def __str__(self) -> str:
        return (
            f"[{self.source}] seen={self.seen} inserted={self.inserted} "
            f"skipped={self.skipped}"
        )


class BaseImporter(abc.ABC):
    """Base class for all source importers."""

    #: Short source name; must match a key in ``database.RAW_TABLES``.
    source: str = ""

    def __init__(self, dataset_version: str = "unknown"):
        if not self.source:
            raise ValueError(f"{type(self).__name__} must set a 'source' name")
        self.dataset_version = dataset_version

    @abc.abstractmethod
    def fetch(self) -> Iterator[RawRecord]:
        """Yield raw records from the source. Must set a stable source_id."""
        raise NotImplementedError

    def run(self, db: Database) -> ImportResult:
        """Fetch all records and store them idempotently in the raw layer."""
        result = ImportResult(source=self.source)
        for record in self.fetch():
            record.dataset_version = self.dataset_version
            result.seen += 1
            if db.insert_raw(self.source, record):
                result.inserted += 1
            else:
                result.skipped += 1
        return result
