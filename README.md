# meeting-research-pipeline

A reproducible ETL pipeline for municipal meeting research data (MeetingBank,
LocalView, and extensible to more sources).

**Design principle: raw data is immutable. Every transformation downstream of
it is deterministic, versioned, and rebuildable.** That gives researchers
provenance, reproducibility, and the ability to rerun any stage as methods
improve — without ever re-importing.

## Architecture — three layers, like a data lake

```
   Source systems (MeetingBank, LocalView, …)
                    │
                    ▼
        Importers (idempotent, "stupid")     ─► Layer 1: RAW
                    │                             one table per source,
                    │                             raw JSON, never edited
                    ▼
        Standardizers (deterministic)         ─► Layer 2: STANDARDIZED
                    │                             single `meeting` schema
                    ▼
        Deduplicator (3-stage cascade)        ─► Layer 3: RESEARCH
                    │                             canonical_meeting (+ analysis)
                    ▼
        Published research dataset
```

- **Importers** only download → extract a stable `source_id` → store raw. No
  parsing, no AI, no dedup. Re-running never duplicates or mutates raw rows.
- **Standardizers** are pure functions from a raw payload to the single
  `Meeting` schema. Drop the standardized tables and rebuild them any time.
- **Deduplication is additive**: meetings are tagged with a `canonical_id`;
  originals are never deleted, so every source record stays inspectable.

Every derived row records the version of the stage that produced it
(`standardization_version`, `deduper_version`), so you can filter by method.

## Deduplication: a cheap→expensive, explainable cascade

Not an LLM as the primary mechanism. Instead, three stages, each with a stored
confidence and method:

1. **Deterministic** — same municipality, meeting date within a configurable
   tolerance, then identical video URL or highly similar title. High
   confidence, no ML.
2. **Embedding** — cosine similarity of embeddings built from the first
   transcript paragraphs (or agenda / title as fallback). Catches duplicates
   the rules miss.
3. **LLM adjudication** — *only* for borderline pairs: embedding similarity is
   above a review floor but below auto-accept, and the deterministic rules
   didn't conclude. Everything below auto-accept is flagged `needs_review` for
   a human rather than merged silently.

The default embedding backend is a dependency-free deterministic hashing
embedding, so the whole pipeline runs and tests reproducibly with **zero
external dependencies or API keys**. Swap in `sentence-transformers` (or any
`EmbeddingBackend`) for production, and pass any `llm_adjudicator` callable.

## Install

```bash
pip install -e .            # core: standard library only
pip install -e ".[dev]"     # + pytest
pip install -e ".[embeddings]"  # + sentence-transformers (optional)
```

## Usage — every stage is independently rerunnable

```bash
pipeline import meetingbank --path meetings.jsonl
pipeline import localview   --path localview.json
pipeline standardize
pipeline dedup --date-tolerance 1
pipeline dedup --use-llm --rebuild   # rerun dedup only, no re-import
pipeline stats
```

Or as a library:

```python
from meeting_pipeline import (
    Database, MeetingBankImporter, MeetingBankStandardizer,
    Deduplicator, DedupConfig,
)

with Database("pipeline.db") as db:
    MeetingBankImporter(jsonl_path="meetings.jsonl").run(db)  # Layer 1
    MeetingBankStandardizer().run(db)                          # Layer 2
    summaries = Deduplicator(DedupConfig(date_tolerance_days=1)).run(db)  # L3
```

## Adding a new source

1. Subclass `BaseImporter` (set `source`, implement `fetch()`).
2. Subclass `BaseStandardizer` (map the raw payload onto `Meeting`).
3. Register the raw table in `database.RAW_TABLES` and the classes in the CLI
   registries.

## Layout

```
meeting_pipeline/
    __init__.py       public API
    models.py         Meeting / RawRecord dataclasses + stage versions
    database.py       SQLite schema for all three layers
    importer.py       BaseImporter framework
    meetingbank.py    MeetingBank importer + standardizer
    localview.py      LocalView importer + standardizer
    standardize.py    BaseStandardizer + date/text helpers
    embeddings.py     EmbeddingBackend + hashing default + cosine
    dedup.py          deterministic → embedding → LLM cascade
    cli.py            `pipeline` command
tests/
    test_pipeline.py  idempotency, standardization, dedup, LLM adjudication
```

## Tests

```bash
pytest -q     # 10 passing
```
