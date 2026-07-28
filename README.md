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
        Standardizers (deterministic,         ─► Layer 2: STANDARDIZED
                       FIPS-tagged)              single `meeting` schema
                    │
                    ▼
        Deduplicator (3-stage cascade)        ─► Layer 3: RESEARCH
                    │                             canonical_meeting
                    ▼
        Analyzer (pluggable LLM/heuristic,     ─► Layer 3: RESEARCH
                  learns from prior reviews)     analysis_prompt / run / finding
                    │
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

## Data sources

The importers read local JSON/JSONL files. They do not fetch anything over the
network — that's a deliberate design choice so runs are reproducible and
offline-friendly. You download the source data once, then point the importer
at it.

### MeetingBank

MeetingBank (Hu et al., ACL 2023, [arxiv:2305.17529](https://arxiv.org/abs/2305.17529))
is a benchmark of ~1,366 city council meetings across six U.S. municipalities:
**Alameda**, **Boston**, **Denver**, **King County (WA)**, **Long Beach**, and
**Seattle**. Each row is a *segment* of a meeting with a human-written summary
and full transcript.

Canonical release:
[`huuuyeah/meetingbank` on Hugging Face](https://huggingface.co/datasets/huuuyeah/meetingbank)
(CC-BY-NC-SA 4.0). Audio and PDFs live in a companion release; this pipeline
only consumes the text.

**Fetch the JSONL** — one line per segment, using the `datasets` library:

```python
from datasets import load_dataset
ds = load_dataset("huuuyeah/meetingbank", split="train")
ds.to_json("meetingbank.jsonl", lines=True)
```

Each line ends up shaped like:

```json
{
  "id": 0,
  "uid": "SeattleCityCouncil_06132016_Res 31669",
  "summary": "A RESOLUTION encouraging ...",
  "transcript": "The report of the Civil Rights, Utilities, ..."
}
```

Note: the raw payload does **not** carry `city`, `state`, or `date` fields —
those are packed inside `uid` (`{PlacePrefix}_{MMDDYYYY}_{ItemId}`). The
`MeetingBankStandardizer` unpacks them for you, and `summary` maps onto the
canonical `minutes` field (it's a summary of the meeting minutes). Run:

```bash
pipeline import meetingbank --path meetingbank.jsonl
pipeline standardize meetingbank
```

### LocalView

LocalView (Barari & Simko, [*Scientific Data* 2023](https://www.nature.com/articles/s41597-023-02044-y),
data at [localview.net](https://localview.net/)) is ~140,000 U.S. local
government meeting videos sourced from YouTube, with auto-generated captions
serving as transcripts. Coverage: 1,012 places and 2,861 governments,
2006–2022. If you want breadth across small and mid-sized municipalities,
this is the one.

**Fetch the data** from the LocalView download page (registration free; the
data is published as CSV/RDS on the project site and on Harvard Dataverse).
Convert to JSONL. Any of these column names will be picked up by the
`LocalViewStandardizer`:

| Canonical field | Accepted source keys                                     |
| --------------- | -------------------------------------------------------- |
| `source_id`     | `id` / `video_id` / `vid_id` / `meeting_id`              |
| `meeting_date`  | `date` / `meeting_date` / `date_uploaded` / `upload_date` / `published_at` |
| `municipality`  | `place_name` / `place` / `locality` / `city`             |
| `state`         | `state` / `state_name` / `state_abbr` / `state_fips` (numeric 2-digit) |
| `meeting_name`  | `title` / `meeting_name` / `caption`                     |
| `transcript`    | `caption_text_pipe` / `transcript` / `captions` / `text` |
| `video_url`     | `video_url` / `youtube_url` / `url`                      |
| `latitude`      | `lat` / `latitude`                                       |
| `longitude`     | `lon` / `longitude`                                      |

Unknown keys are preserved verbatim in the raw layer, so a future
standardizer revision can pick them up without a re-import. Run:

```bash
pipeline import localview --path localview.jsonl
pipeline standardize localview
```

### Adding another source

Same three-step recipe as MeetingBank and LocalView:
1. Subclass `BaseImporter` (set `source`, implement `fetch()`).
2. Subclass `BaseStandardizer` (map the raw payload onto `Meeting`).
3. Register the raw table in `database.RAW_TABLES` and the classes in the
   CLI registries.

## Analysis: prompted findings that improve over time

The analyzer runs a natural-language question against every standardized
meeting and stores a per-meeting finding: municipality, meeting date, action
summary, supporting quotes, confidence, and a human review field
(`unreviewed | confirmed_match | incorrect_match`).

- **Prompts are named and stored** (`analysis_prompt` table), so the same
  question can be rerun as new meetings arrive or better models become
  available — the prompt keeps its identity across runs.
- **Every execution is a new `analysis_run`**, preserving history for
  side-by-side comparison across models and dataset versions.
- **Each rerun consumes prior reviews as few-shot exemplars.** Confirmed
  matches expand the analyzer's positive term set; rejected matches trim it.
  So false-negative and false-positive rates fall as reviewers work through
  the backlog.

The default `KeywordAnalyzer` is deterministic and zero-dependency. Swap in
an LLM backend by subclassing `AnalysisBackend` — the orchestrator stays the
same.

```bash
pipeline analyze --name sea-level-rise \
    --prompt "what actions have been taken to mitigate global sea level rise"
pipeline report --name sea-level-rise
pipeline review 42 --status confirmed_match --reviewer alice
pipeline analyze --name sea-level-rise    # rerun; loads alice's reviews
```

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
pytest -q     # 19 passing
```
