"""End-to-end and unit tests for the meeting pipeline."""

import pytest

from meeting_pipeline import (
    Database,
    DedupConfig,
    Deduplicator,
    LocalViewImporter,
    LocalViewStandardizer,
    MeetingBankImporter,
    MeetingBankStandardizer,
)
from meeting_pipeline.embeddings import HashingEmbedding, cosine_similarity
from meeting_pipeline.standardize import parse_date, clean_text


@pytest.fixture
def db(tmp_path):
    with Database(tmp_path / "test.db") as database:
        yield database


# ---------- helpers ----------

def test_parse_date_formats():
    assert parse_date("2023-01-15") == "2023-01-15"
    assert parse_date("01/15/2023") == "2023-01-15"
    assert parse_date("January 15, 2023") == "2023-01-15"
    assert parse_date("2023-01-15T09:30:00") == "2023-01-15"
    assert parse_date("") is None
    assert parse_date(None) is None
    assert parse_date("not a date") is None


def test_clean_text_collapses_whitespace():
    assert clean_text("  hello   world \n ") == "hello world"
    assert clean_text("") is None
    assert clean_text(None) is None


def test_cosine_similarity_identity():
    v = HashingEmbedding().embed("city council budget hearing")
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-9)


# ---------- raw layer idempotency ----------

def test_import_is_idempotent(db):
    records = [{"id": "1", "city": "Seattle", "date": "2023-01-15"}]
    r1 = MeetingBankImporter(records=records).run(db)
    assert r1.inserted == 1 and r1.skipped == 0
    r2 = MeetingBankImporter(records=records).run(db)
    assert r2.inserted == 0 and r2.skipped == 1
    assert db.count_raw("meetingbank") == 1


# ---------- standardization ----------

def test_standardize_maps_to_canonical_schema(db):
    records = [{
        "id": "mb-1", "city": "Seattle", "state": "WA",
        "date": "01/15/2023", "title": "Council Meeting",
        "transcript": "The   meeting   was called to order.",
    }]
    MeetingBankImporter(records=records).run(db)
    n = MeetingBankStandardizer().run(db)
    assert n == 1
    meeting = next(db.iter_meetings())
    assert meeting["municipality"] == "Seattle"
    assert meeting["meeting_date"] == "2023-01-15"
    assert meeting["transcript"] == "The meeting was called to order."
    assert meeting["source"] == "meetingbank"


def test_standardize_is_rerunnable(db):
    records = [{"id": "mb-1", "city": "Seattle", "date": "2023-01-15"}]
    MeetingBankImporter(records=records).run(db)
    MeetingBankStandardizer().run(db)
    MeetingBankStandardizer().run(db)  # rerun should not duplicate
    assert db.count_meetings() == 1


# ---------- dedup: deterministic ----------

def test_dedup_matches_same_video_url(db):
    mb = [{"id": "mb-1", "city": "Seattle", "date": "2023-01-15",
           "title": "Council Meeting", "video_url": "http://v/1"}]
    lv = [{"id": "lv-1", "place_name": "Seattle", "date": "2023-01-15",
           "title": "Regular Council Session", "video_url": "http://v/1"}]
    MeetingBankImporter(records=mb).run(db)
    LocalViewImporter(records=lv).run(db)
    MeetingBankStandardizer().run(db)
    LocalViewStandardizer().run(db)

    summaries = Deduplicator().run(db)
    matched = [s for s in summaries if s["decision"] == "matched"]
    assert len(matched) == 1
    assert set(matched[0]["members"]) == {1, 2}
    assert matched[0]["confidence"] >= 0.99


def test_dedup_matches_similar_title_within_date_tolerance(db):
    mb = [{"id": "mb-1", "city": "Portland", "date": "2023-03-01",
           "title": "City Budget Hearing 2023"}]
    lv = [{"id": "lv-1", "place_name": "Portland", "date": "2023-03-02",
           "title": "City Budget Hearing 2023"}]
    MeetingBankImporter(records=mb).run(db)
    LocalViewImporter(records=lv).run(db)
    MeetingBankStandardizer().run(db)
    LocalViewStandardizer().run(db)

    summaries = Deduplicator(DedupConfig(date_tolerance_days=1)).run(db)
    matched = [s for s in summaries if s["decision"] == "matched"]
    assert len(matched) == 1
    assert set(matched[0]["members"]) == {1, 2}


def test_dedup_keeps_distinct_meetings_separate(db):
    mb = [{"id": "mb-1", "city": "Denver", "date": "2023-01-15",
           "title": "Zoning Committee"}]
    lv = [{"id": "lv-1", "place_name": "Denver", "date": "2023-08-20",
           "title": "Parks Commission"}]
    MeetingBankImporter(records=mb).run(db)
    LocalViewImporter(records=lv).run(db)
    MeetingBankStandardizer().run(db)
    LocalViewStandardizer().run(db)

    summaries = Deduplicator().run(db)
    # Two singletons, no matches.
    assert all(s["decision"] in ("unique", "needs_review") for s in summaries)
    matched = [s for s in summaries if s["decision"] == "matched"]
    assert matched == []


# ---------- dedup: LLM adjudication of borderline pairs ----------

def test_llm_adjudicates_borderline_pairs(db):
    # Same muni/date, dissimilar titles, similar-ish transcripts -> borderline.
    mb = [{"id": "mb-1", "city": "Austin", "date": "2023-05-01",
           "title": "Regular Meeting",
           "transcript": "budget discussion parks and recreation funding levels"}]
    lv = [{"id": "lv-1", "place_name": "Austin", "date": "2023-05-01",
           "title": "Video Recording 4471",
           "transcript": "budget discussion parks recreation funding proposal today"}]
    MeetingBankImporter(records=mb).run(db)
    LocalViewImporter(records=lv).run(db)
    MeetingBankStandardizer().run(db)
    LocalViewStandardizer().run(db)

    calls = {"n": 0}

    def adjudicator(a, b):
        calls["n"] += 1
        return True, 0.95

    config = DedupConfig(use_llm=True, embedding_auto_accept=0.999,
                         embedding_review_floor=0.5)
    deduper = Deduplicator(config=config, llm_adjudicator=adjudicator)
    summaries = deduper.run(db)
    assert calls["n"] >= 1
    matched = [s for s in summaries if s["decision"] == "matched"]
    assert any(s["method"] == "llm" for s in matched)
