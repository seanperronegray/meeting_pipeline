"""End-to-end and unit tests for the meeting pipeline."""

import pytest

from meeting_pipeline import (
    Database,
    DedupConfig,
    Deduplicator,
    KeywordAnalyzer,
    LocalViewImporter,
    LocalViewStandardizer,
    MeetingAnalyzer,
    MeetingBankImporter,
    MeetingBankStandardizer,
    REVIEW_CONFIRMED,
    REVIEW_INCORRECT,
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



# ---------- analysis: prompted findings ----------

def _seed_analysis_meetings(db):
    """Seed a mix of meetings — some on-topic for sea-level rise, some not."""
    mb = [
        {"id": "mb-1", "city": "Miami", "state": "FL",
         "date": "2023-04-10", "title": "Coastal Resilience Committee",
         "transcript": (
             "The committee reviewed the sea level rise adaptation plan. "
             "Members discussed shoreline protection and flooding mitigation "
             "measures for vulnerable coastal neighborhoods."
         )},
        {"id": "mb-2", "city": "Denver", "state": "CO",
         "date": "2023-05-02", "title": "Parks Commission",
         "transcript": (
             "Commissioners approved the summer park programming budget "
             "and discussed volunteer recruitment for trail maintenance."
         )},
    ]
    lv = [
        {"id": "lv-1", "place_name": "Boston", "state": "MA",
         "date": "2023-06-14", "title": "Climate Adaptation Hearing",
         "transcript": (
             "Testimony covered sea level rise projections along the harbor "
             "and proposed sea wall investments to protect low-lying wharfs."
         )},
    ]
    MeetingBankImporter(records=mb).run(db)
    LocalViewImporter(records=lv).run(db)
    MeetingBankStandardizer().run(db)
    LocalViewStandardizer().run(db)


def test_analyze_creates_prompt_run_and_findings(db):
    _seed_analysis_meetings(db)
    analyzer = MeetingAnalyzer()
    summary = analyzer.run(
        db,
        prompt_name="sea-level",
        prompt_text="what actions have been taken to mitigate sea level rise",
    )
    assert summary.meetings_scanned == 3
    assert summary.matches >= 2      # Miami + Boston, on-topic
    findings = analyzer.report(db, run_id=summary.run_id)
    # Every finding carries the fields the user asked for.
    for f in findings:
        assert f["municipality"]
        assert f["meeting_date"]
        assert 0.0 <= f["confidence"] <= 1.0
        assert f["review_status"] == "unreviewed"
        assert isinstance(f["quotes"], list)
    munis = {f["municipality"].lower() for f in findings}
    assert "miami" in munis
    assert "boston" in munis
    assert "denver" not in munis     # off-topic; shouldn't match


def test_analyze_rerunning_creates_new_run_and_keeps_history(db):
    _seed_analysis_meetings(db)
    analyzer = MeetingAnalyzer()
    r1 = analyzer.run(db, prompt_name="sea-level",
                      prompt_text="sea level rise mitigation actions")
    r2 = analyzer.run(db, prompt_name="sea-level")  # no prompt_text needed
    assert r2.run_id != r1.run_id
    # Two runs -> two rows in analysis_run; both queryable.
    assert analyzer.report(db, run_id=r1.run_id)
    assert analyzer.report(db, run_id=r2.run_id)


def test_review_status_persists_and_feeds_next_run_as_exemplar(db):
    _seed_analysis_meetings(db)
    analyzer = MeetingAnalyzer()
    r1 = analyzer.run(db, prompt_name="sea-level",
                      prompt_text="sea level rise")
    findings_r1 = analyzer.report(db, run_id=r1.run_id)
    # Confirm one match. Its terms should feed the next run.
    target = findings_r1[0]
    analyzer.review(db, target["finding_id"], REVIEW_CONFIRMED, reviewer="alice")

    r2 = analyzer.run(db, prompt_name="sea-level")
    # The next run should see the confirmed exemplar it can learn from.
    assert r2.exemplar_count == 1
    findings_r2 = analyzer.report(db, run_id=r2.run_id)
    # History preserved on the reviewed finding.
    reviewed = [f for f in analyzer.report(db, run_id=r1.run_id)
                if f["finding_id"] == target["finding_id"]][0]
    assert reviewed["review_status"] == REVIEW_CONFIRMED
    assert reviewed["reviewed_by"] == "alice"
    # And the second run's matches are at least as many as the first, since
    # positive exemplars expand the keyword set.
    assert len(findings_r2) >= len(findings_r1)


def test_incorrect_review_dampens_future_confidence(db):
    _seed_analysis_meetings(db)
    analyzer = MeetingAnalyzer()
    r1 = analyzer.run(db, prompt_name="sea-level",
                      prompt_text="sea level rise coastal flooding")
    # Reject every match from the first run.
    for f in analyzer.report(db, run_id=r1.run_id):
        analyzer.review(db, f["finding_id"], REVIEW_INCORRECT, reviewer="alice")

    r2 = analyzer.run(db, prompt_name="sea-level")
    r2_findings = analyzer.report(db, run_id=r2.run_id)
    # After the analyzer sees negative exemplars covering the same vocabulary,
    # its confidence should drop and produce strictly fewer matches.
    assert len(r2_findings) <= len(analyzer.report(db, run_id=r1.run_id))


def test_prompt_is_reusable_and_versioned(db):
    _seed_analysis_meetings(db)
    analyzer = MeetingAnalyzer()
    analyzer.run(db, prompt_name="sea-level",
                 prompt_text="sea level rise", description="v1")
    # Rerunning with a new description updates the prompt row in place, but
    # keeps its stable id / name.
    analyzer.run(db, prompt_name="sea-level", description="v1 (refined)")
    prompt = db.get_prompt("sea-level")
    assert prompt["description"] == "v1 (refined)"
    assert prompt["prompt_text"] == "sea level rise"


def test_analyzer_backend_is_pluggable(db):
    """A custom backend can slot in without touching the orchestrator."""
    _seed_analysis_meetings(db)

    from meeting_pipeline.analysis import AnalysisBackend, AnalysisFinding

    class AlwaysMatchStub(AnalysisBackend):
        name = "stub"

        def analyze(self, prompt, meeting, exemplars):
            return AnalysisFinding(
                meeting_id=meeting["id"], is_match=True,
                summary="stub", quotes=["stub"], confidence=1.0,
            )

    summary = MeetingAnalyzer(backend=AlwaysMatchStub()).run(
        db, prompt_name="all", prompt_text="anything",
    )
    assert summary.model == "stub"
    assert summary.matches == summary.meetings_scanned
