"""AI-driven analysis of standardized meetings (Layer 3).

Given a natural-language prompt (e.g. *"what actions have been taken to
mitigate global sea level rise"*), the analyzer scans every meeting in the
standardized layer and emits a per-meeting finding with:

    - the municipality and meeting date,
    - a short summary of the action (or ``None`` if not a match),
    - supporting quotes from the transcript / agenda / title,
    - a numeric confidence score in [0, 1],
    - a human review field: ``unreviewed | confirmed_match | incorrect_match``.

Prompts are named and stored in ``analysis_prompt`` so the same query can be
rerun as new meetings arrive or better models become available. Each execution
creates an ``analysis_run`` row; findings hang off that run. **Every subsequent
run of a prompt automatically loads the human review history for that prompt
as few-shot exemplars**, so the analyzer sharpens with each review — confirmed
matches boost the terms it looks for, rejected matches dampen them.

The backend is pluggable (see ``AnalysisBackend``): swap in an OpenAI /
Anthropic / local LLM client without touching the orchestrator. The shipped
default ``KeywordAnalyzer`` uses keyword overlap and is fully deterministic,
so the pipeline runs and its tests are reproducible with zero external
dependencies. Nothing in this module makes a network call.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Sequence

from .database import (
    REVIEW_CONFIRMED,
    REVIEW_INCORRECT,
    REVIEW_STATES,
    Database,
)


# --------------------------------------------------------------------- types

@dataclass
class Exemplar:
    """A prior human-reviewed finding, used to steer a new analysis run."""

    meeting_id: int
    review_status: str  # REVIEW_CONFIRMED | REVIEW_INCORRECT
    summary: Optional[str]
    quotes: list[str] = field(default_factory=list)

    @property
    def is_positive(self) -> bool:
        return self.review_status == REVIEW_CONFIRMED

    @property
    def is_negative(self) -> bool:
        return self.review_status == REVIEW_INCORRECT


@dataclass
class AnalysisFinding:
    """One backend verdict for one meeting."""

    meeting_id: int
    is_match: bool
    summary: Optional[str]
    quotes: list[str]
    confidence: float  # in [0, 1]


@dataclass
class RunSummary:
    run_id: int
    prompt_id: int
    prompt_name: str
    model: str
    exemplar_count: int
    meetings_scanned: int
    matches: int


# ---------------------------------------------------------------- backends

class AnalysisBackend(abc.ABC):
    """Interface every analyzer backend implements.

    Concrete implementations may call an LLM, a local model, or a rule-based
    heuristic — the orchestrator only depends on this interface. The ``name``
    attribute is stored on each ``analysis_run`` row so runs across
    different models/versions can be compared.
    """

    name: str = ""

    @abc.abstractmethod
    def analyze(
        self,
        prompt: str,
        meeting: dict[str, Any],
        exemplars: Sequence[Exemplar],
    ) -> AnalysisFinding:
        raise NotImplementedError


# --- helpers shared by the reference backend ---

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Common English stopwords + question-shell tokens that appear in prompts like
# "what actions have been taken to mitigate ..." but carry no topical signal.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "by",
    "with", "is", "are", "was", "were", "be", "been", "being", "has", "have",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "must", "can", "what", "which", "who", "whom", "whose", "when",
    "where", "why", "how", "this", "that", "these", "those", "it", "its",
    "he", "she", "they", "we", "you", "us", "our", "your", "their", "them",
    "as", "if", "from", "not", "no", "so", "than", "then", "there", "here",
    "any", "all", "some", "one", "two", "three", "also", "just", "such",
    "action", "actions", "taken", "take", "taking", "took",
})


def _tokens(text: Optional[str]) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _keywords(text: Optional[str]) -> list[str]:
    """Content words: lowercase tokens ≥3 chars with stopwords removed."""
    return [t for t in _tokens(text) if len(t) >= 3 and t not in _STOPWORDS]


def _meeting_text(meeting: dict[str, Any]) -> str:
    """Concatenate the searchable text for a meeting, transcript first."""
    parts = [
        meeting.get("transcript"),
        meeting.get("agenda"),
        meeting.get("minutes"),
        meeting.get("meeting_name"),
    ]
    return " ".join(p for p in parts if p)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


class KeywordAnalyzer(AnalysisBackend):
    """Deterministic reference backend — no external dependencies.

    Scores a meeting by the fraction of the prompt's content keywords that
    appear in its text. Extracts the highest-density sentences as quotes and
    uses the top quote as the summary.

    Prior human reviews steer the score:
    - ``confirmed_match`` findings contribute terms that are added to the
      keyword set (positive expansion), so future runs pick up meetings that
      use synonymous vocabulary the raw prompt missed.
    - ``incorrect_match`` findings contribute terms that are subtracted from
      the keyword set (negative expansion), so future runs stop misfiring on
      the same red herrings.

    This is intentionally simple. Swap in an LLM backend for production; the
    orchestrator will pass the same exemplars into that backend's context
    window instead of into this term-set arithmetic.
    """

    name = "keyword-v1"

    def __init__(
        self,
        match_threshold: float = 0.30,
        max_quotes: int = 3,
        exemplar_term_cap: int = 25,
    ):
        self.match_threshold = match_threshold
        self.max_quotes = max_quotes
        self.exemplar_term_cap = exemplar_term_cap

    #: How much each positive-exemplar term (beyond the prompt) nudges
    #: confidence up when it appears in the meeting text. Kept small so
    #: exemplars refine — they don't override — the analyst's prompt.
    POS_BOOST_PER_HIT = 0.02
    #: How much each negative-exemplar term (beyond the prompt) subtracts
    #: from confidence. Larger than the positive boost, on the principle that
    #: a rejected match is more informative than a confirmed one.
    NEG_PENALTY_PER_HIT = 0.05

    def analyze(
        self,
        prompt: str,
        meeting: dict[str, Any],
        exemplars: Sequence[Exemplar],
    ) -> AnalysisFinding:
        text = _meeting_text(meeting)
        prompt_terms = set(_keywords(prompt))

        if not prompt_terms or not text:
            return AnalysisFinding(
                meeting_id=meeting["id"], is_match=False, summary=None,
                quotes=[], confidence=0.0,
            )

        pos_expansion, neg_expansion = self._exemplar_terms(exemplars)
        text_terms = set(_keywords(text))

        # Base confidence = fraction of the *prompt's* keywords present in the
        # meeting text. The prompt is ground truth; exemplars can't remove a
        # term from this set — they only shift the score after the fact.
        prompt_hits = prompt_terms & text_terms
        confidence = len(prompt_hits) / len(prompt_terms)

        # Positive exemplars: each *extra* term (i.e. one not already in the
        # prompt) that also appears in the text nudges confidence up. Capped
        # at 1.0 overall.
        pos_extra_hits = (pos_expansion - prompt_terms) & text_terms
        if pos_extra_hits:
            confidence = min(
                1.0, confidence + self.POS_BOOST_PER_HIT * len(pos_extra_hits)
            )

        # Negative exemplars: each hit dampens confidence. Prompt terms are
        # excluded — a rejected exemplar cannot rescind the analyst's query.
        neg_hits = (neg_expansion - prompt_terms) & text_terms
        if neg_hits:
            confidence = max(
                0.0, confidence - self.NEG_PENALTY_PER_HIT * len(neg_hits)
            )

        is_match = confidence >= self.match_threshold
        # Quotes are extracted with prompt + positive expansion so we pick up
        # meaningful sentences from meetings that use exemplar vocabulary.
        # Negative-exemplar terms (that aren't in the prompt) are excluded so
        # rejected patterns don't get echoed back in the report.
        quote_terms = (prompt_terms | pos_expansion) - (
            neg_expansion - prompt_terms
        )
        quotes = self._extract_quotes(text, quote_terms) if is_match else []
        summary = quotes[0] if quotes else None

        return AnalysisFinding(
            meeting_id=meeting["id"],
            is_match=is_match,
            summary=summary,
            quotes=quotes,
            confidence=round(confidence, 4),
        )

    # ---- internals ----

    def _exemplar_terms(
        self, exemplars: Sequence[Exemplar]
    ) -> tuple[set[str], set[str]]:
        """Build positive/negative term expansions from exemplar summaries."""
        pos: set[str] = set()
        neg: set[str] = set()
        for e in exemplars:
            terms = set(
                _keywords(" ".join([e.summary or "", *(e.quotes or [])]))
            )
            if e.is_positive:
                pos |= terms
            elif e.is_negative:
                neg |= terms
        # Cap size so a runaway exemplar pool doesn't drown the prompt terms.
        if len(pos) > self.exemplar_term_cap:
            pos = set(sorted(pos)[: self.exemplar_term_cap])
        if len(neg) > self.exemplar_term_cap:
            neg = set(sorted(neg)[: self.exemplar_term_cap])
        # Never let a term be both positive and negative — negative wins,
        # because "confirmed here, rejected there" is signal to avoid the term
        # unless the prompt itself demands it.
        pos -= neg
        return pos, neg

    def _extract_quotes(self, text: str, keywords: set[str]) -> list[str]:
        scored: list[tuple[int, str]] = []
        for sentence in _sentences(text):
            s_terms = set(_keywords(sentence))
            hits = len(keywords & s_terms)
            if hits:
                scored.append((hits, sentence))
        scored.sort(key=lambda pair: (-pair[0], len(pair[1])))
        return [s for _, s in scored[: self.max_quotes]]


# ------------------------------------------------------------- orchestrator

class MeetingAnalyzer:
    """Run a prompt against every meeting, storing findings for later review."""

    def __init__(
        self,
        backend: Optional[AnalysisBackend] = None,
    ):
        self.backend = backend or KeywordAnalyzer()

    def run(
        self,
        db: Database,
        *,
        prompt_name: str,
        prompt_text: Optional[str] = None,
        description: Optional[str] = None,
        model: Optional[str] = None,
    ) -> RunSummary:
        """Analyze all meetings for a named prompt.

        On first use pass ``prompt_text`` to create the prompt. On subsequent
        runs just pass ``prompt_name`` — the stored text is reused.
        """
        prompt = db.upsert_prompt(prompt_name, prompt_text, description)
        exemplars = [
            Exemplar(
                meeting_id=row["meeting_id"],
                review_status=row["review_status"],
                summary=row.get("summary"),
                quotes=row.get("quotes") or [],
            )
            for row in db.iter_reviewed_findings(prompt["id"])
        ]

        model_name = model or self.backend.name
        run_id = db.create_analysis_run(
            prompt["id"], model=model_name, exemplar_count=len(exemplars),
        )

        scanned = 0
        matches = 0
        for row in db.iter_meetings():
            meeting = dict(row)
            finding = self.backend.analyze(
                prompt["prompt_text"], meeting, exemplars,
            )
            scanned += 1
            # Only persist positive verdicts. Non-matches would dominate the
            # findings table; storing every miss for every prompt for every
            # run scales badly. Confidence for non-matches is still reflected
            # in analysis_run counts.
            if finding.is_match:
                db.save_finding(
                    run_id,
                    meeting_id=finding.meeting_id,
                    is_match=True,
                    summary=finding.summary,
                    quotes=finding.quotes,
                    confidence=finding.confidence,
                )
                matches += 1
        db.finalize_run(run_id, meetings_scanned=scanned, matches=matches)

        return RunSummary(
            run_id=run_id,
            prompt_id=prompt["id"],
            prompt_name=prompt["name"],
            model=model_name,
            exemplar_count=len(exemplars),
            meetings_scanned=scanned,
            matches=matches,
        )

    # ---- reporting / review ----

    def report(
        self,
        db: Database,
        *,
        prompt_name: Optional[str] = None,
        run_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return the finding report.

        Precedence: ``run_id`` if given, else the latest run for
        ``prompt_name``. Raises if neither identifies a run.
        """
        if run_id is None:
            if not prompt_name:
                raise ValueError("provide run_id or prompt_name")
            prompt = db.get_prompt(prompt_name)
            if prompt is None:
                raise ValueError(f"prompt {prompt_name!r} does not exist")
            run_id = db.latest_run_id(prompt["id"])
            if run_id is None:
                return []
        return list(db.iter_findings_report(run_id=run_id))

    def review(
        self,
        db: Database,
        finding_id: int,
        status: str,
        reviewer: Optional[str] = None,
    ) -> None:
        """Set the human review field on a finding.

        ``status`` must be one of ``unreviewed``, ``confirmed_match``, or
        ``incorrect_match``. These labels feed the next run's exemplars.
        """
        db.set_review_status(finding_id, status, reviewer)


# Convenience re-exports for callers that only want the constants.
REVIEW_LABELS = REVIEW_STATES
