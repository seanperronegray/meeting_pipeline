"""Entity resolution / deduplication (Layer 2 -> Layer 3).

Prerequisite: the standardization pass must have run first, populating
``fips_code`` on each ``meeting`` row. Dedup uses that structured location to
bucket candidates instead of free-text municipality names.

Three-stage cascade, cheapest and most explainable first. **Two hard gates
apply to every pair before any content is compared**:

    Gate 1 (location): both meetings must resolve to the same ``fips_code``
        (or, if neither resolved, the same free-text municipality). Different
        locations are never merged.
    Gate 2 (date): the two meeting dates must fall within the configured
        tolerance. Title similarity is only inspected *after* the date gate
        passes — a meeting held on a different day is not the same meeting
        even if it shares a title.

Only pairs that clear both gates enter the cascade:

    1. Deterministic pass  — identical video URL or highly similar title.
    2. Embedding pass      — cosine similarity of transcript/agenda embeddings
       identifies likely duplicates the rules missed.
    3. LLM adjudication    — only for borderline pairs: embedding similarity is
       above a review threshold but below the auto-accept threshold, and the
       deterministic rules did not conclusively match.

Every decision stores a confidence score and method. Low-confidence matches
are flagged ``needs_review`` for a human rather than merged silently.
Deduplication is additive: meetings are tagged with a ``canonical_id``, never
deleted.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from difflib import SequenceMatcher
from typing import Any, Callable, Optional, Sequence

from .database import Database
from .embeddings import (
    EmbeddingBackend,
    HashingEmbedding,
    cosine_similarity,
    embedding_text,
)


@dataclass
class DedupConfig:
    date_tolerance_days: int = 1
    title_similarity_threshold: float = 0.85
    embedding_auto_accept: float = 0.92
    embedding_review_floor: float = 0.75
    #: When True, borderline pairs are sent to ``llm_adjudicator``.
    use_llm: bool = False


# An adjudicator takes two meeting rows and returns (is_duplicate, confidence).
LLMAdjudicator = Callable[[dict[str, Any], dict[str, Any]], tuple[bool, float]]


def _normalize_title(title: Optional[str]) -> str:
    if not title:
        return ""
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def title_similarity(a: Optional[str], b: Optional[str]) -> float:
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _parse_iso(d: Optional[str]) -> Optional[date]:
    if not d:
        return None
    try:
        return date.fromisoformat(d[:10])
    except ValueError:
        return None


def _dates_within(a: Optional[str], b: Optional[str], tol_days: int) -> bool:
    da, db = _parse_iso(a), _parse_iso(b)
    if da is None or db is None:
        return da is None and db is None  # both undated -> allowed in bucket
    return abs((da - db).days) <= tol_days


class Deduplicator:
    def __init__(
        self,
        config: Optional[DedupConfig] = None,
        embedder: Optional[EmbeddingBackend] = None,
        llm_adjudicator: Optional[LLMAdjudicator] = None,
    ):
        self.config = config or DedupConfig()
        self.embedder = embedder or HashingEmbedding()
        self.llm_adjudicator = llm_adjudicator

    # ---------- bucketing ----------

    def _bucket_key(self, row: dict[str, Any]) -> tuple:
        """Group candidates by structured location.

        Preferred: the standardized FIPS composite (``SS CCC MM``). Rows that
        couldn't be resolved to a FIPS code fall back to the free-text
        municipality — this keeps them clustered with their peers instead of
        silently drifting into the ``("fips", <known code>)`` buckets, which
        would incorrectly compare unresolved rows against resolved ones.
        """
        fips = (row.get("fips_code") or "").strip()
        if fips:
            return ("fips", fips)
        muni = (row.get("municipality") or "").lower().strip()
        state = (row.get("state") or "").lower().strip()
        return ("muni", state, muni)

    @staticmethod
    def _same_location(a: dict[str, Any], b: dict[str, Any]) -> bool:
        """Location gate — a match requires the *same* place.

        If both rows have a resolved FIPS code we compare on that. Otherwise
        we fall back to matching the free-text state + municipality tuple.
        Two rows in different fallback tuples are never considered the same
        location, even if their municipality strings happen to coincide
        (e.g. Springfield, MO vs Springfield, IL).
        """
        fa = (a.get("fips_code") or "").strip()
        fb = (b.get("fips_code") or "").strip()
        if fa and fb:
            return fa == fb
        ma = (a.get("municipality") or "").lower().strip()
        mb = (b.get("municipality") or "").lower().strip()
        sa = (a.get("state") or "").lower().strip()
        sb = (b.get("state") or "").lower().strip()
        return bool(ma) and ma == mb and sa == sb

    def _load_unresolved(self, db: Database) -> list[dict[str, Any]]:
        return [dict(r) for r in db.iter_meetings(only_unresolved=True)]

    # ---------- the cascade for a single candidate pair ----------

    def _deterministic_match(
        self, a: dict[str, Any], b: dict[str, Any]
    ) -> Optional[tuple[float, dict[str, Any]]]:
        # Gate 1: location must match. Bucketing normally guarantees this, but
        # we re-check so this method is also safe to call on ad-hoc pairs.
        if not self._same_location(a, b):
            return None
        # Gate 2: dates must match (within tolerance) before we let title
        # similarity vote. Two meetings on different days aren't the same
        # meeting, no matter how similar their titles read.
        if not _dates_within(
            a.get("meeting_date"), b.get("meeting_date"),
            self.config.date_tolerance_days,
        ):
            return None
        va, vb = a.get("video_url"), b.get("video_url")
        if va and vb and va == vb:
            return 0.99, {"rule": "same_video_url", "video_url": va}
        sim = title_similarity(a.get("meeting_name"), b.get("meeting_name"))
        if sim >= self.config.title_similarity_threshold:
            return sim, {"rule": "title_similarity", "title_score": round(sim, 3)}
        return None

    def _embedding_match(
        self, a: dict[str, Any], b: dict[str, Any], emb: dict[int, list[float]]
    ) -> tuple[float, dict[str, Any]]:
        score = cosine_similarity(emb[a["id"]], emb[b["id"]])
        return score, {"rule": "embedding", "cosine": round(score, 3)}

    # ---------- main entry point ----------

    def run(self, db: Database) -> list[dict[str, Any]]:
        """Resolve all unresolved meetings. Returns per-canonical summaries."""
        rows = self._load_unresolved(db)
        buckets: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[self._bucket_key(row)].append(row)

        # Precompute embeddings once per meeting.
        emb: dict[int, list[float]] = {}
        for row in rows:
            emb[row["id"]] = self.embedder.embed(
                embedding_text(
                    row.get("transcript"),
                    row.get("agenda"),
                    row.get("meeting_name"),
                )
            )

        summaries: list[dict[str, Any]] = []
        for bucket_rows in buckets.values():
            summaries.extend(self._resolve_bucket(db, bucket_rows, emb))
        return summaries

    def _resolve_bucket(
        self,
        db: Database,
        rows: list[dict[str, Any]],
        emb: dict[int, list[float]],
    ) -> list[dict[str, Any]]:
        # Union-find over the bucket; edges come from the cascade.
        parent = {r["id"]: r["id"] for r in rows}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        by_id = {r["id"]: r for r in rows}
        # (representative_pair) -> (confidence, method, evidence)
        edge_meta: dict[frozenset, tuple[float, str, dict[str, Any]]] = {}

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                meta = self._classify_pair(a, b, emb)
                if meta is None:
                    continue
                confidence, method, evidence, decision = meta
                # Any candidate edge (matched OR needs_review) groups the
                # meetings; the component's decision is derived from its edges
                # afterward. This keeps borderline pairs together for review
                # instead of scattering them into singletons.
                union(a["id"], b["id"])
                key = frozenset((a["id"], b["id"]))
                edge_meta[key] = (confidence, method, evidence, decision)

        # Group by union-find component and persist.
        components: dict[int, list[int]] = defaultdict(list)
        for mid in parent:
            components[find(mid)].append(mid)

        summaries: list[dict[str, Any]] = []
        for member_ids in components.values():
            # Edges internal to this component.
            edges = [
                (k, v)
                for k, v in edge_meta.items()
                if k <= set(member_ids) and len(k) == 2
            ]
            if len(member_ids) == 1 and not edges:
                # Singleton with no candidate edge -> its own canonical entity
                # so every meeting is resolvable, but flagged as unique.
                mid = member_ids[0]
                cid = db.record_canonical(
                    member_ids, mid, 1.0, "deterministic", "unique", {}
                )
                summaries.append(
                    {"canonical_id": cid, "members": member_ids,
                     "decision": "unique", "method": "deterministic"}
                )
                continue

            # A component needs human review if any of its edges is borderline.
            needs_review = any(e[1][3] == "needs_review" for e in edges)
            decision = "needs_review" if needs_review else "matched"
            confidence = min((e[1][0] for e in edges), default=0.0)
            methods = {e[1][1] for e in edges}
            method = "mixed" if len(methods) > 1 else next(iter(methods))
            primary = self._pick_primary(member_ids, by_id)
            evidence = {"edges": [
                {"pair": sorted(list(e[0])), **e[1][2],
                 "confidence": e[1][0], "method": e[1][1], "decision": e[1][3]}
                for e in edges
            ]}
            cid = db.record_canonical(
                member_ids, primary, confidence, method, decision, evidence
            )
            summaries.append({
                "canonical_id": cid, "members": member_ids,
                "primary": primary, "decision": decision,
                "method": method, "confidence": round(confidence, 3),
            })
        return summaries

    def _classify_pair(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
        emb: dict[int, list[float]],
    ) -> Optional[tuple[float, str, dict[str, Any], str]]:
        """Run the cascade on one pair. Returns None if clearly not a match."""
        # Same two hard gates as in the deterministic pass — enforced here as
        # well so embedding/LLM candidates can never sneak past a location or
        # date mismatch.
        if not self._same_location(a, b):
            return None
        if not _dates_within(
            a.get("meeting_date"), b.get("meeting_date"),
            self.config.date_tolerance_days,
        ):
            return None

        det = self._deterministic_match(a, b)
        if det is not None:
            confidence, evidence = det
            return confidence, "deterministic", evidence, "matched"

        score, evidence = self._embedding_match(a, b, emb)
        if score >= self.config.embedding_auto_accept:
            return score, "embedding", evidence, "matched"
        if score >= self.config.embedding_review_floor:
            # Borderline: optionally ask the LLM, else flag for human review.
            if self.config.use_llm and self.llm_adjudicator is not None:
                is_dup, llm_conf = self.llm_adjudicator(a, b)
                evidence = {**evidence, "llm_confidence": round(llm_conf, 3)}
                decision = "matched" if is_dup else "needs_review"
                return llm_conf, "llm", evidence, decision
            return score, "embedding", evidence, "needs_review"
        return None

    @staticmethod
    def _pick_primary(member_ids: Sequence[int], by_id: dict[int, dict]) -> int:
        """Prefer the record with the richest content as the canonical primary."""
        def richness(mid: int) -> int:
            r = by_id[mid]
            return sum(bool(r.get(f)) for f in
                       ("transcript", "agenda", "minutes", "video_url"))
        return max(member_ids, key=lambda m: (richness(m), -m))
