"""Embedding generation for the dedup similarity pass.

The default backend is a dependency-free, deterministic hashing embedding so
the pipeline runs out of the box and tests are reproducible. A real deployment
can swap in a transformer backend (e.g. sentence-transformers) by passing a
different ``EmbeddingBackend`` — the dedup code only depends on the interface.
"""

from __future__ import annotations

import abc
import hashlib
import math
import re
from typing import Optional, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class EmbeddingBackend(abc.ABC):
    @abc.abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class HashingEmbedding(EmbeddingBackend):
    """A deterministic bag-of-words hashing embedding (no external deps).

    Each token is hashed into one of ``dim`` buckets with a signed weight;
    the resulting vector is L2-normalized. This is a "hashing trick" / random
    projection — crude versus a transformer, but stable, fast, and good enough
    to demonstrate the cosine-similarity dedup stage end to end.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokens(text):
            h = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors. Returns 0.0 for zero vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def embedding_text(
    transcript: Optional[str],
    agenda: Optional[str],
    meeting_name: Optional[str],
    max_chars: int = 2000,
) -> str:
    """Build the text used to embed a meeting.

    Prefers the first few transcript paragraphs; falls back to agenda, then to
    the meeting name. Truncated to ``max_chars`` to focus on the opening.
    """
    for candidate in (transcript, agenda, meeting_name):
        if candidate and candidate.strip():
            return candidate.strip()[:max_chars]
    return ""
