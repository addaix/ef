"""Frozen pure-Python reference of :class:`ef.embedders.HashingEmbedder` (do not edit).

This module is the **oracle** for the optional native backend
(``HashingEmbedder(backend="rust")``, provided by the separate ``ef-rust`` wheel
built from ``rust/``). It is a verbatim copy of the vectorizing path of
``HashingEmbedder`` as of ``_HASHING_EMBEDDER_VERSION = 1`` (ef 0.1.30), kept
apart so that an edit to ``ef.embedders`` cannot silently move the reference the
native backend is tested against (``tests/test_differential.py``).

**Frozen: do not edit.** A change to how a hashing vector is produced is a new
algorithm version: bump ``_HASHING_EMBEDDER_VERSION`` in ``ef.embedders``, change
both backends, and re-freeze this file in the same change.

Bit-exactness is defined under numpy >= 2: ``row[bucket] += sign * weight`` rounds
the Python float to float32 before adding (NEP 50); numpy 1 adds in float64 and
rounds once, so its bytes differ in the last bit, for this reference too.

Not exported from ``ef``.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator

import numpy as np

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def word_ngrams(text: str, ngram_range: tuple[int, int]) -> Iterator[str]:
    """Yield the lowercased word n-grams of ``text`` for ``ngram_range``.

    >>> list(word_ngrams('The quick fox', (1, 2)))
    ['the', 'quick', 'fox', 'the quick', 'quick fox']
    """
    words = _WORD_RE.findall(text.lower())
    n_min, n_max = ngram_range
    for n in range(n_min, n_max + 1):
        if n == 1:
            yield from words
        else:
            for i in range(len(words) - n + 1):
                yield " ".join(words[i : i + n])


def hash_bucket(token: str, dim: int) -> tuple[int, float]:
    """Hash ``token`` to a ``(bucket, sign)`` pair with an 8-byte blake2b digest."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    h = int.from_bytes(digest, "big")
    return (h >> 1) % dim, (1.0 if h & 1 else -1.0)


def vectorize_into(
    text: str,
    row: np.ndarray,
    *,
    dim: int,
    ngram_range: tuple[int, int],
    sublinear_tf: bool,
) -> None:
    """Fill the float32 ``row`` in place with the L2-normalized hashed vector of ``text``."""
    counts = Counter(word_ngrams(text, ngram_range))
    for token, count in counts.items():
        bucket, sign = hash_bucket(token, dim)
        weight = 1.0 + math.log(count) if sublinear_tf else float(count)
        row[bucket] += sign * weight
    norm = float(np.linalg.norm(row))
    if norm > 0.0:
        row /= norm


def hashing_vectors(
    texts: Iterable[str],
    *,
    dim: int,
    ngram_range: tuple[int, int],
    sublinear_tf: bool,
) -> np.ndarray:
    """The ``(len(texts), dim)`` float32 matrix ``HashingEmbedder.__call__`` returns.

    >>> m = hashing_vectors(['hello world', ''], dim=8, ngram_range=(1, 2), sublinear_tf=True)
    >>> m.shape, m.dtype.name, bool(m[1].any())
    ((2, 8), 'float32', False)
    """
    texts = list(texts)
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        vectorize_into(
            text,
            out[row],
            dim=dim,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
        )
    return out
