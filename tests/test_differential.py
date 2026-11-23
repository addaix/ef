"""Differential tests: ``HashingEmbedder(backend="rust")`` must equal the frozen reference.

The native backend (the optional ``ef-rust`` wheel, built from ``rust/``) must give
the **same bytes** as the pure-Python algorithm frozen in ``ef/_reference.py``: the
vectors are content-addressed through ``model_id``, which does not name the backend.
So results compare bit for bit (``tobytes``), not within a tolerance, and errors
compare by class.

Built from the ``pyo3-port`` skill's template. The module is skipped when ``ef_rust``
is not installed (ef's CI does not build Rust yet). Profiles: ``fast`` (default,
deterministic) and ``thorough``::

    HYPOTHESIS_PROFILE=thorough pytest tests/test_differential.py
"""

from __future__ import annotations

import copy
import math
import os
import string

import numpy as np
import pytest
from hypothesis import example, given, note, settings
from hypothesis import strategies as st

ef_rust = pytest.importorskip("ef_rust")

# isort: split
from ef import HashingEmbedder  # noqa: E402 - the public facade
from ef import _reference as ref  # noqa: E402 - the frozen pre-port implementation

settings.register_profile("fast", max_examples=300, derandomize=True, deadline=None)
settings.register_profile("thorough", max_examples=5000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "fast"))


# --- Strategies seeded with the Python -> Rust traps --------------------------------

TRAP_CHARS = (
    "\x1c\x1d\x1e\x1f\x85\xa0 "  # whitespace that Python and Rust classify differently
    "_‿⁀"  # connector punctuation: _ ‿ ⁀
    "́ͅि"  # combining marks: Mn, Mn with Other_Alphabetic, Mc
    "²½①Ⅻ"  # non-decimal numerics: ² ½ ① Ⅻ
    "İΣßẞǅ"  # case traps: İ Σ ß ẞ ǅ
    "一\U0001f600"  # CJK, emoji (4-byte UTF-8)
)


def py_text(*, max_size: int = 40, surrogates: bool = True) -> st.SearchStrategy[str]:
    """Python ``str`` values, including what a Rust ``&str`` cannot hold (lone surrogates).

    Half are free text; half are short tokens joined by separators, so that a single
    trap character often stands alone as a whole word or field.
    """
    any_char = st.characters(exclude_categories=() if surrogates else ("Cs",))
    common = st.sampled_from(string.ascii_letters + string.digits + " " + TRAP_CHARS)
    free = st.text(st.one_of(common, common, any_char), max_size=max_size)
    token = st.one_of(
        st.sampled_from(TRAP_CHARS),
        st.text(st.one_of(common, any_char), min_size=1, max_size=4),
    )
    joined = st.lists(token, max_size=max(1, max_size // 4)).flatmap(
        lambda toks: st.sampled_from([" ", ",", "\t", ""]).map(
            lambda sep: sep.join(toks)
        )
    )
    return st.one_of(free, joined)


def py_ints(*, small: tuple[int, int] = (-2, 8)) -> st.SearchStrategy[int]:
    """Mostly small ints (where sizes and thresholds interact with short inputs), plus the
    64-bit edges and unbounded Python ints."""
    edges = [0, -1, 2**31 - 1, 2**31, 2**63 - 1, 2**63, 2**64 - 1, 2**64, -(2**63) - 1]
    return st.one_of(
        st.integers(*small), st.integers(-3, 100), st.sampled_from(edges), st.integers()
    )


def py_floats() -> st.SearchStrategy[float]:
    """Floats with NaN, infinities, signed zero and subnormals."""
    return st.one_of(
        st.floats(), st.sampled_from([0.0, -0.0, math.nan, math.inf, -math.inf, 5e-324])
    )


# --- Outcome comparison ---------------------------------------------------------------


def same(a, b, *, rel: float = 0.0, abs_: float = 0.0) -> bool:
    """Structural equality that also compares types; floats may get a tolerance."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        if a == b:
            return math.copysign(1.0, a) == math.copysign(1.0, b)
        return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(
            same(x, y, rel=rel, abs_=abs_) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, dict):  # insertion order is part of a dict's observable behaviour
        return list(a) == list(b) and all(
            same(a[k], b[k], rel=rel, abs_=abs_) for k in a
        )
    return a == b


def close(*, rel: float = 1e-12, abs_: float = 0.0):
    """A ``compare`` for float results that may differ in the last bits (exp/log, sums)."""
    return lambda a, b: same(a, b, rel=rel, abs_=abs_)


def outcome(fn, args: tuple, kwargs: dict, *, messages: bool = False):
    """Run ``fn`` on copies of the inputs: ('ok', value, args after) or ('err', class, args after)."""
    args, kwargs = copy.deepcopy(args), copy.deepcopy(kwargs)
    try:
        result = ("ok", fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - exception classes are part of the API
        result = ("err", type(exc), str(exc)) if messages else ("err", type(exc))
    return (*result, args, kwargs)


_NO_ARGS, _NO_KWARGS = st.just(()), st.just({})


def differential(
    ref_fn,
    port_fn,
    /,
    *,
    args=_NO_ARGS,
    kwargs=_NO_KWARGS,
    compare=same,
    messages=False,
    examples=(),
):
    """Build a Hypothesis test asserting ``port_fn`` behaves like ``ref_fn`` on generated inputs.

    ``examples`` is the seed corpus: ``(args, kwargs)`` pairs always run first. Add every
    divergence found during the port, and any trap too narrow for random generation to hit.
    """

    def test(args, kwargs):
        expected = outcome(ref_fn, args, kwargs, messages=messages)
        actual = outcome(port_fn, args, kwargs, messages=messages)
        note(f"reference: {expected[:-2]!r}")
        note(f"port:      {actual[:-2]!r}")
        if expected[0] == actual[0] == "ok":
            assert compare(actual[1], expected[1]), "results differ"
        else:
            assert actual[:-2] == expected[:-2], (
                "outcomes differ (value vs exception, or exception class)"
            )
        assert same(actual[-2:], expected[-2:]), "arguments were mutated differently"

    for ex_args, ex_kwargs in examples:
        test = example(args=ex_args, kwargs=ex_kwargs)(test)
    test = given(args=args, kwargs=kwargs)(test)
    test.__qualname__ = test.__name__ = (
        f"test_{getattr(ref_fn, '__name__', 'fn')}_matches_reference"
    )
    return test


def trap_grid(ref_fn, port_fn, /, *, args, kwargs=({},), compare=same, messages=False):
    """Build a parametrised test over every ``args`` x ``kwargs`` combination.

    Random generation rarely hits a conjunction (one trap character *and* one exact
    parameter value), so pair the Hypothesis test with this deterministic grid: every
    trap character alone and inside a word, times the small parameter values.
    """
    cases = [(a, k) for a in args for k in kwargs]

    @pytest.mark.parametrize(("case_args", "case_kwargs"), cases)
    def test(case_args, case_kwargs):
        expected = outcome(ref_fn, case_args, case_kwargs, messages=messages)
        actual = outcome(port_fn, case_args, case_kwargs, messages=messages)
        if expected[0] == actual[0] == "ok":
            assert compare(actual[1], expected[1]), (actual[:-2], expected[:-2])
        else:
            assert actual[:-2] == expected[:-2], (actual[:-2], expected[:-2])
        assert same(actual[-2:], expected[-2:]), "arguments were mutated differently"

    test.__qualname__ = test.__name__ = (
        f"test_{getattr(ref_fn, '__name__', 'fn')}_trap_grid"
    )
    return test


def trap_strings() -> list[str]:
    """Each trap character alone, inside a word, and doubled: inputs for ``trap_grid``."""
    return [
        "",
        *TRAP_CHARS,
        *(f"ab{c}cd" for c in TRAP_CHARS),
        *(c + c for c in TRAP_CHARS),
    ]


# --- The native module must be the one under test -------------------------------------


def test_native_extension_is_loaded():
    """Fails if a stale build, a fallback or the source tree is what got imported."""
    assert ef_rust._core.__file__.endswith((".so", ".pyd")), ef_rust._core.__file__


# --- HashingEmbedder ------------------------------------------------------------------


def same_array(a, b) -> bool:
    """Bit-identical arrays: same dtype, same shape, same bytes."""
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def reference_vectors(texts, /, *, dim=512, ngram_range=(1, 2), sublinear_tf=True):
    return ref.hashing_vectors(
        texts, dim=dim, ngram_range=ngram_range, sublinear_tf=sublinear_tf
    )


def native_vectors(texts, /, **config):
    return HashingEmbedder(backend="rust", **config)(texts)


class _ShoutingStr(str):
    """A ``str`` subclass whose ``lower`` differs: it must take the Python path."""

    def lower(self):
        return self.upper()


def _repeated() -> st.SearchStrategy[str]:
    """One token repeated many times: exercises the sublinear weight ``1 + log(count)``."""
    token = st.one_of(
        st.sampled_from(["a", "hello", "\u00e9t\u00e9", "x1"]), py_text(max_size=4)
    )
    return st.tuples(
        token, st.integers(1, 60), st.sampled_from([" ", "  ", ",", "\n"])
    ).map(lambda t: t[2].join([t[0]] * t[1]))


def _mixed_counts() -> st.SearchStrategy[str]:
    """Many distinct tokens with counts above 1, shuffled: several sublinear weights land
    in one bucket at a small ``dim``, which is where float32 rounding order shows."""
    vocab = st.lists(
        st.sampled_from([f"w{i}" for i in range(40)]),
        min_size=2,
        max_size=16,
        unique=True,
    )
    blocks = vocab.flatmap(
        lambda ws: st.tuples(*[st.integers(1, 9) for _ in ws]).map(
            lambda ns: [w for w, n in zip(ws, ns) for _ in range(n)]
        )
    )
    return blocks.flatmap(st.permutations).map(" ".join)


_MIXED = " ".join(f"w{i}" for i in range(12) for _ in range(i % 7 + 2))

_texts = st.lists(
    st.one_of(
        py_text(max_size=60),
        py_text(max_size=60),
        _repeated(),
        _mixed_counts(),
        st.none(),  # not a str: the reference raises, so must the port
        st.builds(_ShoutingStr, py_text(max_size=20)),
    ),
    max_size=6,
)

_config = st.fixed_dictionaries(
    {},
    optional={
        "dim": st.one_of(st.sampled_from([1, 2, 3, 7, 64, 512]), st.integers(1, 2**20)),
        "ngram_range": st.integers(1, 4).flatmap(
            lambda lo: st.tuples(st.just(lo), st.integers(lo, lo + 3))
        ),
        "sublinear_tf": st.booleans(),
    },
)

test_hashing_vectors = differential(
    reference_vectors,
    native_vectors,
    args=st.tuples(_texts),
    kwargs=_config,
    compare=same_array,
    examples=[
        ((["\u0130stanbul \u0130"],), {}),  # 'İ'.lower() is 2 code points
        (
            (["\u03a3\u0391\u03a3 \u039f\u0394\u03a5\u03a3\u03a3\u0395\u038e\u03a3"],),
            {},
        ),  # final sigma
        (
            (["e\u0301 x\u0332y \u0905\u093f"],),
            {"ngram_range": (1, 3)},
        ),  # combining marks
        ((["\ud800abc def"],), {}),  # a lone surrogate: Python path
        ((["\u00b2\u00bd \u2460 \u216b \u0663\u0664"],), {}),  # non-decimal numerics
        ((["a " * 40],), {"dim": 1}),  # every token in one bucket
        (([_MIXED],), {"dim": 3}),  # float32 += float rounds the addend first (NEP 50)
        ((["the quick brown fox", "", None],), {}),  # an error after good rows
        # sizes numpy refuses: the same exception class, never a crash
        (([],), {"dim": 2**61}),
        ((["a"],), {"dim": 2**61}),
        (([],), {"dim": 2**63}),
        ((["a"],), {"dim": 2**64}),
        # an n-gram size past usize: no text has that many words, so zeros
        # (not a wide range: the Python path loops over every size in it)
        ((["a b"],), {"ngram_range": (2**64, 2**64)}),
    ],
)

test_hashing_vectors_grid = trap_grid(
    reference_vectors,
    native_vectors,
    args=[([s, f"x {s} y {s}"],) for s in trap_strings()] + [([_MIXED],)],
    kwargs=[
        {"dim": d, "ngram_range": r, "sublinear_tf": tf}
        for d in (1, 7, 512)
        for r in ((1, 1), (1, 2), (2, 3))
        for tf in (True, False)
    ],
    compare=same_array,
)


def test_every_code_point_tokenizes_like_python():
    """Each code point alone and between two letters: the tokenizer table is exact."""
    chars = [chr(c) for c in range(0x110000) if not 0xD800 <= c < 0xE000]
    for start in range(0, len(chars), 4096):
        block = chars[start : start + 4096]
        texts = [f"q{c}z {c}" for c in block]
        assert same_array(
            native_vectors(texts, dim=4096),
            reference_vectors(texts, dim=4096),
        ), f"block starting at U+{ord(block[0]):04X}"


def test_native_output_is_a_numpy_owned_array():
    out = native_vectors(["hello world"])
    assert out.flags.owndata and out.flags.c_contiguous and out.flags.writeable


def test_backend_is_not_in_model_id():
    assert HashingEmbedder(backend="rust").model_id == HashingEmbedder().model_id


def test_native_embedder_pickles():
    import pickle

    e = HashingEmbedder(backend="rust")
    e(["warm the kernel"])
    clone = pickle.loads(pickle.dumps(e))
    assert same_array(clone(["hello world"]), e(["hello world"]))
