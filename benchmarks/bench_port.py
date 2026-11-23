#!/usr/bin/env python3
"""Benchmark ``HashingEmbedder(backend="rust")`` against ``backend="python"``, end to end.

From the ``pyo3-port`` skill's template. Both columns go through the public facade, so
the native column includes everything a user pays: lowercasing, the boundary, the
per-row L2 normalisation in Python. Times are **process CPU time** (the median of
``--repeat`` runs, each auto-ranged), so a busy machine slows both columns alike.

Sizes follow the port assessment's workload: short lines (one sentence), ingest-sized
segments (~350 words, the default ~512-token chunk), long documents (~2,000 words), and
``ef.ingest``'s real call pattern, one document (~2 segments) per call::

    python benchmarks/bench_port.py                     # synthetic text, all cases
    python benchmarks/bench_port.py --quick             # fewer repeats
    python benchmarks/bench_port.py --corpus texts.json # also time a JSON list of your strings
    python benchmarks/bench_port.py --json out.json

Needs the ``ef-rust`` wheel (``rust/``), built with ``--release``: it refuses a debug build.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
import timeit
from statistics import median

from ef import HashingEmbedder

SEED = 0  # inputs are deterministic, so two runs time the same work

_VOCAB = (
    "the of and to a in is for that with as on by this be are from it an or at "
    "embedding vector search corpus segment index query document model token store "
    "python rust numpy array function package install default cache graph source "
    "café naïve Straße résumé über 数据 日本語 ΣΑΣ"
).split()


def _text(n_words: int, rng: random.Random) -> str:
    """``n_words`` words of prose-like shape: Zipf-ish word choice, some punctuation."""
    words = rng.choices(
        _VOCAB, weights=[1 / (i + 1) for i in range(len(_VOCAB))], k=n_words
    )
    return " ".join(w + rng.choice(["", "", "", ",", ".", ":"]) for w in words)


def _batch(n_texts: int, n_words: tuple[int, int]):
    def factory(rng: random.Random) -> list[str]:
        return [_text(rng.randint(*n_words), rng) for _ in range(n_texts)]

    return factory


# (label, texts factory, calls per timed run): one call embeds the whole list, except
# the last case, which embeds it two texts at a time, as ef.ingest's embed_op does.
CASES = [
    ("1000 short lines", _batch(1000, (5, 40)), "batch"),
    ("100 segments x 350 w", _batch(100, (350, 350)), "batch"),
    ("10 docs x 2000 w", _batch(10, (2000, 2000)), "batch"),
    ("ingest: 2 segs/call", _batch(100, (350, 350)), "pairs"),
]


def conditions() -> dict:
    """What a speed-up needs next to it to mean anything."""
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    return {
        "python": platform.python_version(),
        "build": "GIL" if gil else "free-threaded",
        "machine": f"{platform.system()} {platform.machine()}",
        "cpus": os.cpu_count(),
        "load": round(os.getloadavg()[0], 1) if hasattr(os, "getloadavg") else None,
        "timer": "process CPU time",
    }


def _runner(embedder, texts: list[str], mode: str):
    if mode == "pairs":
        pairs = [texts[i : i + 2] for i in range(0, len(texts), 2)]
        return lambda: [embedder(p) for p in pairs]
    return lambda: embedder(texts)


def time_call(fn, *, repeat: int) -> float:
    """Median CPU seconds per call over ``repeat`` runs, each auto-ranged to >= 0.2 s."""
    timer = timeit.Timer(fn, timer=time.process_time)
    number, _ = timer.autorange()
    return median(t / number for t in timer.repeat(repeat=repeat, number=number))


def _fmt(seconds: float) -> str:
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6)):
        if seconds >= scale:
            return f"{seconds / scale:7.2f} {unit}"
    return f"{seconds / 1e-9:7.2f} ns"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="3 repeats")
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument(
        "--corpus", metavar="PATH", help="a JSON list of strings to time too"
    )
    parser.add_argument("--json", metavar="PATH", help="also write the results as JSON")
    opts = parser.parse_args(argv)

    import ef_rust

    if ef_rust.DEBUG_BUILD:
        print("refusing to benchmark a debug build: build ef-rust with --release first")
        return 2

    python, native = HashingEmbedder(), HashingEmbedder(backend="rust")
    native(["warm up: build the kernel and the word table once"])
    cases = [
        (label, factory(random.Random(SEED)), mode) for label, factory, mode in CASES
    ]
    if opts.corpus:
        with open(opts.corpus, encoding="utf-8") as f:
            cases.append(
                (f"corpus ({os.path.basename(opts.corpus)})", json.load(f), "batch")
            )

    repeat = 3 if opts.quick else opts.repeat
    cond = conditions()
    print("conditions: " + ", ".join(f"{k}={v}" for k, v in cond.items()))
    rows = []
    for label, texts, mode in cases:
        assert python(texts).tobytes() == native(texts).tobytes(), (
            f"{label}: outputs differ"
        )
        t_py = time_call(_runner(python, texts, mode), repeat=repeat)
        t_rs = time_call(_runner(native, texts, mode), repeat=repeat)
        rows.append(
            {"case": label, "python_s": t_py, "rust_s": t_rs, "speedup": t_py / t_rs}
        )
        print(
            f"{label:<28} python {_fmt(t_py)}   rust {_fmt(t_rs)}   x{t_py / t_rs:,.1f}"
        )
    if opts.json:
        with open(opts.json, "w", encoding="utf-8") as f:
            json.dump({"conditions": cond, "results": rows}, f, indent=2)
        print(f"wrote {opts.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
