# Port log: HashingEmbedder → Rust (`ef-rust`)

State of the native backend of `HashingEmbedder`, kept where the next session finds it (the `pyo3-port` method). Issue: [addaix/ef#56](https://github.com/addaix/ef/issues/56).

## Status

**Prototype, opt-in, not released.** `HashingEmbedder(backend="rust")` routes to the `ef-rust` wheel built from `rust/`; the default stays `backend="python"`, so nothing changes for anyone who does not opt in. `ef-rust` is not on PyPI yet, so the `ef[rust]` extra does not resolve until it is published.

## Assessment

`rust-port-assess` verdict: **port, ranked below the equate pilot** (score 73/100), because nothing calls this loop hard today. Gate 1: the best pure-Python variants (token memo, `np.unique` + `np.add.at`) gave 1.2–1.6×, below the 5× bar. Gate 2: sklearn's `HashingVectorizer` is 2–3× faster but changes every vector (murmur3, different signs and buckets). Expected gain: ~6.5–7.6× per call, ~3.4× on `ef.ingest`.

## What moved, what stayed

| Step | Where | Why |
|---|---|---|
| `str.lower` | Python | Its Unicode semantics (final sigma, `İ`) are the specification; calling it costs ~0.35 µs per text. |
| word splitting (`\w+`) | Rust, with Python's table | The facade passes the code-point ranges `re`'s `\w` matches in the running interpreter (748 ranges, computed once in ~0.2 s). The tokenizer cannot drift from `re` or from Python's Unicode version. |
| n-grams, counting in first-occurrence order | Rust | Distinct tokens go in a byte arena behind a `hashbrown::HashTable`, so nothing is allocated per token. The table's hasher is `foldhash`'s randomly seeded one, so nobody can precompute colliding n-grams (Python's `Counter` is protected the same way, by seeded `str` hashing). |
| 8-byte BLAKE2b, bucket, sign | Rust (`blake2b_simd`) | The v1 algorithm fixes the hash. It is the floor: ~165 ns per distinct token on this Mac. |
| signed weight, float32 accumulation | Rust | `f64` weight, rounded to `f32`, then an `f32` add, in token order: NumPy's NEP 50 rule for `float32 += float`. |
| L2 normalisation | Python | `np.linalg.norm` goes through BLAS `sdot`; keeping it in Python keeps the same bytes on each machine. |
| inputs outside the native domain | Python | An item that is not exactly `str` (a subclass may override `lower`), or a `str` with a lone surrogate, comes back in `skipped` and goes through the Python path, which also raises its errors. |

`_HASHING_EMBEDDER_VERSION` and `model_id` are unchanged: the outputs are bit-identical, so the backend is not part of the content address.

## Oracle

- `ef/_reference.py`: the Python path, frozen verbatim.
- `tests/test_differential.py`: Hypothesis compares `backend="rust"` against the reference **bit for bit** (`tobytes`), plus a 1,350-case trap grid and a sweep of every code point. It is skipped when `ef_rust` is not installed.
- Seeded examples: `İ`, final sigma, combining marks, a lone surrogate, non-decimal numerics, `dim=1`, an error after good rows, and the mixed-count text below.

Proof that the oracle sees:

- A stub kernel that returns zero rows fails 1,316 of 1,319 tests.
- Mutant "Rust's own `is_alphanumeric`" fails 106 tests.
- Mutant "accumulate in `f64`, round once" **survived** the first version of the suite. Random texts never put several tokens with counts ≥ 2 into one bucket at a small `dim`. Two things kill it now: a `_mixed_counts` strategy, and the seeded example `_MIXED` at `dim=3`, which fails 2 tests.

## Results (2026-11-23, Mac arm64, Python 3.12.12, GIL; process CPU time, load ~300 on 10 cores)

`benchmarks/bench_port.py --quick`, both columns through the public facade:

| Case | python | rust | speed-up |
|---|---|---|---|
| 1000 short lines | 48.6 ms | 9.7 ms | 5.0× |
| 100 segments × 350 words | 45.1 ms | 7.6 ms | 6.0× |
| 10 documents × 2000 words | 18.0 ms | 3.1 ms | 5.8× |
| `ef.ingest` call pattern (2 segments per call) | 44.9 ms | 8.2 ms | 5.5× |
| 3000 README segments (the assessment's corpus) | 2.26 s | 0.36 s | 6.2× |

A rerun after the review fixes (numpy-allocated output; load ~412) gave 5.3×, 5.8×, 6.1×, 5.4× and 6.1× on the same five cases; one run with the test suites alongside gave the corpus case at 3.7×, so read the ratios as ±1×.

`ef.ingest` on 1000 README segments, end to end: 1.06 s → 0.34 s (**3.1×**).

What is left in the native path, for 1000 short lines: the kernel takes 7.8 ms, BLAKE2b being most of it; the per-row norm loop in Python takes 2.2 ms; `lower` takes 0.35 ms.

## Decisions

- `gil_used = true`. The kernel borrows each `str`'s UTF-8 buffer, so it runs attached. The handle is frozen, and every call gets its own scratch buffers.
- **The output array is allocated by numpy** (`np.zeros`, as the Python path does) and the kernel fills it in place (`accumulate_into`). Size errors, the allocator (float32 `np.linalg.norm` can depend on memory alignment) and the array flags are then the Python path's own. An earlier version allocated in Rust and crashed the interpreter on `dim=2**61` (see Review).
- **numpy >= 2 is required** for `backend="rust"`: the kernel reproduces NEP 50 rounding (round the addend to float32, then add). numpy 1 adds in float64 and rounds once, one ulp apart. Both `ef-rust` and the `ef[rust]` extra pin `numpy>=2`, and the facade refuses numpy 1 with an `ImportError` that says why.
- n-gram bounds are clamped to `sys.maxsize` before they reach the kernel's `usize` (no text has that many words, so the result is unchanged).
- Errors: core errors map to `ValueError`. A caller cannot reach them through the facade, which validates first.
- The kernel handle is process-local: `HashingEmbedder.__getstate__` drops it, and it is rebuilt after unpickling.
- Build output goes outside Dropbox (`CARGO_TARGET_DIR`). The wheel is abi3-py310.

## Review (independent agent: review checklist plus adversarial pass)

| Severity | Finding | Fix |
|---|---|---|
| blocker | `dim=2**61` with empty input killed the process with SIGSEGV: the Rust-built array's byte stride overflowed | numpy allocates the output; `[]` and `["a"]` at `dim` 2\*\*61, 2\*\*63 and 2\*\*64 are now seeded examples (both backends raise the same exception class, from the same `np.zeros` call) |
| major | under numpy 1.x the backends differ by one ulp (`_MIXED`, `dim=3`) | `numpy>=2` pinned and checked |
| major | `dim`/`ngram` beyond `usize` raised `OverflowError` or `MemoryError` where Python raised `ValueError` or returned zeros | numpy allocation; clamped n-gram bounds; seeded examples |
| minor | Rust-allocated result had `OWNDATA=False`, other allocator | numpy allocation; a test checks the flags |
| minor | `np.str_` and other `str` subclasses take the Python path (correct, no speed-up) | documented in the facade |
| gap | nothing pinned the default Python path to `_reference.py` | `test_python_backend_equals_the_frozen_reference` (runs in CI, no Rust needed) |

The reviewer found no text-level divergence after 3,000 random cases (combining marks, case traps, NUL, line separators, astral characters, dim 1 to 2\*\*16), a 400k-word text, non-`str` items, surrogates, generators, pickling and input mutation.

### Second review (`rust-reviewer`)

| Severity | Finding | Fix |
|---|---|---|
| major | the counting table used unseeded `FxHash`: crafted documents could make counting quadratic | seeded `foldhash::fast::RandomState` (benchmark unchanged, 5.2–6.1×) |
| major | the cached kernel ignored later changes to `dim`/`ngram_range`/`sublinear_tf` on the instance | the cache is keyed on the configuration; `test_kernel_follows_configuration_changes` |
| major | the `ef[rust]` extra names a package that is not on PyPI yet | publish `ef-rust` before ef is released with the extra (a checklist item in the PR) |
| major | the oracle never runs in CI | `.github/workflows/rust-ci.yml` (fmt, clippy, cargo test, the differential and thorough runs, cargo deny), adapted from equate's |
| minor | instances pickled before the seam failed to load | class-level defaults for `backend` and `_native_kernel`; a test |
| minor | `e(texts, backend="rust")` was silently swallowed by `**backend` | `TypeError` that points at the constructor; a test |
| minor | `backend="rust"` without the wheel failed only on the first call | the import is checked at construction; a test |
| minor | `_HASHING_BACKENDS` repeated the `Literal` | `get_args(HashingBackend)` |
| minor | `cargo deny` never run | the `deny` job in `rust-ci.yml` |

## Checks

`cargo test -p ef-hashing-core` (7 tests), `cargo clippy --workspace --all-targets` with warnings denied, `cargo fmt --check` and `mypy.stubtest ef_rust._core` are all clean. In CI (`rust-ci.yml`, Linux, OpenBLAS), `cargo deny`, the differential tests on Python 3.10 and 3.13, and the thorough profile all pass, so the two backends agree on Linux too. Not run: free-threaded Python, and a comparison of golden vectors between the Mac and Linux (the BLAS question below).

## Open

- **Release path.** `.github/workflows/rust-release.yml` (from equate) builds the abi3 wheels for 5 targets and an sdist, and publishes from a tag `ef-rust-vX.Y.Z` through PyPI trusted publishing. Before the first tag, the owner configures a trusted publisher for the PyPI project `ef-rust`. The wads native-wheel matrix ([i2mint/wads#166](https://github.com/i2mint/wads/issues/166)) and the second package released from a subfolder ([i2mint/wads#220](https://github.com/i2mint/wads/issues/220)) are both open; when they land, this repo-local workflow can give way to them.
- **BLAS.** The docstring promises identical bytes across machines, but the norm goes through BLAS (Accelerate on the Mac, OpenBLAS on Linux). That holds for both backends alike: compare golden vectors from the Mac and Linux before promising it.
- **CI.** ef's CI does not build Rust, so the differential tests skip there. A Linux job running `maturin develop --release` and then `pytest tests/test_differential.py` would close that gap.
