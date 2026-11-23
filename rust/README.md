# ef-rust

Native (Rust) kernel for [`ef`](https://github.com/addaix/ef)'s `HashingEmbedder`, the zero-install default embedder. Install it next to ef and pass `backend="rust"`:

```python
from ef import HashingEmbedder

embed = HashingEmbedder(backend="rust")
embed(["the quick brown fox", "hello world"])  # same bytes as backend="python", faster
```

The vectors are bit-identical to the pure-Python backend, so `model_id` (and every cache key and artifact id derived from it) does not change with the backend. ef's `tests/test_differential.py` checks it.

The Python package `ef_rust` holds a private kernel only; the public API is ef's.

## Layout

- `core/`: the pure-Rust crate `ef-hashing-core` (no Python types; `cargo test -p ef-hashing-core`).
- `py/`: the PyO3 binding crate, built by maturin as `ef_rust._core`.
- `python/ef_rust/`: the Python package (re-exports, the hand-written stub `_core.pyi`).

## Development

```bash
cd rust
maturin develop --release --uv     # into the active virtual environment
cargo test -p ef-hashing-core
cargo clippy --workspace --all-targets
cd .. && python -m pytest tests/test_differential.py tests/test_hashing_embedder.py
python benchmarks/bench_port.py
```
