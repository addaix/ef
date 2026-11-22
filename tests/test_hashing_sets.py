"""Set-valued fields hash identically across processes (PYTHONHASHSEED)."""

import subprocess
import sys

import ef
from ef.hashing import canonical_json

_SNIPPET = "import ef; print(ef.content_hash(ef.hashing.canonical_json({'t': {'x', 'y', 'z', 'w'}})))"


def test_set_order_does_not_matter():
    assert canonical_json({"t": {"a", "b", "c"}}) == canonical_json({"t": {"c", "b", "a"}})


def test_hash_is_stable_across_hash_seeds():
    import os

    outs = {
        subprocess.run(
            [sys.executable, "-c", _SNIPPET],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(outs) == 1


def test_inputs_without_sets_are_unchanged():
    assert canonical_json({"b": 1, "a": [1, "x"]}) == '{"a":[1,"x"],"b":1}'
