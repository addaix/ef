"""Native kernel for ``ef.embedders.HashingEmbedder`` (``backend="rust"``).

This package is an implementation detail of `ef <https://github.com/addaix/ef>`_:
install it and pass ``backend="rust"`` to :class:`ef.HashingEmbedder`. The kernel
here is private, with no defaults and no validation: ef's facade owns the public
signature, lowercasing, validation, normalisation and the routing of inputs outside
the native domain. It reproduces the pure-Python backend bit for bit (ef's
``tests/test_differential.py`` checks it).
"""

from importlib.metadata import PackageNotFoundError, version

from ef_rust._core import DEBUG_BUILD, HashingKernel

__all__ = ["DEBUG_BUILD", "HashingKernel"]

try:
    __version__ = version("ef-rust")
except PackageNotFoundError:  # a source tree without an install
    __version__ = "0.0.0"
