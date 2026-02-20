"""Backward-compatibility shim: gc module has been renamed to dynamics."""

# Re-export everything from the canonical location so that existing code
# using ``streamcutter.gc`` continues to work.
from streamcutter.dynamics import (  # noqa: F401
    _DEFAULT_TABLE,
    _normalize,
    GCParams,
    PotentialFactory,
)
