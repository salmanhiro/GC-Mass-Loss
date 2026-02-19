"""Basic example: import gc_tidal and verify a function works.

Install the package first:
    pip install -e .

Then run this script:
    python scripts/basic.py
"""

import gc_tidal

version = gc_tidal.get_version()
print(f"gc_tidal version: {version}")
assert version == gc_tidal.__version__, "get_version() mismatch"
print("Basic check passed.")
