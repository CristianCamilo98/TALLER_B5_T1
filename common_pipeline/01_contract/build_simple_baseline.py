#!/usr/bin/env python3
"""Build the bootstrap + Gaussian jitter baseline in normalized space."""

from __future__ import annotations

import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import baseline


def main() -> int:
    result = baseline.build_baseline()
    print(f"baseline written: {result.path}")
    print(f"shape: {result.shape}")
    print(f"sha256: {result.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
