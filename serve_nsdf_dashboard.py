#!/usr/bin/env python3
"""Compatibility wrapper for the packaged dashboard service."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from nsdf_dashboard.serve_nsdf_dashboard import main


if __name__ == "__main__":
    main()

