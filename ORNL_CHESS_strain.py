#!/usr/bin/env python3
"""Compatibility wrapper for ``bokeh serve ORNL_CHESS_strain.py``."""
from __future__ import annotations

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

runpy.run_module("nsdf_dashboard.ORNL_CHESS_strain", run_name="__main__")

