"""Portable path resolution for nomocomp benchmarks.

All benchmarks import this module to get workspace-relative paths
without hardcoding any absolute session directory.

Layout assumed:
    workspace/
        observer_geometry/src/nomogeo/...
        nomocomp/
            benchmarks/_paths.py   <-- this file
            src/nomocomp/...
"""
from __future__ import annotations

import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
NOMOCOMP_ROOT = BENCHMARKS_DIR.parent
WORKSPACE = NOMOCOMP_ROOT.parent

NOMOGEO_SRC = WORKSPACE / "observer_geometry" / "src"
NOMOCOMP_SRC = NOMOCOMP_ROOT / "src"

for _p in (NOMOGEO_SRC, NOMOCOMP_SRC):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
