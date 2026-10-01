"""Frozen configuration shared by kernel-lineage scripts."""

from __future__ import annotations

import json
from pathlib import Path

ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (ARTIFACT_ROOT / "config" / "study.json").read_text(encoding="utf-8")
)
LINEAGE = CONFIG["studies"]["lineage"]
CUTOFF_ISO = LINEAGE["cutoff"]
CUTOFF_DATE = CUTOFF_ISO[:10]
CUTOFF_COMMITS = LINEAGE["heads"]
REPOSITORIES = CONFIG["repositories"]
