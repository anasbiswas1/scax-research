"""Frozen configuration files (plan Section 9.2).

Notebook 05 writes one file per task under configs/frozen/ before calibration split 1 is opened;
every later notebook loads it through load_frozen, which checks the SHA-256 recorded in
results/05_frozen.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, indent=1, default=float) + "\n"


def write_frozen(path, cfg: dict) -> str:
    text = canonical(cfg)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def load_frozen(path, expected_sha256: str | None = None):
    text = Path(path).read_text()
    h = hashlib.sha256(text.encode()).hexdigest()
    if expected_sha256 is not None and h != expected_sha256:
        raise ValueError(f"{path} changed after freezing: {h} != {expected_sha256}")
    return json.loads(text), h
