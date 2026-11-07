
"""Project locations. The Drive root can be overridden with SCAX_DRIVE_ROOT (used by tests)."""
import os
from pathlib import Path

DRIVE_ROOT = Path(os.environ.get("SCAX_DRIVE_ROOT", "/content/drive/MyDrive/SCAX_Research"))
REPO_DIR = Path(os.environ.get("SCAX_REPO_ROOT", str(DRIVE_ROOT / "scax-research")))
DATASETS = DRIVE_ROOT / "datasets"
RAW = DATASETS / "raw"
DERIVED = DATASETS / "derived"
RESULTS = REPO_DIR / "results"
FIGURES = RESULTS / "figures"


def ensure_dirs():
    for d in (DATASETS, RAW, DERIVED, RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
