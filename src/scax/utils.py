
"""Seeding, hashing and result writing shared by every notebook."""
import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "z3", "pysat", "statsmodels"):
        try:
            mod = __import__(name)
            v = getattr(mod, "__version__", None)
            if name == "z3":
                v = mod.get_version_string()
            out[name] = v or "installed"
        except Exception:
            out[name] = None
    return out


def write_json(path, obj, add_meta: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if add_meta and isinstance(obj, dict):
        obj = {**obj, "_written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "_versions": versions()}
    path.write_text(json.dumps(obj, indent=2, default=float))
