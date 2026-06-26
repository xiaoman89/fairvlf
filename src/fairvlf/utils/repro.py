"""Reproducibility utilities: seeding, run directories, and provenance.

Every training run creates a timestamped directory that records exactly what
produced it (config, code version, seed, full log). This is what makes the
public repository's results traceable.
"""

import os
import sys
import json
import random
import subprocess
from datetime import datetime

import numpy as np


def set_seed(seed: int) -> None:
    """Fix all RNG seeds we can, for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms where available (may be slower).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def git_commit_hash() -> str:
    """Return the current git commit hash, or 'unknown' if not in a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def git_is_dirty() -> bool:
    """True if there are uncommitted changes (so we can warn in the log)."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        )
        return len(out.decode().strip()) > 0
    except Exception:
        return False


def make_run_dir(run_name: str, base: str = "runs") -> str:
    """Create runs/<run_name>_<timestamp>/ and return its path."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"{run_name}_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


def save_provenance(run_dir: str, config: dict) -> None:
    """Write the resolved config + code/environment provenance into run_dir.

    This single file lets anyone reproduce (or audit) the run later.
    """
    provenance = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit_hash(),
        "git_dirty": git_is_dirty(),
        "python": sys.version,
        "argv": sys.argv,
        "config": config,
    }
    with open(os.path.join(run_dir, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2, default=str)
