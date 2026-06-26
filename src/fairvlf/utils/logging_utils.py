"""Lightweight logging: console + a per-step metrics CSV.

We deliberately keep this dependency-free (no wandb required) so the repo runs
anywhere. The metrics CSV is what you commit to GitHub as the experiment record.
"""

import os
import csv
import logging
from datetime import datetime


def get_logger(run_dir: str, name: str = "fairvlf") -> logging.Logger:
    """Logger that prints to console AND appends to run_dir/train.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(os.path.join(run_dir, "train.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


class MetricsWriter:
    """Append per-step metrics to a CSV. Header is written on first row."""

    def __init__(self, run_dir: str, filename: str = "metrics.csv"):
        self.path = os.path.join(run_dir, filename)
        self._fields = None
        self._fh = None
        self._writer = None

    def log(self, step: int, metrics: dict) -> None:
        row = {"step": step, "time": datetime.now().isoformat(), **metrics}
        if self._writer is None:
            self._fields = list(row.keys())
            self._fh = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
            self._writer.writeheader()
        self._writer.writerow({k: row.get(k, "") for k in self._fields})
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
