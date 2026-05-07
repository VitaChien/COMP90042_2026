"""Generic helpers used across modules."""

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Seed every RNG we use for reproducibility (rule #5)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def load_json(path: Path | str) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def get_logger(name: str = "factcheck", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        log.addHandler(h)
        log.propagate = False
    log.setLevel(level)
    return log


@contextmanager
def timer(label: str, log=None):
    log = log or get_logger()
    t0 = time.perf_counter()
    yield
    log.info("%s took %.2fs", label, time.perf_counter() - t0)
