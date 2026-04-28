import random
from pathlib import Path

import numpy as np

from src.utils import get_logger, load_json, save_json, set_seed


def test_set_seed_makes_python_random_deterministic():
    set_seed(123)
    a = [random.random() for _ in range(3)]
    set_seed(123)
    b = [random.random() for _ in range(3)]
    assert a == b


def test_set_seed_makes_numpy_random_deterministic():
    set_seed(7)
    a = np.random.rand(5)
    set_seed(7)
    b = np.random.rand(5)
    assert (a == b).all()


def test_save_load_json_roundtrip(tmp_path: Path):
    obj = {"claim-1": {"label": "SUPPORTS", "ev": ["e-0"]}}
    fp = tmp_path / "x.json"
    save_json(obj, fp)
    assert load_json(fp) == obj


def test_get_logger_returns_named_logger():
    log = get_logger("foo")
    assert log.name == "foo"
