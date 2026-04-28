# Climate Fact-Checking Plan B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage fact-checking pipeline (BM25 → BERT cross-encoder → RoBERTa classifier with noise-aware training) for COMP90042 Project that beats baseline on dev set and is portable to the provided ipynb template.

**Architecture:** Stage 1 retrieval = BM25 (1.2M → top-200) + BERT cross-encoder re-ranker trained with hard negatives (top-200 → top-K). Stage 2 classification = RoBERTa fine-tuned on a 50/50 mix of (claim, gold-evidence) and (claim, retrieved-evidence) pairs to mitigate train-test distribution mismatch.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace Transformers, rank_bm25, scikit-learn, pytest, tqdm.

---

## Locked Defaults

The following defaults are used unless changed in `src/config.py`:

| Param | Default | Notes |
|-------|---------|-------|
| Random seed | `42` | reproducibility rule |
| BM25 candidates | `200` | first-stage |
| Final K (top-K evidence) | `4` | tunable in `{3,4,5,6}` |
| Cross-encoder backbone | `bert-base-uncased` | 110M, fits T4 |
| Classifier backbone | `roberta-base` | 125M, fits T4 |
| Hard negatives per positive | `4` | from BM25 top-50 \ gold |
| Noise mix ratio (gold:retrieved) | `0.5 : 0.5` | classifier training |
| Cross-enc LR / epochs / batch | `2e-5 / 2 / 32` | with grad-accum if needed |
| Classifier LR / epochs / batch | `2e-5 / 3 / 16` | |
| Max seq len | `256` (CE), `384` (cls) | claim ~20w, evidence ~20w |

---

## File Structure

```
COMP90042_2026/
├── data/                          # existing (evidence.json must be downloaded)
├── src/                           # NEW: importable modules
│   ├── __init__.py
│   ├── config.py                  # dataclass with all hyperparams + paths
│   ├── utils.py                   # set_seed, json io, logger, timer
│   ├── data_loader.py             # load_claims, load_evidence
│   ├── preprocessing.py           # tokenize_for_bm25
│   ├── retriever_bm25.py          # build_index, BM25Retriever
│   ├── retriever_cross_enc.py     # CrossEncoderModel + dataset + train + rerank
│   ├── classifier.py              # ClaimClassifier + dataset + train + predict
│   ├── pipeline.py                # FactCheckingPipeline (orchestrator)
│   └── evaluator.py               # wraps eval.py, adds per-class metrics
├── scripts/                       # NEW: reproducible entry points
│   ├── __init__.py                #   (required for `python -m scripts.X`)
│   ├── build_bm25.py              # M1
│   ├── train_cross_encoder.py     # M2
│   ├── train_classifier.py        # M3
│   ├── run_inference.py           # M4 (modes: bm25-random / retriever-only / full / oracle)
│   ├── ablation_table.py          # M4
│   └── error_analysis.py          # M4
├── tests/                         # NEW: pytest suite
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_evaluator.py
│   ├── test_retriever_bm25.py
│   ├── test_retriever_cross_enc.py
│   ├── test_classifier.py
│   └── test_pipeline.py
├── checkpoints/                   # gitignored
├── cache/                         # gitignored
├── outputs/                       # predictions JSONs
├── docs/superpowers/plans/        # this file
├── requirements.txt               # NEW
├── .gitignore                     # NEW or update
└── Group_073_COMP90042_Project_2026.ipynb
```

**ipynb migration mapping** (Phase 5):
| ipynb Section | Source modules |
|---|---|
| `1.DataSet Processing` | `config.py` + `utils.py` + `data_loader.py` + `preprocessing.py` |
| `2.Model Implementation` | `retriever_bm25.py` + `retriever_cross_enc.py` + `classifier.py` |
| `3.Testing and Evaluation` | `pipeline.py` + `evaluator.py` + scripts content |
| `OOP section` | (overflow if needed) |

---

## Commit Message Rules (MUST FOLLOW)

Every commit produced by this plan MUST conform to all of the following.
The pre-supplied subject lines below have already been audited against
these rules; do NOT rewrite them.

**Structural rules:**
- Subject line ≤ 50 characters (hard limit)
- Body wrapped at 72 characters
- Capitalize the description (the part after `: `)
- No trailing period in the subject
- Use imperative mood (`Add`, `Fix`, `Train`, not `Added` / `Fixes` / `Training`)
- Body explains the **what** and **why**, not the **how** (the diff shows how)

**Conventional Commits format:**
```
<type>(<scope>): <Description>
```
- `<type>` is lowercase and is one of: `feat`, `fix`, `docs`, `style`,
  `refactor`, `test`, `chore`
- `<scope>` is lowercase and identifies the area of code (e.g.
  `config`, `utils`, `retriever`, `classifier`, `scripts`, `pipeline`,
  `notebook`, `data`, `eval`, `preproc`)
- `<Description>` starts with a capital letter, imperative mood, no period

**Type cheatsheet:**
- `feat` — new feature for the user
- `fix` — bug fix
- `docs` — documentation only
- `style` — formatting / whitespace only, no code logic change
- `refactor` — code change that neither fixes a bug nor adds a feature
- `test` — adding / fixing tests
- `chore` — maintenance (deps, config, build)

**Example (good):**
```
feat(retriever): Add BERT cross-encoder reranker

The cross-encoder is trained on (claim, evidence) pairs with hard
negatives mined from BM25 top-50. We use bert-base-uncased + a 1-d
linear head over the [CLS] embedding so the same checkpoint format
works for inference.
```

**Common mistakes to avoid:**
- ❌ `feat(retriever): hard negative miner from BM25 top-N` (lowercase
  description, 52 chars > 50)
- ❌ `Fixed bug in classifier.` (past tense, trailing period, no type)
- ❌ `feat(retriever): Add cross-encoder model + dataset + train + rerank`
  (62 chars; over the 50-char limit — describe the *what*, not the
  internal file split)

---

# Phase 0: Project Bootstrap

### Task 0.1: Create directory skeleton + gitignore

> **Note:** Dependency management is already handled by the existing
> `environment.yml` (conda env `comp90042`, Python 3.11). Do NOT create a
> `requirements.txt` — it would fragment the source of truth. The existing
> `pyproject.toml` (ruff config, line-length 100, target py311) and
> `.pre-commit-config.yaml` (ruff + format on commit) also already exist;
> respect both. All Python code in this plan must satisfy ruff lint/format,
> otherwise `git commit` will fail via the pre-commit hook.

**Files:**
- Create: `.gitignore`
- Create: `src/__init__.py`, `scripts/__init__.py`, `tests/__init__.py`, `tests/conftest.py`
- Create: empty dirs `scripts/`, `checkpoints/`, `cache/`, `outputs/` (with `.gitkeep` for tracked dirs)

- [ ] **Step 1: Create `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
checkpoints/
cache/
outputs/*.json
data/evidence.json
*.ipynb_checkpoints
```

- [ ] **Step 2: Create empty package files**

```bash
mkdir -p src tests scripts checkpoints cache outputs
touch src/__init__.py tests/__init__.py scripts/__init__.py
touch outputs/.gitkeep checkpoints/.gitkeep cache/.gitkeep
```

- [ ] **Step 3: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures."""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir(repo_root: Path) -> Path:
    return repo_root / "data"
```

- [ ] **Step 4: Verify pytest collects nothing yet**

Run: `pytest --collect-only -q`
Expected: 0 tests collected, no errors.

- [ ] **Step 5: Install pre-commit hook (one-time)**

```bash
pre-commit install
```
Expected: `pre-commit installed at .git/hooks/pre-commit`. This makes
subsequent commits run ruff automatically.

- [ ] **Step 6: Commit**

```bash
git add .gitignore src/__init__.py tests/__init__.py scripts/__init__.py \
  tests/conftest.py outputs/.gitkeep checkpoints/.gitkeep cache/.gitkeep
git commit -m "chore: bootstrap project structure for plan B"
```
If pre-commit hook auto-formats anything (e.g. trailing whitespace), re-stage
and re-commit — the hook output will tell you exactly what changed.

---

# Phase 1: Foundation + BM25 Baseline (Milestone M1)

Goal: produce a working `outputs/dev-bm25-random.json` and run `eval.py` end-to-end. This locks down data IO, evaluator wrapper, and BM25 retrieval before any neural training.

### Task 1.1: Config module

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from src.config import Config


def test_default_config_has_required_paths():
    cfg = Config()
    assert cfg.seed == 42
    assert cfg.bm25_top_k == 200
    assert cfg.final_top_k == 4
    assert cfg.cross_encoder_model == "bert-base-uncased"
    assert cfg.classifier_model == "roberta-base"
    assert str(cfg.evidence_path).endswith("evidence.json")


def test_config_paths_are_resolved_to_repo():
    cfg = Config()
    assert cfg.repo_root.is_absolute()
    assert cfg.data_dir.name == "data"
```

- [ ] **Step 2: Run to verify it fails**

`pytest tests/test_config.py -v` → ImportError.

- [ ] **Step 3: Implement `src/config.py`**

```python
"""Centralized hyperparameters and paths."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    seed: int = 42

    # paths
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = field(init=False)
    evidence_path: Path = field(init=False)
    train_path: Path = field(init=False)
    dev_path: Path = field(init=False)
    test_path: Path = field(init=False)
    cache_dir: Path = field(init=False)
    ckpt_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    # retrieval
    bm25_top_k: int = 200
    final_top_k: int = 4
    hard_negatives_per_pos: int = 4

    # cross-encoder
    cross_encoder_model: str = "bert-base-uncased"
    ce_max_len: int = 256
    ce_lr: float = 2e-5
    ce_epochs: int = 2
    ce_batch_size: int = 32

    # classifier
    classifier_model: str = "roberta-base"
    cls_max_len: int = 384
    cls_lr: float = 2e-5
    cls_epochs: int = 3
    cls_batch_size: int = 16
    noise_mix_ratio: float = 0.5  # fraction of training samples using retrieved evidence

    label_names: tuple = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED")

    def __post_init__(self) -> None:
        self.data_dir = self.repo_root / "data"
        self.evidence_path = self.data_dir / "evidence.json"
        self.train_path = self.data_dir / "train-claims.json"
        self.dev_path = self.data_dir / "dev-claims.json"
        self.test_path = self.data_dir / "test-claims-unlabelled.json"
        self.cache_dir = self.repo_root / "cache"
        self.ckpt_dir = self.repo_root / "checkpoints"
        self.output_dir = self.repo_root / "outputs"
        for d in (self.cache_dir, self.ckpt_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def label2id(self) -> dict:
        return {name: i for i, name in enumerate(self.label_names)}

    @property
    def id2label(self) -> dict:
        return {i: name for i, name in enumerate(self.label_names)}
```

- [ ] **Step 4: Run to verify pass**

`pytest tests/test_config.py -v` → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat(config): centralized config dataclass with paths and hyperparams"
```

---

### Task 1.2: Utils (seed, IO, logger)

**Files:**
- Create: `src/utils.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_utils.py
import json
import random
from pathlib import Path

import numpy as np

from src.utils import set_seed, save_json, load_json, get_logger


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
```

- [ ] **Step 2: Run to verify failure**

`pytest tests/test_utils.py -v` → ImportError.

- [ ] **Step 3: Implement `src/utils.py`**

```python
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
    with path.open("w") as f:
        json.dump(obj, f, ensure_ascii=False)


def load_json(path: Path | str) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def get_logger(name: str = "factcheck", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s | %(message)s",
                                         datefmt="%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(level)
        log.propagate = False
    return log


@contextmanager
def timer(label: str, log=None):
    log = log or get_logger()
    t0 = time.perf_counter()
    yield
    log.info("%s took %.2fs", label, time.perf_counter() - t0)
```

- [ ] **Step 4: Run to verify pass**

`pytest tests/test_utils.py -v` → 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils.py tests/test_utils.py
git commit -m "feat(utils): seed/io/logger helpers"
```

---

### Task 1.3: Data loader

**Files:**
- Create: `src/data_loader.py`
- Test: `tests/test_data_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_loader.py
from src.config import Config
from src.data_loader import (
    load_claims, load_evidence_streaming, EVIDENCE_KEY_PATTERN,
    Claim,
)


def test_load_train_claims_returns_dict_of_claims():
    cfg = Config()
    claims = load_claims(cfg.train_path)
    assert len(claims) == 1228
    sample = next(iter(claims.values()))
    assert isinstance(sample, Claim)
    assert sample.claim_text
    assert sample.claim_label in cfg.label_names
    assert isinstance(sample.evidences, list) and len(sample.evidences) > 0


def test_load_test_claims_has_no_label():
    cfg = Config()
    claims = load_claims(cfg.test_path)
    assert len(claims) == 153
    sample = next(iter(claims.values()))
    assert sample.claim_label is None
    assert sample.evidences == []


def test_load_evidence_streaming_yields_all():
    cfg = Config()
    count = 0
    seen_first = None
    for eid, text in load_evidence_streaming(cfg.evidence_path):
        if count == 0:
            seen_first = eid
            assert EVIDENCE_KEY_PATTERN.match(eid)
            assert isinstance(text, str) and text
        count += 1
        if count >= 1000:
            break
    assert count == 1000
    assert seen_first.startswith("evidence-")
```

- [ ] **Step 2: Run to verify failure**

`pytest tests/test_data_loader.py -v` → ImportError.

- [ ] **Step 3: Implement `src/data_loader.py`**

```python
"""Load claim and evidence files into typed objects."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.utils import load_json

EVIDENCE_KEY_PATTERN = re.compile(r"^evidence-\d+$")


@dataclass
class Claim:
    claim_id: str
    claim_text: str
    claim_label: str | None = None
    evidences: list[str] = field(default_factory=list)


def load_claims(path: Path | str) -> dict[str, Claim]:
    """Load a {train,dev,test}-claims JSON into a dict of Claim objects."""
    raw = load_json(path)
    out: dict[str, Claim] = {}
    for cid, body in raw.items():
        out[cid] = Claim(
            claim_id=cid,
            claim_text=body["claim_text"],
            claim_label=body.get("claim_label"),
            evidences=list(body.get("evidences", [])),
        )
    return out


def load_evidence(path: Path | str) -> dict[str, str]:
    """Load full evidence corpus into memory (~1.2M items, ~1GB resident)."""
    return load_json(path)


def load_evidence_streaming(path: Path | str) -> Iterator[tuple[str, str]]:
    """Yield (evidence_id, text) one at a time. Useful for very low-memory hosts."""
    # JSON file is a single object so we still parse fully, but expose generator API
    # to keep call sites streaming-style. If RAM becomes a problem, swap for ijson.
    data = load_json(path)
    for k, v in data.items():
        yield k, v
```

- [ ] **Step 4: Run to verify pass**

`pytest tests/test_data_loader.py -v` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data_loader.py tests/test_data_loader.py
git commit -m "feat(data): typed Claim loader + evidence iterator"
```

---

### Task 1.4: Evaluator wrapper (sanity-check eval.py reproduction)

**Files:**
- Create: `src/evaluator.py`
- Test: `tests/test_evaluator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluator.py
from src.config import Config
from src.evaluator import evaluate_predictions


def test_baseline_eval_matches_official_script():
    """Our wrapper should reproduce the exact numbers from `python eval.py`
    on the bundled baseline (verified manually: F=0.3378, A=0.3506, HM=0.3441)."""
    cfg = Config()
    metrics = evaluate_predictions(
        predictions_path=cfg.data_dir / "dev-claims-baseline.json",
        groundtruth_path=cfg.dev_path,
    )
    assert abs(metrics["evidence_f"] - 0.3378) < 1e-3
    assert abs(metrics["claim_accuracy"] - 0.3506) < 1e-3
    assert abs(metrics["harmonic_mean"] - 0.3441) < 1e-3
    assert "per_class_accuracy" in metrics
    assert set(metrics["per_class_accuracy"].keys()) == set(cfg.label_names)
```

- [ ] **Step 2: Run to verify failure**

`pytest tests/test_evaluator.py -v` → ImportError.

- [ ] **Step 3: Implement `src/evaluator.py`**

```python
"""Wraps the official eval.py logic and adds per-class accuracy and confusion matrix."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.utils import load_json


def evaluate_predictions(predictions_path: Path | str,
                         groundtruth_path: Path | str) -> dict:
    """Return overall + per-class metrics as a flat dict."""
    preds = load_json(predictions_path)
    gold = load_json(groundtruth_path)

    f_scores: list[float] = []
    correct: list[float] = []
    per_class_total: Counter = Counter()
    per_class_correct: Counter = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)

    for cid, gold_inst in sorted(gold.items()):
        if cid not in preds:
            continue
        p = preds[cid]
        if "claim_label" not in p or "evidences" not in p:
            continue

        is_correct = float(p["claim_label"] == gold_inst["claim_label"])
        correct.append(is_correct)
        per_class_total[gold_inst["claim_label"]] += 1
        per_class_correct[gold_inst["claim_label"]] += int(is_correct)
        confusion[gold_inst["claim_label"]][p["claim_label"]] += 1

        retrieved = set(p.get("evidences") or [])
        gold_ev = gold_inst["evidences"]
        if retrieved and gold_ev:
            tp = sum(1 for g in gold_ev if g in retrieved)
            if tp > 0:
                recall = tp / len(gold_ev)
                precision = tp / len(retrieved)
                f = 2 * precision * recall / (precision + recall)
            else:
                f = 0.0
        else:
            f = 0.0
        f_scores.append(f)

    mean_f = float(np.mean(f_scores)) if f_scores else 0.0
    mean_acc = float(np.mean(correct)) if correct else 0.0
    hmean = (2 * mean_f * mean_acc / (mean_f + mean_acc)) if (mean_f + mean_acc) else 0.0

    per_class_acc = {
        lbl: per_class_correct[lbl] / per_class_total[lbl] if per_class_total[lbl] else 0.0
        for lbl in per_class_total
    }

    return {
        "evidence_f": mean_f,
        "claim_accuracy": mean_acc,
        "harmonic_mean": hmean,
        "per_class_accuracy": per_class_acc,
        "per_class_total": dict(per_class_total),
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }
```

- [ ] **Step 4: Cross-check with official script**

```bash
python eval.py --predictions data/dev-claims-baseline.json --groundtruth data/dev-claims.json
pytest tests/test_evaluator.py -v
```
Both numbers should match: F≈0.3378, A≈0.3506, HM≈0.3441.

- [ ] **Step 5: Commit**

```bash
git add src/evaluator.py tests/test_evaluator.py
git commit -m "feat(eval): wrapper reproducing eval.py + per-class metrics"
```

---

### Task 1.5: Preprocessing for BM25

**Files:**
- Create: `src/preprocessing.py`
- Test: `tests/test_preprocessing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocessing.py
from src.preprocessing import tokenize_for_bm25


def test_tokenize_lowercases_and_splits():
    out = tokenize_for_bm25("CO2 emissions Increase!")
    assert out == ["co2", "emissions", "increase"]


def test_tokenize_drops_pure_punct_and_short_noise():
    out = tokenize_for_bm25("--- a, the of climate.")
    # function words like 'a', 'the', 'of' are stopwords, '---' is punct.
    assert "climate" in out
    assert "a" not in out and "the" not in out and "of" not in out


def test_tokenize_handles_empty():
    assert tokenize_for_bm25("") == []
```

- [ ] **Step 2: Run to verify failure**

`pytest tests/test_preprocessing.py -v` → ImportError.

- [ ] **Step 3: Implement `src/preprocessing.py`**

```python
"""Lightweight BM25-friendly tokenizer (lecture 2 preprocessing concepts)."""
from __future__ import annotations

import re

# minimal stopword list (subset of NLTK english) – kept here to avoid runtime download
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "of", "on",
    "or", "she", "that", "the", "their", "they", "this", "to", "was", "we", "were",
    "will", "with", "you", "your",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase, regex-tokenize, drop stopwords/empties."""
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower())
            if t not in _STOPWORDS and len(t) > 1 or t.isdigit()]
```

- [ ] **Step 4: Run to verify pass**

`pytest tests/test_preprocessing.py -v` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preproc): bm25 tokenizer with stopword removal"
```

---

### Task 1.6: BM25 retriever (build index + query)

**Files:**
- Create: `src/retriever_bm25.py`
- Test: `tests/test_retriever_bm25.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retriever_bm25.py
import pickle
from pathlib import Path

import pytest

from src.retriever_bm25 import BM25Retriever, build_bm25_index


@pytest.fixture(scope="module")
def toy_corpus() -> dict[str, str]:
    return {
        "evidence-0": "CO2 emissions cause global warming",
        "evidence-1": "Coral reefs depend on stable ocean temperatures",
        "evidence-2": "Wind turbines reduce greenhouse gas",
        "evidence-3": "Boston is a city in Massachusetts",
    }


def test_build_bm25_index_persists_and_reloads(tmp_path: Path, toy_corpus):
    cache = tmp_path / "bm25.pkl"
    build_bm25_index(toy_corpus, cache_path=cache)
    assert cache.exists()
    r = BM25Retriever.from_cache(cache)
    assert len(r.evidence_ids) == 4


def test_bm25_retrieves_relevant_first(tmp_path, toy_corpus):
    cache = tmp_path / "bm25.pkl"
    build_bm25_index(toy_corpus, cache_path=cache)
    r = BM25Retriever.from_cache(cache)
    hits = r.search("greenhouse gas emissions cause warming", top_k=2)
    assert hits[0][0] in {"evidence-0", "evidence-2"}
    assert all(score > 0 for _, score in hits)


def test_bm25_returns_at_most_top_k(tmp_path, toy_corpus):
    cache = tmp_path / "bm25.pkl"
    build_bm25_index(toy_corpus, cache_path=cache)
    r = BM25Retriever.from_cache(cache)
    assert len(r.search("warming", top_k=2)) == 2
```

- [ ] **Step 2: Run to verify failure**

`pytest tests/test_retriever_bm25.py -v` → ImportError.

- [ ] **Step 3: Implement `src/retriever_bm25.py`**

```python
"""BM25 first-stage retriever over the evidence corpus."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.preprocessing import tokenize_for_bm25
from src.utils import get_logger

log = get_logger("bm25")


def build_bm25_index(evidence: dict[str, str], cache_path: Path | str) -> None:
    """Tokenize all evidences and persist (BM25Okapi, ids) to disk."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(evidence.keys())
    log.info("Tokenizing %d evidences for BM25 ...", len(ids))
    corpus_tokens = [tokenize_for_bm25(evidence[i]) for i in tqdm(ids, desc="tokenize")]
    log.info("Building BM25Okapi index ...")
    bm25 = BM25Okapi(corpus_tokens)
    with cache_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)
    log.info("Saved BM25 index → %s (%.1f MB)", cache_path,
             cache_path.stat().st_size / 1e6)


@dataclass
class BM25Retriever:
    bm25: BM25Okapi
    evidence_ids: list[str]

    @classmethod
    def from_cache(cls, cache_path: Path | str) -> "BM25Retriever":
        with Path(cache_path).open("rb") as f:
            blob = pickle.load(f)
        return cls(bm25=blob["bm25"], evidence_ids=blob["ids"])

    def search(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.evidence_ids[i], float(scores[i])) for i in top_idx]

    def search_batch(self, queries: Iterable[str], top_k: int = 200) -> list[list[tuple[str, float]]]:
        return [self.search(q, top_k=top_k) for q in queries]
```

- [ ] **Step 4: Run to verify pass**

`pytest tests/test_retriever_bm25.py -v` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/retriever_bm25.py tests/test_retriever_bm25.py
git commit -m "feat(retriever): bm25 first-stage retriever"
```

---

### Task 1.7: Build BM25 index over real corpus (script + run)

**Files:**
- Create: `scripts/build_bm25.py`

- [ ] **Step 1: Implement `scripts/build_bm25.py`**

```python
"""Build BM25 index from data/evidence.json into cache/bm25.pkl. Idempotent."""
from src.config import Config
from src.data_loader import load_evidence
from src.retriever_bm25 import build_bm25_index
from src.utils import get_logger, timer, set_seed


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)
    log = get_logger("bm25-build")
    out = cfg.cache_dir / "bm25.pkl"
    if out.exists():
        log.info("BM25 index already exists at %s; skipping. Delete to rebuild.", out)
        return
    log.info("Loading evidence ...")
    with timer("load_evidence", log):
        ev = load_evidence(cfg.evidence_path)
    log.info("Loaded %d evidence passages", len(ev))
    with timer("build_index", log):
        build_bm25_index(ev, cache_path=out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run + sanity check**

```bash
python -m scripts.build_bm25
ls -lh cache/bm25.pkl
```
Expected: ~5–15 minutes wall clock; pickle file ~300–800 MB.

- [ ] **Step 3: Smoke-test loading**

```bash
python -c "from src.retriever_bm25 import BM25Retriever; from src.config import Config; r=BM25Retriever.from_cache(Config().cache_dir/'bm25.pkl'); print(len(r.evidence_ids)); print(r.search('CO2 doubling climate sensitivity 1C', top_k=5))"
```
Expected: 1208827, plus 5 tuples with positive scores.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_bm25.py
git commit -m "feat(scripts): build BM25 index from evidence corpus"
```

---

### Task 1.8: BM25 baseline submission + measure F-score on dev

**Files:**
- Create: `scripts/run_inference.py` (initial BM25-only version; will grow in Phase 4)

- [ ] **Step 1: Implement initial inference script**

```python
"""Run BM25-only retrieval + (initially) random-label classification.
This is the M1 sanity baseline to confirm the pipeline produces a valid JSON."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.config import Config
from src.data_loader import load_claims
from src.evaluator import evaluate_predictions
from src.retriever_bm25 import BM25Retriever
from src.utils import get_logger, save_json, set_seed, timer

log = get_logger("infer")


def run_baseline(claims_path: Path, output_path: Path, top_k: int) -> dict:
    cfg = Config()
    set_seed(cfg.seed)
    bm25 = BM25Retriever.from_cache(cfg.cache_dir / "bm25.pkl")
    claims = load_claims(claims_path)

    preds: dict[str, dict] = {}
    with timer(f"BM25 retrieval x{len(claims)}", log):
        for cid, claim in claims.items():
            hits = bm25.search(claim.claim_text, top_k=top_k)
            preds[cid] = {
                "claim_text": claim.claim_text,
                "claim_label": random.choice(cfg.label_names),
                "evidences": [eid for eid, _ in hits],
            }
    save_json(preds, output_path)
    log.info("Saved %d predictions → %s", len(preds), output_path)
    return preds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--top-k", type=int, default=Config().final_top_k)
    p.add_argument("--mode", choices=["bm25-random"], default="bm25-random",
                   help="Phase 4 will add 'full-pipeline'.")
    args = p.parse_args()

    cfg = Config()
    claims_path = cfg.dev_path if args.split == "dev" else cfg.test_path
    output_path = cfg.output_dir / f"{args.split}-{args.mode}-k{args.top_k}.json"
    run_baseline(claims_path, output_path, args.top_k)

    if args.split == "dev":
        m = evaluate_predictions(output_path, cfg.dev_path)
        log.info("F=%.4f  A=%.4f  HM=%.4f", m["evidence_f"], m["claim_accuracy"],
                 m["harmonic_mean"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on dev set**

```bash
python -m scripts.run_inference --split dev --top-k 4
```
Expected: F > 0 (BM25 should retrieve at least some gold evidences); A ≈ 0.25 (random baseline over 4 classes).

- [ ] **Step 3: Cross-check with official eval**

```bash
python eval.py --predictions outputs/dev-bm25-random-k4.json --groundtruth data/dev-claims.json
```
Expected: same numbers as our wrapper.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_inference.py
git commit -m "feat(scripts): bm25 baseline inference + dev evaluation"
```

---

### Task 1.9: M1 milestone summary commit

- [ ] **Step 1: Record M1 results**

Create `docs/superpowers/plans/m1-baseline-results.md`:

```markdown
# M1 BM25 baseline results (top-K sweep)

| top-K | F | A (random) | HM |
|---|---|---|---|
| 3 | ? | ? | ? |
| 4 | ? | ? | ? |
| 5 | ? | ? | ? |
| 6 | ? | ? | ? |
```

Run sweep:

```bash
for k in 3 4 5 6; do python -m scripts.run_inference --split dev --top-k $k; done
```

Fill in the table with measured values.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/m1-baseline-results.md
git commit -m "chore(m1): record bm25 baseline sweep results"
```

---

# Phase 2: Cross-Encoder Re-ranker (Milestone M2)

Goal: train BERT cross-encoder on (claim, evidence) pairs with hard negatives. Use it to re-rank BM25 top-200 → top-K.

### Task 2.1: Hard negative miner

**Files:**
- Create: `src/hard_negatives.py`
- Test: `tests/test_hard_negatives.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_hard_negatives.py
from src.hard_negatives import build_training_pairs


def test_build_training_pairs_has_pos_and_negs():
    claims = {
        "claim-1": {"claim_text": "Q1", "evidences": ["e-1", "e-2"]},
    }
    bm25_results = {
        "claim-1": [("e-1", 5.0), ("e-9", 4.5), ("e-8", 4.0),
                    ("e-7", 3.5), ("e-2", 3.0), ("e-6", 2.5)],
    }
    pairs = build_training_pairs(claims, bm25_results, n_neg=2, seed=0)
    labels = [p["label"] for p in pairs]
    assert labels.count(1) == 2  # two gold evidences → two positives
    assert labels.count(0) == 4  # 2 positives × 2 hard negs each
    neg_eids = {p["evidence_id"] for p in pairs if p["label"] == 0}
    assert neg_eids.issubset({"e-9", "e-8", "e-7", "e-6"})
    assert "e-1" not in neg_eids and "e-2" not in neg_eids


def test_no_pairs_when_bm25_only_returns_gold():
    claims = {"claim-1": {"claim_text": "Q", "evidences": ["e-1"]}}
    bm25_results = {"claim-1": [("e-1", 5.0)]}
    pairs = build_training_pairs(claims, bm25_results, n_neg=2, seed=0)
    # only one positive, no negatives available
    assert sum(p["label"] for p in pairs) == 1
    assert len(pairs) == 1
```

- [ ] **Step 2: Run to verify failure**

`pytest tests/test_hard_negatives.py -v` → ImportError.

- [ ] **Step 3: Implement `src/hard_negatives.py`**

```python
"""Mine hard negatives for cross-encoder training: BM25 top-N \\ gold."""
from __future__ import annotations

import random
from typing import Sequence


def build_training_pairs(
    claims: dict,
    bm25_results: dict[str, Sequence[tuple[str, float]]],
    n_neg: int = 4,
    seed: int = 42,
) -> list[dict]:
    """Return list of dicts {claim_id, claim_text, evidence_id, label}."""
    rng = random.Random(seed)
    pairs: list[dict] = []
    for cid, claim in claims.items():
        ctext = claim["claim_text"] if isinstance(claim, dict) else claim.claim_text
        gold_list = claim["evidences"] if isinstance(claim, dict) else claim.evidences
        gold_set = set(gold_list)
        bm_hits = bm25_results.get(cid, [])
        candidates = [eid for eid, _ in bm_hits if eid not in gold_set]
        for ge in gold_list:
            pairs.append({"claim_id": cid, "claim_text": ctext,
                          "evidence_id": ge, "label": 1})
            if not candidates:
                continue
            sampled = rng.sample(candidates, k=min(n_neg, len(candidates)))
            for ne in sampled:
                pairs.append({"claim_id": cid, "claim_text": ctext,
                              "evidence_id": ne, "label": 0})
    return pairs
```

- [ ] **Step 4: Run to verify pass**

`pytest tests/test_hard_negatives.py -v` → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/hard_negatives.py tests/test_hard_negatives.py
git commit -m "feat(retriever): Mine hard negatives via BM25"
```

---

### Task 2.2: Cross-encoder dataset + model + train + rerank module

**Files:**
- Create: `src/retriever_cross_enc.py`
- Test: `tests/test_retriever_cross_enc.py`

- [ ] **Step 1: Write failing tests (smoke + I/O level)**

```python
# tests/test_retriever_cross_enc.py
import pytest
import torch

from src.retriever_cross_enc import (
    CrossEncoderDataset, build_cross_encoder, rerank,
)


@pytest.fixture(scope="module")
def tokenizer_and_model():
    return build_cross_encoder("prajjwal1/bert-tiny")  # tiny for tests


def test_dataset_yields_tensors(tokenizer_and_model):
    tok, _ = tokenizer_and_model
    pairs = [{"claim_text": "claim", "evidence_text": "ev", "label": 1}]
    ds = CrossEncoderDataset(pairs, tok, max_len=32)
    item = ds[0]
    assert "input_ids" in item and "attention_mask" in item
    assert item["labels"].dtype == torch.float32
    assert item["input_ids"].shape[0] <= 32


def test_rerank_orders_by_score(tokenizer_and_model):
    tok, model = tokenizer_and_model
    candidates = [("e-1", "x"), ("e-2", "y"), ("e-3", "z")]
    out = rerank(model, tok, claim_text="q", candidates=candidates,
                 evidence_lookup={"e-1": "x", "e-2": "y", "e-3": "z"},
                 top_k=2, batch_size=2, device="cpu")
    assert len(out) == 2
    assert all(isinstance(s, float) for _, s in out)
    assert out[0][1] >= out[1][1]
```

- [ ] **Step 2: Run failure**

`pytest tests/test_retriever_cross_enc.py -v` → ImportError.

- [ ] **Step 3: Implement `src/retriever_cross_enc.py`**

```python
"""BERT cross-encoder for relevance re-ranking (Lecture 11 BERT [CLS] pattern)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.utils import get_logger

log = get_logger("cross-enc")


def build_cross_encoder(model_name: str):
    """Returns (tokenizer, model). Model = encoder + linear head producing 1 logit."""
    tok = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    model = CrossEncoderHead(encoder)
    return tok, model


class CrossEncoderHead(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [CLS] vector (Lecture 11)
        return self.classifier(self.dropout(cls)).squeeze(-1)  # logits


class CrossEncoderDataset(Dataset):
    """Dataset of (claim, evidence, label) → tokenized pair."""

    def __init__(self, pairs: Sequence[dict], tokenizer, max_len: int = 256,
                 evidence_lookup: dict[str, str] | None = None):
        self.pairs = list(pairs)
        self.tok = tokenizer
        self.max_len = max_len
        self.lookup = evidence_lookup or {}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        p = self.pairs[idx]
        ev = p.get("evidence_text") or self.lookup[p["evidence_id"]]
        enc = self.tok(p["claim_text"], ev, truncation=True,
                       max_length=self.max_len, padding="max_length",
                       return_tensors="pt")
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(float(p["label"]), dtype=torch.float32),
        }


def train_cross_encoder(model, tokenizer, train_pairs: Sequence[dict],
                        evidence_lookup: dict[str, str],
                        max_len: int, batch_size: int, lr: float, epochs: int,
                        device: str, save_path: Path | str) -> None:
    ds = CrossEncoderDataset(train_pairs, tokenizer, max_len=max_len,
                             evidence_lookup=evidence_lookup)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        running = 0.0
        for batch in tqdm(loader, desc=f"CE epoch {ep+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"])
            loss = loss_fn(logits, batch["labels"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        log.info("epoch %d mean_loss=%.4f", ep + 1, running / len(loader))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    log.info("Saved cross-encoder ckpt → %s", save_path)


def load_cross_encoder(model_name: str, ckpt_path: Path | str, device: str = "cpu"):
    tok, model = build_cross_encoder(model_name)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return tok, model.to(device).eval()


@torch.no_grad()
def rerank(model, tokenizer, claim_text: str,
           candidates: Sequence[tuple[str, float]],
           evidence_lookup: dict[str, str],
           top_k: int, batch_size: int = 64, device: str = "cpu",
           max_len: int = 256) -> list[tuple[str, float]]:
    """Return top-k candidates ordered by cross-encoder score."""
    if not candidates:
        return []
    eids = [eid for eid, _ in candidates]
    texts = [evidence_lookup[e] for e in eids]
    scores: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer([claim_text] * len(batch_texts), batch_texts,
                        truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt").to(device)
        logits = model(**enc)
        scores.extend(torch.sigmoid(logits).cpu().tolist())
    ranked = sorted(zip(eids, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

- [ ] **Step 4: Run tests (CPU + tiny BERT)**

`pytest tests/test_retriever_cross_enc.py -v` → 2 passed (slow first time as `prajjwal1/bert-tiny` downloads).

- [ ] **Step 5: Commit**

```bash
git add src/retriever_cross_enc.py tests/test_retriever_cross_enc.py
git commit -m "feat(retriever): Add cross-encoder reranker"
```

---

### Task 2.3: Train cross-encoder on real data (script)

**Files:**
- Create: `scripts/train_cross_encoder.py`

- [ ] **Step 1: Implement script**

```python
"""Train BERT cross-encoder on (claim, gold/hard-neg evidence) pairs."""
from __future__ import annotations

import torch

from src.config import Config
from src.data_loader import load_claims, load_evidence
from src.hard_negatives import build_training_pairs
from src.retriever_bm25 import BM25Retriever
from src.retriever_cross_enc import build_cross_encoder, train_cross_encoder
from src.utils import get_logger, set_seed, timer, save_json, load_json

log = get_logger("ce-train")


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_claims = load_claims(cfg.train_path)
    bm25 = BM25Retriever.from_cache(cfg.cache_dir / "bm25.pkl")

    bm25_cache = cfg.cache_dir / "bm25_train_top50.json"
    if bm25_cache.exists():
        log.info("Loading cached BM25 train top-50 ...")
        bm25_results = load_json(bm25_cache)
    else:
        with timer("BM25 search train", log):
            bm25_results = {cid: bm25.search(c.claim_text, top_k=50)
                            for cid, c in train_claims.items()}
        save_json(bm25_results, bm25_cache)

    pairs = build_training_pairs(
        {cid: {"claim_text": c.claim_text, "evidences": c.evidences}
         for cid, c in train_claims.items()},
        bm25_results,
        n_neg=cfg.hard_negatives_per_pos,
        seed=cfg.seed,
    )
    log.info("Built %d training pairs (pos+neg)", len(pairs))

    log.info("Loading evidence corpus into memory ...")
    evidence = load_evidence(cfg.evidence_path)

    tok, model = build_cross_encoder(cfg.cross_encoder_model)
    train_cross_encoder(
        model, tok, pairs, evidence,
        max_len=cfg.ce_max_len,
        batch_size=cfg.ce_batch_size,
        lr=cfg.ce_lr,
        epochs=cfg.ce_epochs,
        device=device,
        save_path=cfg.ckpt_dir / "cross_encoder.pt",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run training (Colab T4 expected)**

```bash
python -m scripts.train_cross_encoder
```
Expected: ~30–60 min/epoch × 2; final mean_loss < 0.3.

- [ ] **Step 3: Sanity check checkpoint loads**

```bash
python -c "from src.retriever_cross_enc import load_cross_encoder; from src.config import Config; c=Config(); load_cross_encoder(c.cross_encoder_model, c.ckpt_dir/'cross_encoder.pt'); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_cross_encoder.py
git commit -m "feat(scripts): Train cross-encoder reranker"
```

---

### Task 2.4: Cache BM25 + cross-encoder retrieval results for dev/train/test

**Files:**
- Modify: `scripts/run_inference.py` (add `--mode retriever-only`)

- [ ] **Step 1: Add retrieval-only mode**

Add this function to `scripts/run_inference.py`:

```python
def run_retriever_only(claims_path: Path, output_path: Path, top_k: int) -> dict:
    """BM25 → cross-encoder rerank → top-K. Random label (placeholder)."""
    cfg = Config()
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bm25 = BM25Retriever.from_cache(cfg.cache_dir / "bm25.pkl")
    tok, model = load_cross_encoder(cfg.cross_encoder_model,
                                    cfg.ckpt_dir / "cross_encoder.pt", device=device)
    evidence = load_evidence(cfg.evidence_path)
    claims = load_claims(claims_path)

    preds: dict[str, dict] = {}
    with timer(f"Retriever pipeline x{len(claims)}", log):
        for cid, claim in claims.items():
            cand = bm25.search(claim.claim_text, top_k=cfg.bm25_top_k)
            ranked = rerank(model, tok, claim.claim_text, cand, evidence,
                            top_k=top_k, batch_size=64, device=device,
                            max_len=cfg.ce_max_len)
            preds[cid] = {
                "claim_text": claim.claim_text,
                "claim_label": random.choice(cfg.label_names),
                "evidences": [eid for eid, _ in ranked],
            }
    save_json(preds, output_path)
    return preds
```

Add CLI choice `retriever-only` and dispatch.

Add imports at top of script:

```python
import torch
from src.data_loader import load_evidence
from src.retriever_cross_enc import load_cross_encoder, rerank
```

- [ ] **Step 2: Run on dev**

```bash
python -m scripts.run_inference --split dev --mode retriever-only --top-k 4
```

- [ ] **Step 3: Verify F-score improves over BM25 only**

Expect F to go up (e.g. 0.10 → 0.18+). Append result to `m1-baseline-results.md`.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_inference.py docs/superpowers/plans/m1-baseline-results.md
git commit -m "feat(scripts): Add retriever-only inference mode"
```

---

# Phase 3: Noise-Aware RoBERTa Classifier (Milestone M3)

Goal: 4-way classifier trained on a 50/50 mix of (claim, gold-evidences) and (claim, retrieved-evidences). Run two ablations: gold-only vs noise-aware.

### Task 3.1: Classifier dataset (gold + noise modes)

**Files:**
- Create: `src/classifier.py` (full module: dataset + model + train + predict)
- Test: `tests/test_classifier.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_classifier.py
import pytest
import torch
from transformers import AutoTokenizer

from src.classifier import (
    ClassifierDataset, ClaimClassifier, predict_label, build_classifier,
)


@pytest.fixture(scope="module")
def tok_and_model():
    return build_classifier("prajjwal1/bert-tiny", num_labels=4)


def test_dataset_concatenates_evidences(tok_and_model):
    tok, _ = tok_and_model
    examples = [
        {"claim_text": "c1", "evidences_text": ["e1", "e2"], "label_id": 0},
    ]
    ds = ClassifierDataset(examples, tok, max_len=64)
    item = ds[0]
    assert item["input_ids"].shape[0] <= 64
    assert int(item["labels"]) == 0


def test_predict_label_returns_valid_id(tok_and_model):
    tok, model = tok_and_model
    out = predict_label(model, tok, claim_text="c", evidences_text=["e1"],
                        device="cpu", max_len=64)
    assert isinstance(out, dict)
    assert out["label_id"] in (0, 1, 2, 3)
    assert 0 <= out["confidence"] <= 1
```

- [ ] **Step 2: Run failure**

`pytest tests/test_classifier.py -v` → ImportError.

- [ ] **Step 3: Implement `src/classifier.py`**

```python
"""4-way claim classifier (RoBERTa) with noise-aware training support."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (AutoModel, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src.utils import get_logger

log = get_logger("classifier")


def build_classifier(model_name: str, num_labels: int = 4):
    tok = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    model = ClaimClassifier(encoder, num_labels=num_labels)
    return tok, model


class ClaimClassifier(nn.Module):
    def __init__(self, encoder, num_labels: int = 4):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [CLS] / <s>
        return self.classifier(self.dropout(cls))


class ClassifierDataset(Dataset):
    """Each example: {claim_text, evidences_text: list[str], label_id}."""

    def __init__(self, examples: Sequence[dict], tokenizer, max_len: int = 384):
        self.examples = list(examples)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        joined_evidence = " ".join(f"[{i+1}] {t}" for i, t in enumerate(ex["evidences_text"]))
        enc = self.tok(ex["claim_text"], joined_evidence, truncation=True,
                       max_length=self.max_len, padding="max_length",
                       return_tensors="pt")
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(ex["label_id"], dtype=torch.long),
        }


def build_examples(claims: dict, evidence_lookup: dict[str, str],
                   label2id: dict[str, int],
                   retrieved_topk: dict[str, list[str]] | None = None,
                   noise_mix_ratio: float = 0.0,
                   seed: int = 42) -> list[dict]:
    """If retrieved_topk + noise_mix_ratio > 0, swap evidence for retrieved
    on a fraction of (deterministic) claims to teach the model to handle
    noisy evidence at train time."""
    rng = random.Random(seed)
    out: list[dict] = []
    cids = list(claims.keys())
    use_retrieved = set()
    if retrieved_topk and noise_mix_ratio > 0:
        n_swap = int(len(cids) * noise_mix_ratio)
        use_retrieved = set(rng.sample(cids, k=n_swap))

    for cid in cids:
        c = claims[cid]
        text = c.claim_text if hasattr(c, "claim_text") else c["claim_text"]
        label = c.claim_label if hasattr(c, "claim_label") else c.get("claim_label")
        gold = c.evidences if hasattr(c, "evidences") else c["evidences"]
        if cid in use_retrieved and cid in retrieved_topk:
            ev_ids = retrieved_topk[cid]
        else:
            ev_ids = gold
        ev_texts = [evidence_lookup[e] for e in ev_ids if e in evidence_lookup]
        if not ev_texts:
            continue  # cannot train without any evidence text
        out.append({
            "claim_id": cid,
            "claim_text": text,
            "evidences_text": ev_texts,
            "label_id": label2id[label] if label else -1,
        })
    return out


def class_weights_from_labels(label_ids: list[int], num_classes: int) -> torch.Tensor:
    counts = [max(1, label_ids.count(i)) for i in range(num_classes)]
    n = sum(counts)
    weights = [n / (num_classes * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def train_classifier(model, tokenizer, examples: Sequence[dict],
                     max_len: int, batch_size: int, lr: float, epochs: int,
                     device: str, save_path: Path | str,
                     class_weights: torch.Tensor | None = None) -> None:
    ds = ClassifierDataset(examples, tokenizer, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for ep in range(epochs):
        running = 0.0
        for batch in tqdm(loader, desc=f"CLS epoch {ep+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"])
            loss = loss_fn(logits, batch["labels"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        log.info("epoch %d mean_loss=%.4f", ep + 1, running / len(loader))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    log.info("Saved classifier ckpt → %s", save_path)


def load_classifier(model_name: str, ckpt_path: Path | str, num_labels: int = 4,
                    device: str = "cpu"):
    tok, model = build_classifier(model_name, num_labels=num_labels)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return tok, model.to(device).eval()


@torch.no_grad()
def predict_label(model, tokenizer, claim_text: str, evidences_text: list[str],
                  device: str = "cpu", max_len: int = 384) -> dict:
    joined = " ".join(f"[{i+1}] {t}" for i, t in enumerate(evidences_text))
    enc = tokenizer(claim_text, joined, truncation=True, max_length=max_len,
                   padding=True, return_tensors="pt").to(device)
    logits = model(**enc)
    probs = torch.softmax(logits, dim=-1)[0].cpu()
    label_id = int(probs.argmax())
    return {"label_id": label_id, "confidence": float(probs[label_id]),
            "probs": probs.tolist()}
```

- [ ] **Step 4: Run tests**

`pytest tests/test_classifier.py -v` → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/classifier.py tests/test_classifier.py
git commit -m "feat(classifier): Add RoBERTa noise-aware model"
```

---

### Task 3.2: Train classifier (gold-only ablation)

**Files:**
- Create: `scripts/train_classifier.py`

- [ ] **Step 1: Implement script**

```python
"""Train RoBERTa claim classifier. Supports two modes (ablation):
  - gold-only: noise_mix_ratio = 0
  - noise-aware: noise_mix_ratio = cfg.noise_mix_ratio (default 0.5)
"""
from __future__ import annotations

import argparse

import torch

from src.classifier import (build_classifier, build_examples,
                            class_weights_from_labels, train_classifier)
from src.config import Config
from src.data_loader import load_claims, load_evidence
from src.utils import get_logger, load_json, save_json, set_seed, timer

log = get_logger("cls-train")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gold-only", "noise-aware"],
                   default="noise-aware")
    args = p.parse_args()

    cfg = Config()
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_claims = load_claims(cfg.train_path)
    evidence = load_evidence(cfg.evidence_path)

    retrieved = None
    noise = 0.0
    if args.mode == "noise-aware":
        cache = cfg.cache_dir / "retrieved_train_topk.json"
        if not cache.exists():
            log.error("Missing %s. Run scripts.run_inference --split train "
                      "--mode retriever-only first (requires extending CLI).",
                      cache)
            raise SystemExit(1)
        retrieved = load_json(cache)
        noise = cfg.noise_mix_ratio

    examples = build_examples(
        train_claims, evidence, cfg.label2id,
        retrieved_topk=retrieved, noise_mix_ratio=noise, seed=cfg.seed,
    )
    log.info("Built %d training examples (mode=%s)", len(examples), args.mode)

    label_ids = [e["label_id"] for e in examples]
    cw = class_weights_from_labels(label_ids, num_classes=len(cfg.label_names))
    log.info("Class weights: %s", cw.tolist())

    tok, model = build_classifier(cfg.classifier_model, num_labels=len(cfg.label_names))
    train_classifier(
        model, tok, examples,
        max_len=cfg.cls_max_len,
        batch_size=cfg.cls_batch_size,
        lr=cfg.cls_lr,
        epochs=cfg.cls_epochs,
        device=device,
        save_path=cfg.ckpt_dir / f"classifier_{args.mode}.pt",
        class_weights=cw,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run gold-only first**

```bash
python -m scripts.train_classifier --mode gold-only
```
Expected: 3 epochs, mean_loss decreasing.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_classifier.py
git commit -m "feat(scripts): Train classifier (gold/noise modes)"
```

---

### Task 3.3: Cache retriever results on TRAIN split (needed for noise-aware)

**Files:**
- Modify: `scripts/run_inference.py` (allow `--split train`)

- [ ] **Step 1: Extend CLI**

In `run_inference.py`:

```python
p.add_argument("--split", choices=["train", "dev", "test"], default="dev")
# in main:
splits = {"train": cfg.train_path, "dev": cfg.dev_path, "test": cfg.test_path}
claims_path = splits[args.split]
```

Also after retriever-only run, save a stripped version:

```python
if args.split == "train" and args.mode == "retriever-only":
    stripped = {cid: pred["evidences"] for cid, pred in preds.items()}
    save_json(stripped, cfg.cache_dir / "retrieved_train_topk.json")
```

- [ ] **Step 2: Run on train**

```bash
python -m scripts.run_inference --split train --mode retriever-only --top-k 4
```
Expected: ~10–20 min on Colab T4.

- [ ] **Step 3: Verify cache exists**

```bash
ls -lh cache/retrieved_train_topk.json
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_inference.py
git commit -m "feat(scripts): Cache retrieval on train split"
```

---

### Task 3.4: Train classifier (noise-aware ablation)

- [ ] **Step 1: Run training**

```bash
python -m scripts.train_classifier --mode noise-aware
```

- [ ] **Step 2: Smoke test prediction**

```bash
python -c "
from src.classifier import load_classifier, predict_label
from src.config import Config
c = Config()
tok, m = load_classifier(c.classifier_model, c.ckpt_dir/'classifier_noise-aware.pt')
print(predict_label(m, tok, 'CO2 doubling causes 1°C warming', ['IPCC says 1.5–4.5°C']))
"
```

- [ ] **Step 3: Commit**

```bash
# nothing new in git, just a tag
git tag m3-classifier-trained
```

---

# Phase 4: End-to-End Pipeline + Analysis (Milestone M4)

### Task 4.1: Pipeline orchestrator

**Files:**
- Create: `src/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test (using stubs)**

```python
# tests/test_pipeline.py
from unittest.mock import MagicMock

from src.pipeline import FactCheckingPipeline


def test_pipeline_returns_top_k_and_label():
    bm25 = MagicMock()
    bm25.search.return_value = [("e-1", 5.0), ("e-2", 4.0), ("e-3", 3.0)]
    ce_tok = MagicMock(); ce_model = MagicMock()
    cls_tok = MagicMock(); cls_model = MagicMock()

    def fake_rerank(model, tok, claim_text, candidates, evidence_lookup,
                    top_k, **kw):
        return [(c[0], c[1]) for c in candidates[:top_k]]

    def fake_predict(model, tok, claim_text, evidences_text, **kw):
        return {"label_id": 1, "confidence": 0.9, "probs": [0.05, 0.9, 0.03, 0.02]}

    p = FactCheckingPipeline(
        bm25=bm25, ce_tok=ce_tok, ce_model=ce_model,
        cls_tok=cls_tok, cls_model=cls_model,
        evidence_lookup={"e-1": "x", "e-2": "y", "e-3": "z"},
        id2label={0: "S", 1: "R", 2: "N", 3: "D"},
        bm25_top_k=3, final_top_k=2,
        rerank_fn=fake_rerank, predict_fn=fake_predict,
    )
    out = p.predict("does CO2 warm?")
    assert out["claim_label"] == "R"
    assert out["evidences"] == ["e-1", "e-2"]
    bm25.search.assert_called_once()
```

- [ ] **Step 2: Run failure**

`pytest tests/test_pipeline.py -v` → ImportError.

- [ ] **Step 3: Implement `src/pipeline.py`**

```python
"""End-to-end orchestrator: claim → BM25 → cross-enc rerank → classifier label."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from src.classifier import predict_label
from src.retriever_cross_enc import rerank


@dataclass
class FactCheckingPipeline:
    bm25: object
    ce_tok: object
    ce_model: object
    cls_tok: object
    cls_model: object
    evidence_lookup: dict[str, str]
    id2label: dict[int, str]
    bm25_top_k: int = 200
    final_top_k: int = 4
    device: str = "cpu"
    ce_max_len: int = 256
    cls_max_len: int = 384
    rerank_fn: Callable = field(default_factory=lambda: rerank)
    predict_fn: Callable = field(default_factory=lambda: predict_label)

    def predict(self, claim_text: str) -> dict:
        cands = self.bm25.search(claim_text, top_k=self.bm25_top_k)
        ranked = self.rerank_fn(self.ce_model, self.ce_tok, claim_text, cands,
                                self.evidence_lookup, top_k=self.final_top_k,
                                device=self.device, max_len=self.ce_max_len)
        evidence_ids = [eid for eid, _ in ranked]
        evidence_texts = [self.evidence_lookup[e] for e in evidence_ids]
        cls_out = self.predict_fn(self.cls_model, self.cls_tok,
                                  claim_text=claim_text,
                                  evidences_text=evidence_texts,
                                  device=self.device, max_len=self.cls_max_len)
        return {
            "claim_text": claim_text,
            "claim_label": self.id2label[cls_out["label_id"]],
            "evidences": evidence_ids,
            "confidence": cls_out["confidence"],
        }
```

- [ ] **Step 4: Run test**

`pytest tests/test_pipeline.py -v` → 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): Add end-to-end orchestrator"
```

---

### Task 4.2: Full inference + Oracle modes in run_inference

**Files:**
- Modify: `scripts/run_inference.py`

- [ ] **Step 1: Add full pipeline mode**

```python
def run_full_pipeline(claims_path: Path, output_path: Path, top_k: int,
                      classifier_ckpt_name: str = "classifier_noise-aware.pt") -> dict:
    cfg = Config()
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bm25 = BM25Retriever.from_cache(cfg.cache_dir / "bm25.pkl")
    ce_tok, ce_model = load_cross_encoder(cfg.cross_encoder_model,
                                          cfg.ckpt_dir / "cross_encoder.pt", device=device)
    cls_tok, cls_model = load_classifier(cfg.classifier_model,
                                         cfg.ckpt_dir / classifier_ckpt_name,
                                         num_labels=len(cfg.label_names), device=device)
    evidence = load_evidence(cfg.evidence_path)
    claims = load_claims(claims_path)

    pipeline = FactCheckingPipeline(
        bm25=bm25, ce_tok=ce_tok, ce_model=ce_model,
        cls_tok=cls_tok, cls_model=cls_model,
        evidence_lookup=evidence, id2label=cfg.id2label,
        bm25_top_k=cfg.bm25_top_k, final_top_k=top_k, device=device,
        ce_max_len=cfg.ce_max_len, cls_max_len=cfg.cls_max_len,
    )

    preds: dict[str, dict] = {}
    with timer(f"Full pipeline x{len(claims)}", log):
        for cid, claim in tqdm(claims.items()):
            preds[cid] = pipeline.predict(claim.claim_text)
    save_json(preds, output_path)
    return preds
```

Add to imports:

```python
from tqdm import tqdm
from src.classifier import load_classifier, predict_label
from src.pipeline import FactCheckingPipeline
```

Add CLI mode `--mode full` and `--classifier {gold-only,noise-aware}`.

- [ ] **Step 2: Add ORACLE mode (spec §7.1: classifier upper bound with gold evidence)**

Append to `scripts/run_inference.py`:

```python
def run_oracle(claims_path: Path, output_path: Path,
               classifier_ckpt_name: str = "classifier_gold-only.pt") -> dict:
    """Skip retrieval entirely. Feed each claim with its GOLD evidence list
    to the classifier. Measures classifier ceiling — answers 'how good
    would we be if retrieval were perfect?' (spec §7.1)."""
    cfg = Config()
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cls_tok, cls_model = load_classifier(cfg.classifier_model,
                                         cfg.ckpt_dir / classifier_ckpt_name,
                                         num_labels=len(cfg.label_names), device=device)
    evidence = load_evidence(cfg.evidence_path)
    claims = load_claims(claims_path)

    preds: dict[str, dict] = {}
    with timer(f"Oracle eval x{len(claims)}", log):
        for cid, claim in tqdm(claims.items()):
            ev_texts = [evidence[e] for e in claim.evidences if e in evidence]
            if not ev_texts:
                ev_texts = [""]
            out = predict_label(cls_model, cls_tok,
                                claim_text=claim.claim_text,
                                evidences_text=ev_texts,
                                device=device, max_len=cfg.cls_max_len)
            preds[cid] = {
                "claim_text": claim.claim_text,
                "claim_label": cfg.id2label[out["label_id"]],
                "evidences": claim.evidences,  # use gold so F-score = 1.0 (isolates classification)
                "confidence": out["confidence"],
            }
    save_json(preds, output_path)
    return preds
```

Update CLI to allow `--mode oracle` (no `--top-k` needed). Output filename: `outputs/dev-oracle-{classifier}.json`.

- [ ] **Step 3: Run on dev (noise-aware classifier)**

```bash
python -m scripts.run_inference --split dev --mode full --top-k 4 --classifier noise-aware
python eval.py --predictions outputs/dev-full-noise-aware-k4.json --groundtruth data/dev-claims.json
```

- [ ] **Step 4: Run ORACLE evaluation**

```bash
python -m scripts.run_inference --split dev --mode oracle --classifier gold-only
python -m scripts.run_inference --split dev --mode oracle --classifier noise-aware
python eval.py --predictions outputs/dev-oracle-gold-only.json --groundtruth data/dev-claims.json
python eval.py --predictions outputs/dev-oracle-noise-aware.json --groundtruth data/dev-claims.json
```

Expected: F=1.0 for both (we feed gold evidences); A reveals the classifier upper bound. Compare to E2E run in step 3 — the **gap** is what your retrieval errors cost you, the central narrative of the report's analysis section.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_inference.py
git commit -m "feat(scripts): Add full + oracle inference modes"
```

---

### Task 4.3: Ablation table generator

**Files:**
- Create: `scripts/ablation_table.py`

- [ ] **Step 1: Implement**

```python
"""Run the ablation matrix and write a markdown table to docs/."""
from __future__ import annotations

import subprocess
from pathlib import Path

from src.config import Config
from src.evaluator import evaluate_predictions
from src.utils import get_logger

log = get_logger("ablation")


def cmd(args: list[str]) -> None:
    log.info("Run: %s", " ".join(args))
    subprocess.run(args, check=True)


def main() -> None:
    cfg = Config()
    rows: list[tuple[str, dict]] = []

    K = str(cfg.final_top_k)

    # Variant 1: BM25 only + random label
    pred = cfg.output_dir / f"dev-bm25-random-k{K}.json"
    if not pred.exists():
        cmd(["python", "-m", "scripts.run_inference",
             "--split", "dev", "--mode", "bm25-random", "--top-k", K])
    rows.append(("BM25 only + random", evaluate_predictions(pred, cfg.dev_path)))

    # Variant 2: BM25 + cross-encoder + random label
    pred = cfg.output_dir / f"dev-retriever-only-k{K}.json"
    if not pred.exists():
        cmd(["python", "-m", "scripts.run_inference",
             "--split", "dev", "--mode", "retriever-only", "--top-k", K])
    rows.append(("+ cross-encoder", evaluate_predictions(pred, cfg.dev_path)))

    # Variant 3: ORACLE classifier (gold evidence, gold-only classifier)
    # → measures classifier's *upper bound* without retrieval errors (spec §7.1)
    pred = cfg.output_dir / "dev-oracle-gold-only.json"
    if not pred.exists():
        cmd(["python", "-m", "scripts.run_inference",
             "--split", "dev", "--mode", "oracle", "--classifier", "gold-only"])
    rows.append(("ORACLE: gold-evidence + classifier", evaluate_predictions(pred, cfg.dev_path)))

    # Variant 4: full pipeline, gold-only classifier (suffers from train-test mismatch)
    pred = cfg.output_dir / f"dev-full-gold-only-k{K}.json"
    if not pred.exists():
        cmd(["python", "-m", "scripts.run_inference",
             "--split", "dev", "--mode", "full", "--top-k", K,
             "--classifier", "gold-only"])
    rows.append(("E2E: + classifier (gold-only)", evaluate_predictions(pred, cfg.dev_path)))

    # Variant 5: full pipeline, noise-aware classifier (Plan B proposed)
    pred = cfg.output_dir / f"dev-full-noise-aware-k{K}.json"
    if not pred.exists():
        cmd(["python", "-m", "scripts.run_inference",
             "--split", "dev", "--mode", "full", "--top-k", K,
             "--classifier", "noise-aware"])
    rows.append(("E2E: + noise-aware (FULL Plan B)", evaluate_predictions(pred, cfg.dev_path)))

    md = ["# Plan B Ablation (dev set, K=4)", "",
          "| Variant | F | A | HM |", "|---|---|---|---|"]
    for name, m in rows:
        md.append(f"| {name} | {m['evidence_f']:.4f} | "
                  f"{m['claim_accuracy']:.4f} | {m['harmonic_mean']:.4f} |")
    (Path("docs/superpowers/plans") / "ablation-results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
python -m scripts.ablation_table
```

- [ ] **Step 3: Commit**

```bash
git add scripts/ablation_table.py docs/superpowers/plans/ablation-results.md
git commit -m "feat(scripts): Add Plan B ablation table"
```

---

### Task 4.4: Top-K hyperparameter sweep

- [ ] **Step 1: Run sweep**

```bash
for k in 3 4 5 6; do
  python -m scripts.run_inference --split dev --mode full --top-k $k --classifier noise-aware
done
```

- [ ] **Step 2: Pick best K**

Compare HM across K values. Update `src/config.py` `final_top_k` to the best K.

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "chore(config): Set final_top_k to dev winner"
```

---

### Task 4.5: Test set predictions (for leaderboard, optional)

- [ ] **Step 1: Run on test**

```bash
# K is whatever final_top_k is in src/config.py after Task 4.4
K=$(python -c "from src.config import Config; print(Config().final_top_k)")
python -m scripts.run_inference --split test --mode full --top-k "$K" --classifier noise-aware
```

- [ ] **Step 2: Verify format**

```bash
K=$(python -c "from src.config import Config; print(Config().final_top_k)")
python -c "
import json
preds = json.load(open(f'outputs/test-full-noise-aware-k${K}.json'))
print(len(preds), 'predictions')
sample = next(iter(preds.values()))
assert 'claim_label' in sample and 'evidences' in sample
assert sample['claim_label'] in ('SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO', 'DISPUTED')
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
# .gitignore excludes outputs/*.json by default; force-add the leaderboard file
K=$(python -c "from src.config import Config; print(Config().final_top_k)")
git add -f outputs/test-full-noise-aware-k${K}.json
git commit -m "feat(outputs): Add test-set predictions"
```

---

### Task 4.6: Error analysis notebook (in code form)

**Files:**
- Create: `scripts/error_analysis.py`

- [ ] **Step 1: Implement**

```python
"""Per-class accuracy + confusion matrix + qualitative failures."""
from __future__ import annotations

import json
from pathlib import Path

from src.config import Config
from src.evaluator import evaluate_predictions
from src.utils import load_json


def main() -> None:
    cfg = Config()
    pred_path = cfg.output_dir / f"dev-full-noise-aware-k{cfg.final_top_k}.json"
    metrics = evaluate_predictions(pred_path, cfg.dev_path)

    print("=== Overall ===")
    print(f"F={metrics['evidence_f']:.4f}  A={metrics['claim_accuracy']:.4f}  "
          f"HM={metrics['harmonic_mean']:.4f}")
    print("\n=== Per-class accuracy ===")
    for k, v in metrics["per_class_accuracy"].items():
        print(f"  {k:<18} {v:.4f}  (n={metrics['per_class_total'][k]})")
    print("\n=== Confusion matrix (rows=gold, cols=pred) ===")
    cm = metrics["confusion_matrix"]
    labels = list(cfg.label_names)
    header = "GOLD\\PRED  | " + " | ".join(f"{l[:6]:>6}" for l in labels)
    print(header); print("-" * len(header))
    for g in labels:
        row = " | ".join(f"{cm.get(g, {}).get(p, 0):>6}" for p in labels)
        print(f"{g[:9]:<10} | {row}")

    # Save 10 failures for report appendix
    preds = load_json(pred_path)
    gold = load_json(cfg.dev_path)
    failures = []
    for cid, g in gold.items():
        p = preds.get(cid, {})
        if p.get("claim_label") != g["claim_label"]:
            failures.append({
                "claim_id": cid, "claim_text": g["claim_text"],
                "gold": g["claim_label"], "pred": p.get("claim_label"),
                "retrieved": p.get("evidences"), "gold_ev": g["evidences"],
            })
    out = Path("docs/superpowers/plans/error-cases.json")
    out.write_text(json.dumps(failures[:10], indent=2, ensure_ascii=False))
    print(f"\nSaved {len(failures)} failures, first 10 → {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
python -m scripts.error_analysis > docs/superpowers/plans/error-analysis.txt
```

- [ ] **Step 3: Commit**

```bash
git add scripts/error_analysis.py docs/superpowers/plans/error-analysis.txt docs/superpowers/plans/error-cases.json
git commit -m "feat(scripts): Add error analysis script"
```

---

# Phase 5: ipynb Migration (Milestone M5)

Goal: copy module contents into `Group_073_COMP90042_Project_2026.ipynb` so it runs end-to-end in fresh Colab. We do **not** delete the `.py` files — they remain the experiment driver. The `.ipynb` is the marker-facing artifact.

### Task 5.1: Migrate Section 1 — Data

- [ ] **Step 1: Replace empty cell after `# 1.DataSet Processing`**

Cell content = paste in order:
1. `requirements.txt` (`!pip install -q ...`)
2. `src/utils.py`
3. `src/config.py` (set `repo_root` to Colab path or `/content/drive/...`)
4. `src/data_loader.py`
5. `src/preprocessing.py`
6. `src/retriever_bm25.py`
7. `scripts/build_bm25.py` body inline (build the index)

Verify the cell runs.

- [ ] **Step 2: Commit**

```bash
git add Group_073_COMP90042_Project_2026.ipynb
git commit -m "feat(notebook): Add section 1 (data + BM25)"
```

---

### Task 5.2: Migrate Section 2 — Models

- [ ] **Step 1: Replace empty cell after `# 2.Model Implementation`**

Cell content = paste:
1. `src/hard_negatives.py`
2. `src/retriever_cross_enc.py`
3. `src/classifier.py`
4. Training calls (inline body of `train_cross_encoder.py` and `train_classifier.py`).

Verify both trainings can be re-run from the notebook.

- [ ] **Step 2: Commit**

```bash
git add Group_073_COMP90042_Project_2026.ipynb
git commit -m "feat(notebook): Add section 2 (CE + classifier)"
```

---

### Task 5.3: Migrate Section 3 — Testing & Eval

- [ ] **Step 1: Replace empty cell after `# 3.Testing and Evaluation`**

Cell content:
1. `src/pipeline.py`
2. `src/evaluator.py`
3. Inline body of `run_inference.py` for dev + test
4. Inline body of `ablation_table.py`
5. Inline body of `error_analysis.py`
6. Final cell: print full eval results so the run log is captured per rule #5.

- [ ] **Step 2: Run notebook top-to-bottom in fresh Colab**

Confirm: ablation table prints, dev predictions JSON saved, test predictions JSON saved, no errors.

- [ ] **Step 3: Commit**

```bash
git add Group_073_COMP90042_Project_2026.ipynb
git commit -m "feat(notebook): Add section 3 (eval + ablation)"
```

---

### Task 5.4: Final reproducibility check

- [ ] **Step 1: Restart kernel + run all in clean Colab session**

Verify all cells pass and outputs match `outputs/*.json`.

- [ ] **Step 2: Save run log to README cell**

In the top "Readme" markdown cell, paste the final ablation table and any caveats (e.g., "BM25 index build takes ~10 min; cached to Drive").

- [ ] **Step 3: Tag release**

```bash
git tag v1.0-plan-b-final
git commit --allow-empty -m "chore: tag plan B final submission"
```

---

# Self-Review (performed after first draft)

Issues found and fixed inline before handoff:

**Critical (would have broken execution):**
1. Scripts originally named `01_build_bm25_index.py` etc. — module names starting
   with digits are not valid Python identifiers, so `python -m scripts.01_xxx`
   would `SyntaxError`. Renamed all scripts to `build_bm25.py`,
   `train_cross_encoder.py`, `train_classifier.py`, `run_inference.py`,
   `ablation_table.py`, `error_analysis.py`.
2. Missing `scripts/__init__.py` — added in Task 0.1.
3. Task 1.8 referenced `outputs/.gitkeep` that was never created — added
   `.gitkeep` files for `outputs/`, `checkpoints/`, `cache/` in Task 0.1.

**Spec gap:**
4. Spec §7.1 requires Oracle vs End-to-End evaluation; original plan only had
   end-to-end variants. Added an `oracle` mode to `run_inference.py` (Task 4.2
   step 2) and an Oracle row to the ablation table (Task 4.3). Now the report
   has a proper "what does retrieval cost us?" narrative.

**Minor:**
5. Task 4.5 hard-coded `k4` in the test-set output filename even though
   Task 4.4 selects best K dynamically. Replaced with `$K` shell var.
6. Task 4.5 commit step would silently skip the test predictions because
   `outputs/*.json` is gitignored — added `git add -f`.

**Spec coverage table** (updated after fixes):
- §1 Goal → Phase 4 (full pipeline + ablation)
- §2 Success criteria → Task 1.4 (eval reproduction), Task 4.3 (ablation table), Task 5.4 (reproducibility check)
- §3 Architecture → Phases 1–4 (Stage 1A BM25, Stage 1B cross-encoder, Stage 2 classifier all covered)
- §4 File structure → Task 0.1 + per-module tasks; matches diagram
- §5 Workflow milestones → Phases 1–5 = M1–M5
- §6 Resources → Task 2.3 (CE wall-clock), Task 5.4 (fresh Colab check)
- §7 Evaluation design → **§7.1 Oracle vs E2E** = Task 4.2 step 2 + Task 4.3 variant 3; **§7.2 per-class** + **§7.3 ablation** = Tasks 4.3 + 4.6
- §8 Compliance → Tasks 1.2 (set_seed), Phase 5 (ipynb migration)
- §9 Out of scope → correctly not implemented
- §10 Open questions → fixed to defaults at top of plan
- §11 Verification items → covered in M1 EDA (Task 1.6 toy + 1.8 sweep)
- §12 DoD → Tasks 4.5, 5.4

**Placeholder scan:** No "TODO", no "TBD". Two phrases "Fill in the table"
(Task 1.9) and "(placeholder)" (docstring in run_inference) are intentional —
the first asks the engineer to record empirical numbers, the second describes
behaviour. Both are accurate, not unfinished.

**Type/name consistency check:**
- `Config` field names used in every script match the dataclass definition
- `build_classifier` / `load_classifier` / `predict_label` signatures consistent across `src/classifier.py`, `tests/test_classifier.py`, and `pipeline.py`
- `build_cross_encoder` / `load_cross_encoder` / `rerank` signatures consistent across `src/retriever_cross_enc.py`, `tests/test_retriever_cross_enc.py`, `scripts/run_inference.py`
- `build_training_pairs` / `build_examples` both accept either `Claim` objects or `dict` — handled inside the function via `hasattr` checks
- `cfg.label_names` is a 4-tuple; `len(cfg.label_names) == 4` used everywhere a `num_labels` is expected

---

# Notes / Risks

- 🚫 **DO NOT MODIFY `eval.py` UNDER ANY CIRCUMSTANCES.** It is the official
  marker-side evaluation script provided by the teaching team and used to grade
  the leaderboard. Touching it (even refactoring, adding logging, fixing
  formatting, or "improving" anything) is **strictly forbidden** because:
  1. The marker re-runs the original `eval.py` against your `predictions.json` —
     your local edits never reach them, so changing it gives you false confidence.
  2. Any divergence between your reported numbers and the marker's run = lost
     marks for reproducibility / integrity (Project Rules §5, §6).
  3. Project Rule §6 explicitly prohibits "post-hoc modifications" of evaluation
     artifacts.

  If you find a bug or want extra metrics: **wrap** `eval.py` from
  `src/evaluator.py` (already done in Task 1.4) — never edit `eval.py` itself.
  Same rule applies to **`data/*.json`** files (train/dev/test claims, evidence) —
  read-only.
- **evidence.json must be downloaded** to `data/evidence.json` before any retrieval runs (see `data/evidence.md` for the link). The plan assumes this is already done (verified at planning time: 174 MB).
- **Colab session timeout (12h)**: if cross-encoder training crosses session boundary, save checkpoints to Google Drive. We mount Drive at notebook top in Phase 5.
- **Class imbalance**: handled via `class_weights_from_labels` in classifier; DISPUTED is rarest (124 train / 18 dev) and likely worst per-class accuracy. This is reported, not fixed.
- **Hard negatives may include false negatives** (BM25 may miss gold evidence that isn't on the gold list but is still correct). Acceptable for first cut; report in limitations.

---

End of plan.
