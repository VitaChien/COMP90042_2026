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
    hard_negatives_per_pos: int = 8
    hard_negatives_bm25_top_k: int = 200

    # cross-encoder
    cross_encoder_model: str = "bert-base-uncased"
    ce_max_len: int = 256
    ce_lr: float = 2e-5
    ce_epochs: int = 2
    ce_batch_size: int = 64

    # classifier
    classifier_model: str = "roberta-base"
    cls_max_len: int = 384
    cls_lr: float = 2e-5
    cls_epochs: int = 3
    cls_batch_size: int = 16
    noise_mix_ratio: float = 0.5  # fraction of training samples using retrieved evidence

    label_names: tuple[str, ...] = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED")

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
    def label2id(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.label_names)}

    @property
    def id2label(self) -> dict[int, str]:
        return dict(enumerate(self.label_names))
