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
