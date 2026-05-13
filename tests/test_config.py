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


def test_config_hard_neg_bm25_pool_default():
    cfg = Config()
    assert cfg.hard_negatives_bm25_top_k == 200


def test_config_has_dense_retrieval_fields():
    from src.config import Config
    cfg = Config()
    assert cfg.dense_encoder == "BAAI/bge-base-en-v1.5"
    assert cfg.dense_top_k == 200
    assert cfg.rrf_k == 60
    assert cfg.hybrid_pool_size == 200
    assert cfg.dense_index_path == cfg.cache_dir / "dense_index_bge.faiss"
    assert cfg.dense_ids_path == cfg.cache_dir / "dense_index_bge.ids.json"
