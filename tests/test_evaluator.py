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
