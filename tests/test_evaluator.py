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


def test_precision_uses_list_length_not_set(tmp_path):
    """Official eval.py uses len(predicted_evidence_list); duplicates count.
    Our wrapper must match so Phase 4 oracle/blend modes don't silently
    diverge from the leaderboard."""
    from src.evaluator import evaluate_predictions
    from src.utils import save_json

    pred_path = tmp_path / "pred.json"
    gold_path = tmp_path / "gold.json"
    save_json(
        {
            "c1": {
                "claim_text": "x",
                "claim_label": "SUPPORTS",
                "evidences": ["e-1", "e-1", "e-2", "e-3"],  # duplicate e-1
            }
        },
        pred_path,
    )
    save_json(
        {"c1": {"claim_text": "x", "claim_label": "SUPPORTS", "evidences": ["e-1"]}}, gold_path
    )

    # Official: precision = 1 (TP=1) / 4 (list length) = 0.25, recall = 1/1 = 1.0
    # F = 2*0.25*1 / 1.25 = 0.4
    m = evaluate_predictions(pred_path, gold_path)
    assert abs(m["evidence_f"] - 0.4) < 1e-6, f"got F={m['evidence_f']}"
