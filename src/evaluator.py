"""Wraps the official eval.py logic and adds per-class accuracy and confusion matrix."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.utils import load_json


def evaluate_predictions(
    predictions_path: Path | str,
    groundtruth_path: Path | str,
) -> dict:
    """Return overall + per-class metrics as a flat dict."""
    preds = load_json(predictions_path)
    gold = load_json(groundtruth_path)

    f_scores: list[float] = []
    correct: list[float] = []
    per_class_total: Counter[str] = Counter()
    per_class_correct: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

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

    all_gold_labels = {gi["claim_label"] for gi in gold.values()}
    per_class_acc = {
        lbl: per_class_correct[lbl] / per_class_total[lbl] if per_class_total[lbl] else 0.0
        for lbl in all_gold_labels
    }

    return {
        "evidence_f": mean_f,
        "claim_accuracy": mean_acc,
        "harmonic_mean": hmean,
        "per_class_accuracy": per_class_acc,
        "per_class_total": dict(per_class_total),
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }
