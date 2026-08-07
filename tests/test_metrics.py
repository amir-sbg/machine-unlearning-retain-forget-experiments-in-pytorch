import numpy as np
import pytest

from unlearning_lab.metrics import (
    compare_to_exact_retrain,
    method_scorecard,
    split_metrics,
)


def test_split_metrics_reports_forget_confidence() -> None:
    logits = np.array(
        [
            [4.0, 1.0],
            [1.0, 4.0],
            [0.5, 2.0],
        ]
    )
    labels = np.array([0, 1, 1])

    metrics = split_metrics(labels, logits, forget_class=1)

    assert metrics["accuracy"] == 1.0
    assert metrics["forget_rows"] == 2
    assert metrics["forget_confidence"] > 0.5


def test_split_metrics_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="matching rows"):
        split_metrics(np.array([0, 1]), np.zeros((3, 2)), forget_class=1)


def test_compare_to_exact_retrain_returns_gaps() -> None:
    metrics = {
        "exact_retrain": {
            "test": {
                "retain_accuracy": 0.95,
                "forget_confidence": 0.10,
                "forget_accuracy": 0.05,
            }
        },
        "cheap_method": {
            "test": {
                "retain_accuracy": 0.90,
                "forget_confidence": 0.25,
                "forget_accuracy": 0.20,
            }
        },
    }

    gaps = compare_to_exact_retrain(metrics)

    assert gaps["cheap_method"]["retain_accuracy_gap"] == pytest.approx(0.05)
    assert gaps["cheap_method"]["forget_confidence_gap"] == pytest.approx(0.15)


def test_method_scorecard_ranks_by_retrain_gap() -> None:
    metrics = {
        "exact_retrain": {
            "test": {
                "retain_accuracy": 0.95,
                "forget_confidence": 0.10,
                "forget_accuracy": 0.05,
            }
        },
        "cheap_method": {
            "test": {
                "retain_accuracy": 0.90,
                "forget_confidence": 0.25,
                "forget_accuracy": 0.20,
            }
        },
    }

    rows = method_scorecard(metrics, timings={"cheap_method": 0.2})

    assert rows[0]["method"] == "exact_retrain"
    assert rows[1]["runtime_seconds"] == 0.2
    assert rows[1]["total_retrain_gap"] == pytest.approx(0.35)
