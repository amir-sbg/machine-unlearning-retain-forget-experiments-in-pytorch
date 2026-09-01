import numpy as np
import pytest

from unlearning_lab.metrics import (
    compare_to_exact_retrain,
    distribution_gap_summary,
    forget_confidence_curve,
    membership_signal_summary,
    method_scorecard,
    pareto_frontier,
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


def test_split_metrics_rejects_unknown_forget_class() -> None:
    with pytest.raises(ValueError, match="forget_class"):
        split_metrics(np.array([0, 1]), np.zeros((2, 2)), forget_class=3)


def test_split_metrics_rejects_non_finite_logits() -> None:
    logits = np.array([[1.0, 0.0], [np.nan, 2.0]])

    with pytest.raises(ValueError, match="finite"):
        split_metrics(np.array([0, 1]), logits, forget_class=1)


def test_membership_signal_compares_train_and_holdout_forget_examples() -> None:
    report = membership_signal_summary(
        train_forget_labels=np.array([1, 1]),
        train_forget_logits=np.array([[0.2, 3.0], [0.1, 2.5]]),
        holdout_forget_labels=np.array([1, 1]),
        holdout_forget_logits=np.array([[0.2, 1.0], [0.4, 1.2]]),
        forget_class=1,
    )

    assert report["train_forget_rows"] == 2
    assert report["holdout_forget_rows"] == 2
    assert report["confidence_gap_train_minus_holdout"] > 0
    assert report["nll_gap_holdout_minus_train"] > 0
    assert report["membership_signal"] > 0


def test_membership_signal_rejects_mixed_forget_split() -> None:
    with pytest.raises(ValueError, match="only the forget class"):
        membership_signal_summary(
            np.array([1, 0]),
            np.zeros((2, 2)),
            np.array([1]),
            np.zeros((1, 2)),
            forget_class=1,
        )


def test_forget_confidence_curve_reports_threshold_counts() -> None:
    report = forget_confidence_curve(
        labels=np.array([1, 1, 0]),
        logits=np.array([[0.0, 4.0], [0.0, 0.0], [4.0, 0.0]]),
        forget_class=1,
        thresholds=(0.25, 0.75),
    )

    assert report["forget_rows"] == 2
    assert report["thresholds"][0]["count_at_or_above"] == 2
    assert report["thresholds"][1]["count_at_or_above"] == 1


def test_forget_confidence_curve_rejects_bad_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        forget_confidence_curve(
            labels=np.array([1]),
            logits=np.array([[0.0, 1.0]]),
            forget_class=1,
            thresholds=(1.2,),
        )


def test_distribution_gap_compares_probability_profiles() -> None:
    labels = np.array([0, 1, 1])
    reference = np.array([[3.0, 0.0], [0.0, 3.0], [0.2, 2.0]])
    same = reference.copy()
    shifted = np.array([[0.0, 3.0], [2.0, 0.0], [1.5, 0.0]])

    exact = distribution_gap_summary(labels, reference, same, forget_class=1)
    report = distribution_gap_summary(labels, reference, shifted, forget_class=1)

    assert exact["mean_js_divergence_to_retrain"] == pytest.approx(0.0)
    assert report["mean_js_divergence_to_retrain"] > 0
    assert report["forget_js_divergence_to_retrain"] > 0
    assert report["prediction_disagreement_to_retrain"] > 0


def test_distribution_gap_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        distribution_gap_summary(
            np.array([0, 1]),
            np.zeros((2, 2)),
            np.zeros((2, 3)),
            forget_class=1,
        )


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


def test_method_scorecard_ranks_by_retrain_gap_and_reports_speedup() -> None:
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

    rows = method_scorecard(metrics, timings={"exact_retrain": 1.0, "cheap_method": 0.2})

    assert rows[0]["method"] == "exact_retrain"
    assert rows[1]["runtime_seconds"] == 0.2
    assert rows[1]["speedup_vs_exact_retrain"] == pytest.approx(5.0)
    assert rows[1]["total_retrain_gap"] == pytest.approx(0.35)


def test_pareto_frontier_keeps_fast_or_low_gap_methods() -> None:
    rows = [
        {"method": "exact", "runtime_seconds": 10.0, "total_retrain_gap": 0.0},
        {"method": "cheap", "runtime_seconds": 1.0, "total_retrain_gap": 0.15},
        {"method": "middle", "runtime_seconds": 4.0, "total_retrain_gap": 0.25},
    ]

    frontier = pareto_frontier(rows)

    assert [row["method"] for row in frontier] == ["cheap", "exact"]
