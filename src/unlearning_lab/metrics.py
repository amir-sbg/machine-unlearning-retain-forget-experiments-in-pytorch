from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import Split, UnlearningData
from .train import predict_logits


def _checked_labels_logits(
    labels: np.ndarray,
    logits: np.ndarray,
    forget_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    logits = np.asarray(logits)
    if labels.ndim != 1 or logits.ndim != 2:
        raise ValueError("labels must be 1-D and logits must be 2-D")
    if len(labels) != len(logits):
        raise ValueError("labels and logits must have matching rows")
    if len(labels) == 0:
        raise ValueError("labels and logits must not be empty")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must contain only finite values")
    if not 0 <= forget_class < logits.shape[1]:
        raise ValueError("forget_class must be inside the logits class dimension")
    return labels, logits


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def split_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    forget_class: int,
) -> dict[str, float | int]:
    labels, logits = _checked_labels_logits(labels, logits, forget_class)

    probabilities = softmax(logits)
    predictions = probabilities.argmax(axis=1)
    forget_mask = labels == forget_class
    retain_mask = ~forget_mask

    result: dict[str, float | int] = {
        "rows": int(len(labels)),
        "accuracy": float(np.mean(predictions == labels)),
        "mean_forget_class_probability": float(np.mean(probabilities[:, forget_class])),
    }
    if np.any(retain_mask):
        result["retain_accuracy"] = float(np.mean(predictions[retain_mask] == labels[retain_mask]))
        result["retain_rows"] = int(np.sum(retain_mask))
    else:
        result["retain_accuracy"] = 0.0
        result["retain_rows"] = 0
    if np.any(forget_mask):
        result["forget_accuracy"] = float(np.mean(predictions[forget_mask] == labels[forget_mask]))
        result["forget_rows"] = int(np.sum(forget_mask))
        result["forget_confidence"] = float(np.mean(probabilities[forget_mask, forget_class]))
    else:
        result["forget_accuracy"] = 0.0
        result["forget_rows"] = 0
        result["forget_confidence"] = 0.0
    return result


def _true_label_stats(labels: np.ndarray, logits: np.ndarray) -> dict[str, float | int]:
    probabilities = np.clip(softmax(logits), 1e-12, 1.0)
    rows = np.arange(len(labels))
    true_probabilities = probabilities[rows, labels]
    nll = -np.log(true_probabilities)
    return {
        "rows": int(len(labels)),
        "true_label_confidence": float(np.mean(true_probabilities)),
        "true_label_nll": float(np.mean(nll)),
    }


def membership_signal_summary(
    train_forget_labels: np.ndarray,
    train_forget_logits: np.ndarray,
    holdout_forget_labels: np.ndarray,
    holdout_forget_logits: np.ndarray,
    forget_class: int,
) -> dict[str, float | int]:
    train_labels, train_logits = _checked_labels_logits(
        train_forget_labels,
        train_forget_logits,
        forget_class,
    )
    holdout_labels, holdout_logits = _checked_labels_logits(
        holdout_forget_labels,
        holdout_forget_logits,
        forget_class,
    )
    if not np.all(train_labels == forget_class):
        raise ValueError("train_forget_labels must contain only the forget class")
    if not np.all(holdout_labels == forget_class):
        raise ValueError("holdout_forget_labels must contain only the forget class")

    train = _true_label_stats(train_labels, train_logits)
    holdout = _true_label_stats(holdout_labels, holdout_logits)
    confidence_gap = float(
        train["true_label_confidence"] - holdout["true_label_confidence"]
    )
    nll_gap = float(holdout["true_label_nll"] - train["true_label_nll"])
    return {
        "train_forget_rows": train["rows"],
        "holdout_forget_rows": holdout["rows"],
        "train_forget_true_confidence": train["true_label_confidence"],
        "holdout_forget_true_confidence": holdout["true_label_confidence"],
        "confidence_gap_train_minus_holdout": confidence_gap,
        "train_forget_nll": train["true_label_nll"],
        "holdout_forget_nll": holdout["true_label_nll"],
        "nll_gap_holdout_minus_train": nll_gap,
        "membership_signal": float(max(0.0, confidence_gap) + max(0.0, nll_gap)),
    }


def _mean_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def distribution_gap_summary(
    labels: np.ndarray,
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    forget_class: int,
) -> dict[str, float]:
    labels, reference_logits = _checked_labels_logits(labels, reference_logits, forget_class)
    _, candidate_logits = _checked_labels_logits(labels, candidate_logits, forget_class)
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError("reference_logits and candidate_logits must have the same shape")

    reference = np.clip(softmax(reference_logits), 1e-12, 1.0)
    candidate = np.clip(softmax(candidate_logits), 1e-12, 1.0)
    midpoint = 0.5 * (reference + candidate)
    js = 0.5 * np.sum(reference * np.log(reference / midpoint), axis=1)
    js += 0.5 * np.sum(candidate * np.log(candidate / midpoint), axis=1)

    retain_mask = labels != forget_class
    forget_mask = labels == forget_class
    reference_predictions = reference.argmax(axis=1)
    candidate_predictions = candidate.argmax(axis=1)
    max_probability_shift = np.abs(candidate.max(axis=1) - reference.max(axis=1))
    return {
        "mean_js_divergence_to_retrain": _mean_or_zero(js),
        "retain_js_divergence_to_retrain": _mean_or_zero(js[retain_mask]),
        "forget_js_divergence_to_retrain": _mean_or_zero(js[forget_mask]),
        "prediction_disagreement_to_retrain": float(
            np.mean(candidate_predictions != reference_predictions)
        ),
        "mean_max_probability_shift": _mean_or_zero(max_probability_shift),
    }


def evaluate_model(
    model: nn.Module,
    split: Split,
    forget_class: int,
    device: torch.device,
) -> dict[str, float | int]:
    logits = predict_logits(model, split, device)
    return split_metrics(split.labels, logits, forget_class)


def evaluate_unlearning_model(
    model: nn.Module,
    data: UnlearningData,
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    return {
        "train_forget": evaluate_model(model, data.train_forget, data.forget_class, device),
        "test": evaluate_model(model, data.test, data.forget_class, device),
        "validation": evaluate_model(model, data.validation, data.forget_class, device),
    }


def compare_to_exact_retrain(
    method_metrics: dict[str, dict],
    exact_key: str = "exact_retrain",
) -> dict[str, dict[str, float]]:
    if exact_key not in method_metrics:
        raise ValueError(f"missing exact retrain metrics: {exact_key}")
    exact = method_metrics[exact_key]["test"]
    rows: dict[str, dict[str, float]] = {}
    for name, metrics in method_metrics.items():
        test = metrics["test"]
        rows[name] = {
            "retain_accuracy_gap": float(abs(test["retain_accuracy"] - exact["retain_accuracy"])),
            "forget_confidence_gap": float(abs(test["forget_confidence"] - exact["forget_confidence"])),
            "forget_accuracy_gap": float(abs(test["forget_accuracy"] - exact["forget_accuracy"])),
        }
    return rows


def method_scorecard(
    method_metrics: dict[str, dict],
    timings: dict[str, float] | None = None,
    exact_key: str = "exact_retrain",
) -> list[dict[str, float | str | bool]]:
    gaps = compare_to_exact_retrain(method_metrics, exact_key=exact_key)
    timings = timings or {}
    exact_runtime = float(timings.get(exact_key, 0.0))
    rows = []
    for name, metrics in method_metrics.items():
        gap = gaps[name]
        test = metrics["test"]
        runtime = float(timings.get(name, 0.0))
        speedup = exact_runtime / runtime if exact_runtime > 0.0 and runtime > 0.0 else 0.0
        total_gap = (
            gap["retain_accuracy_gap"]
            + gap["forget_confidence_gap"]
            + gap["forget_accuracy_gap"]
        )
        rows.append(
            {
                "method": name,
                "is_exact_retrain": name == exact_key,
                "runtime_seconds": runtime,
                "speedup_vs_exact_retrain": float(speedup),
                "test_retain_accuracy": float(test["retain_accuracy"]),
                "test_forget_accuracy": float(test["forget_accuracy"]),
                "test_forget_confidence": float(test["forget_confidence"]),
                "retain_accuracy_gap": gap["retain_accuracy_gap"],
                "forget_confidence_gap": gap["forget_confidence_gap"],
                "forget_accuracy_gap": gap["forget_accuracy_gap"],
                "total_retrain_gap": float(total_gap),
            }
        )
    return sorted(rows, key=lambda row: (row["total_retrain_gap"], row["runtime_seconds"]))


def pareto_frontier(
    scorecard: list[dict[str, float | str | bool]],
    runtime_key: str = "runtime_seconds",
    gap_key: str = "total_retrain_gap",
) -> list[dict[str, float | str | bool]]:
    frontier = []
    for candidate in scorecard:
        candidate_time = float(candidate[runtime_key])
        candidate_gap = float(candidate[gap_key])
        dominated = False
        for other in scorecard:
            if other is candidate:
                continue
            other_time = float(other[runtime_key])
            other_gap = float(other[gap_key])
            if other_time <= candidate_time and other_gap <= candidate_gap:
                if other_time < candidate_time or other_gap < candidate_gap:
                    dominated = True
                    break
        if not dominated:
            frontier.append(dict(candidate))
    return sorted(
        frontier,
        key=lambda row: (float(row[runtime_key]), float(row[gap_key]), str(row["method"])),
    )


def save_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
