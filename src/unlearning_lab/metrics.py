from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import Split, UnlearningData
from .train import predict_logits


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def split_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    forget_class: int,
) -> dict[str, float | int]:
    labels = np.asarray(labels)
    logits = np.asarray(logits)
    if labels.ndim != 1 or logits.ndim != 2:
        raise ValueError("labels must be 1-D and logits must be 2-D")
    if len(labels) != len(logits):
        raise ValueError("labels and logits must have matching rows")
    if len(labels) == 0:
        raise ValueError("labels and logits must not be empty")
    if not 0 <= forget_class < logits.shape[1]:
        raise ValueError("forget_class must be inside the logits class dimension")

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
    rows = []
    for name, metrics in method_metrics.items():
        gap = gaps[name]
        test = metrics["test"]
        total_gap = (
            gap["retain_accuracy_gap"]
            + gap["forget_confidence_gap"]
            + gap["forget_accuracy_gap"]
        )
        rows.append(
            {
                "method": name,
                "is_exact_retrain": name == exact_key,
                "runtime_seconds": float(timings.get(name, 0.0)),
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


def save_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
