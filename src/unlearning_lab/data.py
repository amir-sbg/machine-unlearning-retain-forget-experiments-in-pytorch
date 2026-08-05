from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Split:
    features: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return int(len(self.labels))


@dataclass(frozen=True)
class UnlearningData:
    train_retain: Split
    train_forget: Split
    validation: Split
    test: Split
    forget_class: int
    feature_names: tuple[str, ...]


def _validate_sizes(validation_size: float, test_size: float) -> None:
    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("validation_size and test_size must be between 0 and 1")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must sum to less than 1")


def load_unlearning_data(
    forget_class: int = 8,
    seed: int = 42,
    validation_size: float = 0.20,
    test_size: float = 0.20,
) -> UnlearningData:
    _validate_sizes(validation_size, test_size)
    if not 0 <= forget_class <= 9:
        raise ValueError("forget_class must be a digit from 0 to 9")

    dataset = load_digits()
    features = dataset.data.astype(np.float32) / 16.0
    labels = dataset.target.astype(np.int64)

    train_features, holdout_features, train_labels, holdout_labels = train_test_split(
        features,
        labels,
        test_size=validation_size + test_size,
        stratify=labels,
        random_state=seed,
    )
    test_fraction = test_size / (validation_size + test_size)
    validation_features, test_features, validation_labels, test_labels = train_test_split(
        holdout_features,
        holdout_labels,
        test_size=test_fraction,
        stratify=holdout_labels,
        random_state=seed,
    )

    forget_mask = train_labels == forget_class
    if not np.any(forget_mask):
        raise ValueError("forget split is empty")

    # Keep the forget examples separate; the full model sees them, unlearned models should not.
    train_forget = Split(train_features[forget_mask], train_labels[forget_mask])
    train_retain = Split(train_features[~forget_mask], train_labels[~forget_mask])

    return UnlearningData(
        train_retain=train_retain,
        train_forget=train_forget,
        validation=Split(validation_features, validation_labels),
        test=Split(test_features, test_labels),
        forget_class=forget_class,
        feature_names=tuple(f"pixel_{index}" for index in range(features.shape[1])),
    )


def class_counts(labels: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def dataset_summary(data: UnlearningData) -> dict:
    return {
        "forget_class": data.forget_class,
        "rows": {
            "train_retain": len(data.train_retain),
            "train_forget": len(data.train_forget),
            "validation": len(data.validation),
            "test": len(data.test),
        },
        "class_counts": {
            "train_retain": class_counts(data.train_retain.labels),
            "train_forget": class_counts(data.train_forget.labels),
            "validation": class_counts(data.validation.labels),
            "test": class_counts(data.test.labels),
        },
        "feature_count": len(data.feature_names),
    }
