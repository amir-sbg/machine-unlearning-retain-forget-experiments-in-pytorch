import numpy as np
import pytest

from unlearning_lab.data import Split
from unlearning_lab.experiment import ExperimentConfig, _new_model, join_splits, select_class


def test_join_splits_preserves_rows() -> None:
    first = Split(np.zeros((2, 3), dtype=np.float32), np.array([0, 1]))
    second = Split(np.ones((1, 3), dtype=np.float32), np.array([2]))

    joined = join_splits(first, second)

    assert joined.features.shape == (3, 3)
    assert joined.labels.tolist() == [0, 1, 2]


def test_select_class_extracts_one_label() -> None:
    split = Split(
        np.arange(12, dtype=np.float32).reshape(4, 3),
        np.array([2, 3, 2, 4]),
    )

    selected = select_class(split, 2)

    assert selected.features.shape == (2, 3)
    assert selected.labels.tolist() == [2, 2]


def test_select_class_rejects_missing_label() -> None:
    split = Split(np.zeros((2, 3), dtype=np.float32), np.array([0, 1]))

    with pytest.raises(ValueError, match="does not contain"):
        select_class(split, 9)


def test_experiment_config_controls_model_width() -> None:
    small = _new_model(ExperimentConfig(hidden_dim=32, dropout=0.0))
    wide = _new_model(ExperimentConfig(hidden_dim=64, dropout=0.0))

    assert sum(parameter.numel() for parameter in wide.parameters()) > sum(
        parameter.numel() for parameter in small.parameters()
    )


def test_experiment_rejects_bad_model_shape() -> None:
    with pytest.raises(ValueError, match="hidden_dim"):
        _new_model(ExperimentConfig(hidden_dim=8))
    with pytest.raises(ValueError, match="dropout"):
        _new_model(ExperimentConfig(dropout=1.0))
