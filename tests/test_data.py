import numpy as np
import pytest

from unlearning_lab.data import class_counts, load_unlearning_data


def test_forget_split_contains_only_requested_class() -> None:
    data = load_unlearning_data(forget_class=7, seed=0)

    assert set(data.train_forget.labels.tolist()) == {7}
    assert 7 not in set(data.train_retain.labels.tolist())
    assert data.train_retain.features.shape[1] == 64


def test_data_split_is_deterministic() -> None:
    first = load_unlearning_data(forget_class=8, seed=4)
    second = load_unlearning_data(forget_class=8, seed=4)

    np.testing.assert_allclose(first.train_retain.features, second.train_retain.features)
    np.testing.assert_array_equal(first.test.labels, second.test.labels)


def test_class_counts_are_plain_ints() -> None:
    counts = class_counts(np.array([2, 2, 5]))

    assert counts == {2: 2, 5: 1}


def test_rejects_bad_forget_class() -> None:
    with pytest.raises(ValueError, match="forget_class"):
        load_unlearning_data(forget_class=12)
