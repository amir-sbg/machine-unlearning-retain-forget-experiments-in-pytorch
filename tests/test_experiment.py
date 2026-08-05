import numpy as np

from unlearning_lab.data import Split
from unlearning_lab.experiment import join_splits


def test_join_splits_preserves_rows() -> None:
    first = Split(np.zeros((2, 3), dtype=np.float32), np.array([0, 1]))
    second = Split(np.ones((1, 3), dtype=np.float32), np.array([2]))

    joined = join_splits(first, second)

    assert joined.features.shape == (3, 3)
    assert joined.labels.tolist() == [0, 1, 2]
