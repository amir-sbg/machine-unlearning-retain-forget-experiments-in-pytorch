import torch

from unlearning_lab.data import Split
from unlearning_lab.model import DigitMLP
from unlearning_lab.unlearn import (
    UnlearnConfig,
    negative_gradient_unlearn,
    reset_output_class,
)


def test_reset_output_class_changes_only_requested_row() -> None:
    model = DigitMLP(hidden_dim=32, dropout=0.0)
    original = model.net[-1].weight.detach().clone()

    scrubbed = reset_output_class(model, class_id=3, seed=2)
    updated = scrubbed.net[-1].weight.detach()

    assert not torch.allclose(original[3], updated[3])
    assert torch.allclose(original[2], updated[2])


def test_negative_gradient_unlearning_records_steps() -> None:
    features = torch.eye(20, 64).numpy().astype("float32")
    labels = (torch.arange(20) % 10).numpy()
    retain = Split(features[:16], labels[:16])
    forget = Split(features[16:], labels[16:])
    validation = Split(features, labels)

    model, history = negative_gradient_unlearn(
        DigitMLP(hidden_dim=32, dropout=0.0),
        retain,
        forget,
        validation,
        UnlearnConfig(steps=2, batch_size=4, seed=1),
        torch.device("cpu"),
    )

    assert isinstance(model, DigitMLP)
    assert history[-1]["step"] == 2.0
