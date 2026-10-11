import pytest
import torch

from unlearning_lab.data import Split
from unlearning_lab.model import DigitMLP
from unlearning_lab.unlearn import (
    UnlearnConfig,
    dampen_output_class,
    gradient_conflict_summary,
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


def test_dampen_output_class_scales_requested_row() -> None:
    model = DigitMLP(hidden_dim=32, dropout=0.0)
    original_weight = model.net[-1].weight.detach().clone()
    original_bias = model.net[-1].bias.detach().clone()

    scrubbed = dampen_output_class(model, class_id=4, weight_scale=0.25, bias_shift=-0.5)
    updated_weight = scrubbed.net[-1].weight.detach()
    updated_bias = scrubbed.net[-1].bias.detach()

    assert torch.allclose(updated_weight[4], original_weight[4] * 0.25)
    assert torch.allclose(updated_weight[3], original_weight[3])
    assert torch.isclose(updated_bias[4], original_bias[4] - 0.5)
    assert torch.isclose(updated_bias[3], original_bias[3])


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


def test_gradient_conflict_summary_is_finite_and_non_destructive() -> None:
    features = torch.eye(12, 64).numpy().astype("float32")
    labels = (torch.arange(12) % 3).numpy()
    split = Split(features, labels)
    model = DigitMLP(hidden_dim=32, dropout=0.0)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    report = gradient_conflict_summary(
        model,
        split,
        split,
        batch_size=12,
        device=torch.device("cpu"),
    )

    assert report["gradient_cosine_similarity"] == pytest.approx(1.0, abs=1e-6)
    assert report["gradient_conflict"] == pytest.approx(0.0)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
