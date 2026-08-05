import torch

from unlearning_lab.data import Split
from unlearning_lab.model import DigitMLP, count_parameters
from unlearning_lab.train import TrainingConfig, train_classifier


def test_model_outputs_digit_logits() -> None:
    model = DigitMLP(hidden_dim=32)
    logits = model(torch.zeros(3, 64))

    assert logits.shape == (3, 10)
    assert count_parameters(model) > 0


def test_training_loop_records_history() -> None:
    features = torch.eye(10, 64).numpy().astype("float32")
    labels = torch.arange(10).numpy()
    split = Split(features, labels)

    result = train_classifier(
        DigitMLP(hidden_dim=32, dropout=0.0),
        split,
        split,
        TrainingConfig(epochs=2, batch_size=5, patience=2, seed=1),
        torch.device("cpu"),
    )

    assert result.epochs_trained >= 1
    assert result.history[-1]["validation_loss"] >= 0
