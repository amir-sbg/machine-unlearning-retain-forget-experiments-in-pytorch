from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import Split


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    gradient_clip: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class TrainingResult:
    model: nn.Module
    history: list[dict[str, float]]
    best_validation_loss: float
    epochs_trained: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    split: Split,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(split.features).float(),
        torch.from_numpy(split.labels).long(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def _validate_config(config: TrainingConfig) -> None:
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("optimizer settings are invalid")
    if config.patience < 1 or config.gradient_clip <= 0:
        raise ValueError("patience and gradient_clip must be positive")


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_rows = 0
    with torch.inference_mode():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            loss = loss_fn(model(features), labels)
            total_loss += float(loss) * len(labels)
            total_rows += len(labels)
    return total_loss / max(total_rows, 1)


def train_classifier(
    model: nn.Module,
    train_split: Split,
    validation_split: Split,
    config: TrainingConfig,
    device: torch.device,
) -> TrainingResult:
    _validate_config(config)
    set_seed(config.seed)
    model.to(device)

    train_loader = make_loader(
        train_split,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    validation_loader = make_loader(
        validation_split,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        last_grad_norm = 0.0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip,
            )
            optimizer.step()

            total_loss += float(loss.detach()) * len(labels)
            total_rows += len(labels)
            last_grad_norm = float(grad_norm)

        train_loss = total_loss / max(total_rows, 1)
        validation_loss = evaluate_loss(model, validation_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "grad_norm": last_grad_norm,
            }
        )

        if validation_loss < best_validation_loss - 1e-5:
            best_state = copy.deepcopy(model.state_dict())
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        history=history,
        best_validation_loss=best_validation_loss,
        epochs_trained=len(history),
    )


@torch.inference_mode()
def predict_logits(
    model: nn.Module,
    split: Split,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    loader = make_loader(split, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to(device)
    rows = []
    for features, _ in loader:
        rows.append(model(features.to(device)).cpu().numpy())
    return np.concatenate(rows, axis=0)
