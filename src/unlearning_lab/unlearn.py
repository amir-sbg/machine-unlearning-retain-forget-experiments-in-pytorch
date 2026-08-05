from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass

import torch
from torch import nn

from .data import Split
from .train import TrainingConfig, make_loader, train_classifier


@dataclass(frozen=True)
class UnlearnConfig:
    steps: int = 80
    batch_size: int = 64
    learning_rate: float = 5e-4
    retain_weight: float = 1.0
    forget_weight: float = 0.20
    gradient_clip: float = 1.0
    seed: int = 42


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def retain_finetune(
    model: nn.Module,
    retain_split: Split,
    validation_split: Split,
    config: TrainingConfig,
    device: torch.device,
) -> nn.Module:
    result = train_classifier(
        clone_model(model),
        retain_split,
        validation_split,
        config,
        device,
    )
    return result.model


def _last_linear_layer(model: nn.Module) -> nn.Linear:
    layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if not layers:
        raise ValueError("model does not contain a linear layer")
    return layers[-1]


def reset_output_class(
    model: nn.Module,
    class_id: int,
    seed: int = 42,
) -> nn.Module:
    scrubbed = clone_model(model)
    layer = _last_linear_layer(scrubbed)
    if not 0 <= class_id < layer.out_features:
        raise ValueError("class_id is outside the output layer")

    generator = torch.Generator(device=layer.weight.device).manual_seed(seed)
    with torch.no_grad():
        scale = 1.0 / max(layer.in_features, 1) ** 0.5
        layer.weight[class_id].uniform_(-scale, scale, generator=generator)
        if layer.bias is not None:
            layer.bias[class_id].zero_()
    return scrubbed


def negative_gradient_unlearn(
    model: nn.Module,
    retain_split: Split,
    forget_split: Split,
    validation_split: Split,
    config: UnlearnConfig,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    if config.steps < 1 or config.batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if config.learning_rate <= 0 or config.gradient_clip <= 0:
        raise ValueError("learning_rate and gradient_clip must be positive")
    if config.retain_weight < 0 or config.forget_weight <= 0:
        raise ValueError("retain_weight must be non-negative and forget_weight must be positive")

    torch.manual_seed(config.seed)
    unlearned = clone_model(model).to(device)
    optimizer = torch.optim.AdamW(unlearned.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    retain_iter = itertools.cycle(
        make_loader(retain_split, config.batch_size, shuffle=True, seed=config.seed)
    )
    forget_iter = itertools.cycle(
        make_loader(forget_split, config.batch_size, shuffle=True, seed=config.seed + 1)
    )
    validation_loader = make_loader(
        validation_split,
        config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    history: list[dict[str, float]] = []
    for step in range(1, config.steps + 1):
        retain_features, retain_labels = next(retain_iter)
        forget_features, forget_labels = next(forget_iter)
        retain_features = retain_features.to(device)
        retain_labels = retain_labels.to(device)
        forget_features = forget_features.to(device)
        forget_labels = forget_labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        retain_loss = loss_fn(unlearned(retain_features), retain_labels)
        forget_loss = loss_fn(unlearned(forget_features), forget_labels)
        loss = config.retain_weight * retain_loss - config.forget_weight * forget_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            unlearned.parameters(),
            config.gradient_clip,
        )
        optimizer.step()

        if step == 1 or step == config.steps or step % 10 == 0:
            # The validation pass is deliberately sparse; it is a cheap drift check, not training.
            validation_losses = []
            unlearned.eval()
            with torch.inference_mode():
                for features, labels in validation_loader:
                    features = features.to(device)
                    labels = labels.to(device)
                    validation_losses.append(float(loss_fn(unlearned(features), labels)))
            unlearned.train()
            history.append(
                {
                    "step": float(step),
                    "retain_loss": float(retain_loss.detach()),
                    "forget_loss": float(forget_loss.detach()),
                    "objective": float(loss.detach()),
                    "grad_norm": float(grad_norm),
                    "validation_loss": float(sum(validation_losses) / len(validation_losses)),
                }
            )

    return unlearned, history
