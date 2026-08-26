from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import Split, dataset_summary, load_unlearning_data
from .metrics import (
    compare_to_exact_retrain,
    evaluate_unlearning_model,
    membership_signal_summary,
    method_scorecard,
    pareto_frontier,
    save_json,
)
from .model import DigitMLP, count_parameters
from .train import TrainingConfig, predict_logits, set_seed, train_classifier
from .unlearn import (
    UnlearnConfig,
    negative_gradient_unlearn,
    reset_output_class,
    retain_finetune,
)


@dataclass(frozen=True)
class ExperimentConfig:
    forget_class: int = 8
    seed: int = 42
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    unlearn_steps: int = 80
    hidden_dim: int = 128
    dropout: float = 0.10
    output_dir: Path = Path("artifacts")
    report_dir: Path = Path("reports")
    device: str = "auto"


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def join_splits(first: Split, second: Split) -> Split:
    return Split(
        features=np.concatenate([first.features, second.features], axis=0),
        labels=np.concatenate([first.labels, second.labels], axis=0),
    )


def select_class(split: Split, class_id: int) -> Split:
    mask = split.labels == class_id
    if not np.any(mask):
        raise ValueError(f"split does not contain class {class_id}")
    return Split(features=split.features[mask], labels=split.labels[mask])


def _training_config(config: ExperimentConfig, epochs: int | None = None) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs or config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        patience=max(5, config.epochs // 4),
        seed=config.seed,
    )


def _new_model(config: ExperimentConfig) -> DigitMLP:
    if config.hidden_dim < 16:
        raise ValueError("hidden_dim must be at least 16")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    set_seed(config.seed)
    return DigitMLP(hidden_dim=config.hidden_dim, dropout=config.dropout)


def _flatten_metrics(metrics: dict[str, dict]) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {}
    for split_name, values in metrics.items():
        for key, value in values.items():
            row[f"{split_name}_{key}"] = value
    return row


def save_tradeoff_plot(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(
        frame["test_forget_confidence"],
        frame["test_retain_accuracy"],
        s=90,
    )
    for row in frame.to_dict("records"):
        axis.annotate(
            row["method"],
            (row["test_forget_confidence"], row["test_retain_accuracy"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xlabel("Forget-class confidence on test forget examples")
    axis.set_ylabel("Retain-class accuracy on test retain examples")
    axis.set_title("Machine unlearning tradeoff")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_experiment(config: ExperimentConfig) -> dict:
    device = choose_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    data = load_unlearning_data(forget_class=config.forget_class, seed=config.seed)
    full_train = join_splits(data.train_retain, data.train_forget)
    histories: dict[str, list[dict[str, float]]] = {}
    timings: dict[str, float] = {}
    models: dict[str, torch.nn.Module] = {}

    start = time.perf_counter()
    full_result = train_classifier(
        _new_model(config),
        full_train,
        data.validation,
        _training_config(config),
        device,
    )
    timings["full_model"] = time.perf_counter() - start
    histories["full_model"] = full_result.history
    models["full_model"] = full_result.model

    # Exact retrain is not the cheap method; it is the reference point for the cheaper ones.
    start = time.perf_counter()
    retrain_result = train_classifier(
        _new_model(config),
        data.train_retain,
        data.validation,
        _training_config(config),
        device,
    )
    timings["exact_retrain"] = time.perf_counter() - start
    histories["exact_retrain"] = retrain_result.history
    models["exact_retrain"] = retrain_result.model

    start = time.perf_counter()
    models["retain_finetune"] = retain_finetune(
        models["full_model"],
        data.train_retain,
        data.validation,
        _training_config(config, epochs=max(5, config.epochs // 3)),
        device,
    )
    timings["retain_finetune"] = time.perf_counter() - start

    start = time.perf_counter()
    models["head_reset"] = reset_output_class(
        models["full_model"],
        class_id=config.forget_class,
        seed=config.seed,
    )
    timings["head_reset"] = time.perf_counter() - start

    start = time.perf_counter()
    negative_model, negative_history = negative_gradient_unlearn(
        models["full_model"],
        data.train_retain,
        data.train_forget,
        data.validation,
        UnlearnConfig(
            steps=config.unlearn_steps,
            batch_size=config.batch_size,
            seed=config.seed,
        ),
        device,
    )
    timings["negative_gradient"] = time.perf_counter() - start
    histories["negative_gradient"] = negative_history
    models["negative_gradient"] = negative_model

    method_metrics = {
        name: evaluate_unlearning_model(model, data, device)
        for name, model in models.items()
    }
    test_forget = select_class(data.test, data.forget_class)
    membership_signals = {
        name: membership_signal_summary(
            data.train_forget.labels,
            predict_logits(model, data.train_forget, device),
            test_forget.labels,
            predict_logits(model, test_forget, device),
            data.forget_class,
        )
        for name, model in models.items()
    }
    retrain_gaps = compare_to_exact_retrain(method_metrics)
    scorecard = method_scorecard(method_metrics, timings)
    frontier = pareto_frontier(scorecard)
    metrics_frame = pd.DataFrame(
        [
            {
                "method": name,
                "runtime_seconds": timings.get(name, 0.0),
                **_flatten_metrics(metrics),
                **retrain_gaps[name],
                **membership_signals[name],
            }
            for name, metrics in method_metrics.items()
        ]
    )

    metrics_frame.to_csv(config.report_dir / "method_metrics.csv", index=False)
    pd.DataFrame(scorecard).to_csv(config.report_dir / "method_scorecard.csv", index=False)
    pd.DataFrame(frontier).to_csv(config.report_dir / "method_frontier.csv", index=False)
    save_tradeoff_plot(metrics_frame, config.report_dir / "unlearning_tradeoff.png")
    save_json(method_metrics, config.report_dir / "method_metrics.json")
    save_json(membership_signals, config.report_dir / "membership_signals.json")
    save_json(retrain_gaps, config.report_dir / "retrain_gaps.json")
    save_json({"methods": scorecard}, config.report_dir / "method_scorecard.json")
    save_json({"methods": frontier}, config.report_dir / "method_frontier.json")
    save_json(dataset_summary(data), config.report_dir / "data_summary.json")
    save_json(
        {
            "config": {**asdict(config), "output_dir": str(config.output_dir), "report_dir": str(config.report_dir)},
            "device": str(device),
            "parameter_count": count_parameters(models["full_model"]),
            "model": {
                "hidden_dim": config.hidden_dim,
                "dropout": config.dropout,
            },
            "methods": list(models),
            "timings": timings,
            "best_method_by_retrain_gap": scorecard[0]["method"],
            "pareto_frontier": [row["method"] for row in frontier],
            "lowest_membership_signal": min(
                membership_signals,
                key=lambda name: membership_signals[name]["membership_signal"],
            ),
        },
        config.report_dir / "experiment_summary.json",
    )

    for name, history in histories.items():
        pd.DataFrame(history).to_csv(
            config.report_dir / f"{name}_history.csv",
            index=False,
        )
    torch.save(
        {
            "model_state_dict": models["full_model"].state_dict(),
            "config": asdict(config),
        },
        config.output_dir / "full_model.pt",
    )
    torch.save(
        {
            "model_state_dict": models["exact_retrain"].state_dict(),
            "config": asdict(config),
        },
        config.output_dir / "exact_retrain.pt",
    )

    return {
        "method_metrics": method_metrics,
        "membership_signals": membership_signals,
        "retrain_gaps": retrain_gaps,
        "scorecard": scorecard,
        "pareto_frontier": frontier,
        "summary_path": str(config.report_dir / "experiment_summary.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small machine-unlearning experiment.")
    parser.add_argument("--forget-class", type=int, default=ExperimentConfig.forget_class)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=ExperimentConfig.learning_rate)
    parser.add_argument("--unlearn-steps", type=int, default=ExperimentConfig.unlearn_steps)
    parser.add_argument("--hidden-dim", type=int, default=ExperimentConfig.hidden_dim)
    parser.add_argument("--dropout", type=float, default=ExperimentConfig.dropout)
    parser.add_argument("--output-dir", type=Path, default=ExperimentConfig.output_dir)
    parser.add_argument("--report-dir", type=Path, default=ExperimentConfig.report_dir)
    parser.add_argument("--device", default=ExperimentConfig.device)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(**vars(args))


def main() -> None:
    result = run_experiment(config_from_args(build_parser().parse_args()))
    print(json.dumps(result["retrain_gaps"], indent=2))


if __name__ == "__main__":
    main()
