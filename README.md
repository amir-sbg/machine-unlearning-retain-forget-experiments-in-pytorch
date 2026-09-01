# Machine Unlearning Lab - Retain–Forget Experiments in PyTorch

Small PyTorch project for experimenting with machine unlearning. The goal is to compare what a model knows before and after removing a target class from training influence, while keeping an exact retrain baseline as the reference point.

This project uses the scikit-learn digits dataset instead of a large LLM because the full unlearning loop can run cheaply on CPU. That matters here: exact retraining is expensive for large models, but on a small dataset it gives a clean baseline for checking whether faster unlearning methods are actually behaving well. The same experiment pattern maps to larger models: define retain/forget data, train the original model, retrain without the forget data, apply cheaper unlearning, then compare utility and forgetting metrics.

## What is implemented

- deterministic retain/forget/validation/test splits
- a PyTorch MLP classifier trained on all digit classes
- exact retraining on retain data only
- retain-only fine-tuning from the original model
- negative-gradient unlearning with retain loss mixed in
- last-layer reset for the forgotten class
- output-head dampening as a cheap intermediate scrub baseline
- metrics for retain accuracy, forget accuracy, forget-class confidence, and distance from exact retraining
- class-wise test metrics to check collateral damage on retained classes
- threshold curves showing how many forget examples remain high-confidence
- membership-style forget diagnostics comparing train-forget confidence/loss against held-out forget examples
- Jensen-Shannon probability drift against the exact retrain baseline
- scorecard runtime speedup against exact retraining
- a small notebook for reviewing the generated result table

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
python -m unlearning_lab.experiment
```

For a quick smoke run:

```bash
python -m unlearning_lab.experiment \
  --forget-class 8 \
  --epochs 5 \
  --unlearn-steps 10 \
  --hidden-dim 96 \
  --dropout 0.1
```

## Outputs

```text
reports/
├── classwise_metrics.json
├── data_summary.json
├── experiment_summary.json
├── forget_confidence_curves.json
├── membership_signals.json
├── method_metrics.csv
├── method_metrics.json
├── method_scorecard.csv
├── probability_drift.json
├── retrain_gaps.json
├── unlearning_tradeoff.png
└── *_history.csv

artifacts/
├── full_model.pt
└── exact_retrain.pt
```

`method_metrics.csv` is the detailed output. `method_scorecard.csv` is the compact ranking table. A good unlearning method should reduce confidence on the forgotten class while keeping retain-class accuracy close to the exact retrain baseline.

## Project structure

```text
src/unlearning_lab/
├── data.py        # retain/forget split on the digits dataset
├── model.py       # small PyTorch classifier
├── train.py       # training loop and prediction helpers
├── unlearn.py     # cheap unlearning methods
├── metrics.py     # retain/forget metrics and retrain gaps
└── experiment.py  # command-line experiment runner

notebooks/
└── 00_review_unlearning_results.ipynb
```
