# Reproducibility Guide

## Exact setup
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements-dev.txt`
3. `pip install -e .`

## Deterministic behavior
- Seed is controlled by `run.random_seed`.
- Config snapshot is stored at `outputs/<scenario>/config.snapshot.json`.
- Environment snapshot is stored at `outputs/<scenario>/environment.snapshot.json`.
- Git commit hash is stored in `outputs/<scenario>/run_metadata.json` when available.

## Run commands
- Single scenario:
  - `python -m mirror run-scenario --train-dir ".../scenario - train" --eval-dir ".../scenario - eval" --name scenario`
- All scenarios:
  - `python -m mirror run-all --root-dir data --output-root outputs`

## Hidden-eval safety
By default (`run.allow_eval_to_train_memory: false`), eval artifacts do not feed train-derived memory.

## Observability
If Langfuse env vars are set, run IDs are propagated and persisted into artifacts (`traces.json`).
