# Reply Mirror Fraud Challenge — Agent-Based Competition System

Production-minded, **runnable** Python repository for the Reply Mirror fraud-detection challenge.

## Multi-agent architecture
The pipeline remains explicitly agent-based. Each agent contributes evidence with score + confidence:

- **ProfilerAgent**: scenario + modality profiling
- **TemporalBehaviorAgent**: rolling/decay user baselines, novelty, drift, bursts, sequence anomalies
- **NetworkRiskAgent**: first-seen edges, fan-in/out anomalies, concentration spikes, shared recipients
- **GeoRiskAgent**: mobility drift, impossible-travel heuristics, in-person consistency checks, weak-evidence penalties
- **CommsRiskAgent**: cheap regex layer + selective structured LLM JSON analysis
- **RuleSynthesisAgent**: selective cluster-level LLM synthesis
- **PatternMemoryAgent**: motif extraction, persistence (`patterns.json`), and motif-based rescoring
- **CaseManagerAgent**: evidence fusion into transaction case bundles
- **DecisionAgent**: explainable weighted ensemble with configurable conservative/aggressive mode

Design goals: evolving fraud resilience, hidden-set generalization, controlled false positives, and cost-aware selective LLM usage.

## Cost-aware LLM strategy
- LLM calls are optional and capped via config:
  - `run.max_llm_calls_per_run`
  - `run.max_strong_model_calls`
  - `run.max_messages_for_llm_review`
- Comms LLM runs only on suspicious clusters/users.
- Borderline arbitration supports strong model escalation on a small budget.
- All completions are cached on disk (`output/llm_cache`).
- Token/call usage is written to diagnostics.

## Repository layout

```text
mirror/
  data/            # loaders, normalization, linking
  features/        # transaction/user/geo/comms/graph features
  agents/          # concrete agent implementations + evidence
  llm/             # OpenRouter wrapper + prompts
  memory/          # persistent memory per scenario output
  orchestration/   # agent sequencing
  calibration/     # unsupervised stress + thresholding
  submissions/     # valid txt output writer
  evaluation/      # diagnostics and contribution summaries
  cli.py           # Typer CLI
  pipeline.py      # run_pipeline entrypoint
configs/
  default.yaml
tests/
.env.example
pyproject.toml
```

## Installation

```bash
uv sync
# or with pip:
pip install -e .
```

Optional extras:
```bash
pip install -e '.[dev,observability]'
```

## Environment variables
Copy `.env.example` and set keys as needed:

- `OPENROUTER_API_KEY`
- `MODEL_FAST`
- `MODEL_STRONG`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST`

No secrets are committed.

## Commands

Run per scenario:

```bash
python -m mirror run --input-dir ".../Deus Ex - train" --output-dir outputs/deus_ex
python -m mirror run --input-dir ".../Brave New World - train" --output-dir outputs/brave_new_world
python -m mirror run --input-dir ".../The Truman Show - train" --output-dir outputs/truman_show

# Batch mode (auto-discovers scenario folders with transactions.csv)
python -m mirror run-all --data-root ".../train" --output-root outputs
```

Inspect diagnostics:
```bash
python -m mirror inspect --input-dir ".../Deus Ex - train"
```

Backtest (time-aware unsupervised thresholding):
```bash
python -m mirror backtest --input-dir ".../Deus Ex - train" --output-dir outputs/deus_ex_backtest
```

Create challenge submission file:
```bash
python -m mirror make-submission --input-dir ".../Deus Ex - train" --output outputs/deus_ex_submission.txt
```

## Input modalities
Expected in each scenario folder:
- `transactions.csv` (required)
- `users.json` (optional)
- `locations.json` (optional)
- `sms.json` (optional)
- `mails.json` (optional)
- `audio/*.mp3` (optional; disabled by default)

The pipeline degrades gracefully if optional files are absent.

## Output artifacts
For each run output folder:
- `submission.txt` (ASCII transaction IDs, one per line)
- `cases.parquet` (machine-readable flagged case records)
- `diagnostics.json`
- `patterns.json` (persisted motif memory + pattern cards)
- `traces.json` (agent run trace + budget summary)
- `memory/memory.json`
- `memory/decisions.parquet`
- `llm_cache/*.json` (when LLM is used)

## Reproducibility
- Deterministic seed (`run.random_seed`)
- Config-driven thresholds and suspect-rate guardrails
- Cached LLM outputs
- Persisted memory and decision parquet snapshots

## Preparing for hidden eval sets
Use identical command structure with the eval directory matching scenario format. No code changes needed.

## Controls
- Disable LLM: `run.disable_llm: true`
- Disable audio: `run.disable_audio: true`
- Fast mode: `run.fast_mode: true`
- Audio transcription (optional): `run.transcribe_audio: true` (gracefully degrades if backend unavailable)

Both are disabled safely if keys/files are missing.
