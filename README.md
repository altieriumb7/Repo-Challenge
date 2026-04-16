# Reply Mirror Fraud Challenge — Agent-Based Baseline

Production-minded, **runnable** Python repository for the Reply Mirror fraud-detection challenge.

## Why this is agent-based (not static scoring)
The pipeline is built as a multi-agent system where each agent has explicit responsibility and contributes evidence:

- **ProfilerAgent**: scenario + modality profiling
- **TemporalBehaviorAgent**: per-user behavioral drift and burst/off-hour anomalies
- **NetworkRiskAgent**: sender/recipient graph novelty and degree outliers
- **GeoRiskAgent**: location consistency risk
- **CommsRiskAgent**: heuristic phishing/urgency indicators from SMS/email
- **RuleSynthesisAgent**: selective LLM synthesis over high-risk clusters (never one-call-per-row)
- **CaseManagerAgent**: evidence fusion into transaction case bundles
- **DecisionAgent**: final risk score for calibration and thresholding

This design supports evolving fraud strategies by combining temporal, graph, geo, and comms signals with persistent memory.

## Cost-aware LLM strategy
- LLM calls are **optional** and gated by config + API key.
- LLM only runs for **cluster-level summarization**, not per transaction.
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
python -m mirror run --input-dir ".../The Truman Show - train" --output-dir outputs/truman
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
- `diagnostics.json`
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

## Disable modalities
- Disable LLM: `run.llm_enabled: false`
- Disable audio: `run.audio_enabled: false`

Both are disabled safely if keys/files are missing.
