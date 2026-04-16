# Reply Mirror Fraud Challenge — Competition-Ready Multi-Agent System

## Quickstart
```bash
make install
python -m mirror run-scenario --train-dir "data/My Scenario - train" --eval-dir "data/My Scenario - eval" --name my_scenario --output-dir outputs
```

## Architecture (text diagram)
1. **Load + Validate I/O** (`transactions.csv` required; optional users/locations/sms/mails/audio).
2. **Entity Linking + Features** (shared transaction graph/temporal/comms/geo features).
3. **Agent Layer**
   - ProfilerAgent
   - TemporalBehaviorAgent
   - NetworkRiskAgent
   - GeoRiskAgent
   - CommsRiskAgent
   - RuleSynthesisAgent
   - PatternMemoryAgent
   - CaseManagerAgent
   - DecisionAgent
4. **Adaptive Thresholding + Submission Safety Checks**
5. **Artifacts** (submission, report, diagnostics, traces, reproducibility snapshots, experiment registry)

## Agents and memory
- Each risk agent writes evidence (`score`, `confidence`, `reasons`).
- CaseManager merges evidence into case bundles.
- DecisionAgent performs weighted ensemble arbitration.
- Memory is persisted in `outputs/<scenario>/memory/`.
- Eval leakage into train memory is blocked unless `run.allow_eval_to_train_memory=true`.

## LLM constraints
- LLM usage is optional and bounded by config (`max_llm_calls_per_run`, selective comms review).
- Caching is enabled in `llm_cache/`.
- No per-transaction LLM calls.

## Running modes
- Cheap: `--config configs/cheap.yaml`
- Default: `--config configs/default.yaml`
- Strong: `--config configs/strong.yaml`

## Train-only backtest
```bash
python -m mirror backtest --train-dir "data/My Scenario - train" --output-dir outputs/backtest
```

## Train + eval inference + submission
```bash
python -m mirror run-scenario --train-dir "data/My Scenario - train" --eval-dir "data/My Scenario - eval" --name my_scenario
python -m mirror make-submission --train-dir "data/My Scenario - train" --eval-dir "data/My Scenario - eval" --output outputs/my_scenario/submission.txt
```

## Run all scenarios under a root
```bash
python -m mirror run-all --root-dir data --output-root outputs
```

## Ablation comparison
```bash
python scripts/run_ablations.py --train-dir "...train" --eval-dir "...eval" --scenario my_scenario --output-root outputs/ablations
```
Produces `ablation_results.csv`.

## Disable expensive components
- Disable LLM: `run.disable_llm: true`
- Disable comms agent: `agents.disable: ["CommsRiskAgent"]`
- Disable geo agent: `agents.disable: ["GeoRiskAgent"]`
- Disable graph agent: `agents.disable: ["NetworkRiskAgent"]`
- Disable audio: `run.disable_audio: true`

## OpenRouter and Langfuse
Set `.env` from `.env.example`:
- `OPENROUTER_API_KEY`, `MODEL_FAST`, `MODEL_STRONG`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

## Hidden-eval readiness
- No hardcoded scenario names in runtime logic.
- Scenario discovery is schema-based (`* - train` / `* - eval` folders with `transactions.csv`).
- Supports re-evaluation on unseen folders with the same schema.

## Optional audio dependency
Install `pip install -e '.[audio]'` for Whisper-based transcription.
If unavailable, pipeline degrades safely without failing.
