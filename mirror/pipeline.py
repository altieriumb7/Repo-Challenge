from __future__ import annotations

import json
from pathlib import Path

from mirror.calibration.thresholds import apply_threshold, choose_threshold, synthetic_stress
from mirror.data.linking import link_entities
from mirror.data.loaders import load_modalities
from mirror.evaluation.diagnostics import summarize
from mirror.features.builders import build_feature_matrix
from mirror.llm.provider import OpenRouterClient
from mirror.memory.store import MemoryStore
from mirror.orchestration.orchestrator import Orchestrator
from mirror.submissions.writer import write_submission
from mirror.types import PipelineContext


def run_pipeline(input_dir: str, output_dir: str, config: dict) -> dict:
    scenario = Path(input_dir).name
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    data = load_modalities(input_dir, audio_enabled=config.get("run", {}).get("audio_enabled", False))
    data["linked_transactions"] = link_entities(data)

    llm_client = OpenRouterClient(cache_dir=out_root / "llm_cache", timeout_seconds=config.get("llm", {}).get("timeout_seconds", 20), max_retries=config.get("llm", {}).get("max_retries", 2))
    data["llm_client"] = llm_client

    features = {"matrix": build_feature_matrix(data)}

    ctx = PipelineContext(scenario_name=scenario, input_dir=input_dir, output_dir=output_dir, config=config, data=data, features=features)
    ctx = Orchestrator().run(ctx)

    decisions = synthetic_stress(ctx.agent_outputs["DecisionAgent"].evidence, seed=config.get("run", {}).get("random_seed", 42))
    threshold_cfg = config.get("thresholding", {})
    threshold = choose_threshold(
        decisions["score"],
        min_frac=threshold_cfg.get("min_suspect_fraction", 0.005),
        max_frac=threshold_cfg.get("max_suspect_fraction", 0.2),
        target_frac=threshold_cfg.get("target_suspect_fraction", 0.04),
    )
    decisions = apply_threshold(decisions, threshold)

    submission = write_submission(
        decisions,
        out_root / config.get("run", {}).get("output_submission_name", "submission.txt"),
        min_frac=threshold_cfg.get("min_suspect_fraction", 0.005),
        max_frac=threshold_cfg.get("max_suspect_fraction", 0.2),
    )

    memory = MemoryStore(out_root / "memory")
    risky = decisions.loc[decisions["is_suspect"], "transaction_id"].astype(str).tolist()
    memory.update("suspicious_entities", risky)
    memory.save()
    memory.save_frame("decisions", decisions)

    diag = summarize(decisions, ctx.agent_outputs)
    diag["llm_usage"] = llm_client.usage
    diag["threshold"] = threshold
    (out_root / "diagnostics.json").write_text(json.dumps(diag, indent=2, ensure_ascii=True), encoding="utf-8")

    return {"submission": str(submission), "diagnostics": diag}
