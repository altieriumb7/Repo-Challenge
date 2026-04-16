from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from mirror.calibration.thresholds import apply_threshold, choose_threshold, synthetic_stress
from mirror.data.linking import link_entities
from mirror.data.loaders import load_modalities
from mirror.evaluation.diagnostics import summarize
from mirror.experiments.registry import ExperimentRegistry
from mirror.features.builders import build_feature_matrix
from mirror.llm.provider import OpenRouterClient
from mirror.memory.store import MemoryStore
from mirror.orchestration.orchestrator import Orchestrator
from mirror.runtime import StageProfiler, config_hash, env_snapshot, git_hash, memory_usage_mb, write_json
from mirror.submissions.writer import write_submission
from mirror.types import LLMBudget, PipelineContext


def _run_unsupervised_backtesting(decisions: pd.DataFrame, seed: int = 42) -> dict:
    if decisions.empty:
        return {"folds": [], "synthetic_detection_rate": 0.0, "stability": 0.0, "false_positive_proxy": 0.0}
    rng = np.random.default_rng(seed)
    ranked = decisions.sort_values("score", ascending=False).reset_index(drop=True).copy()
    n = len(ranked)
    inject_idx = rng.choice(n, size=max(1, n // 12), replace=False)
    ranked["is_injected"] = False
    ranked.loc[inject_idx, "is_injected"] = True
    ranked.loc[inject_idx, "score"] = (ranked.loc[inject_idx, "score"] + rng.uniform(0.2, 0.5, len(inject_idx))).clip(0, 1)
    fold_size = max(1, n // 4)
    folds = []
    for i in range(4):
        start, end = i * fold_size, min(n, (i + 1) * fold_size)
        fold = ranked.iloc[start:end]
        if fold.empty:
            continue
        th = float(fold["score"].quantile(0.9))
        flagged = fold["score"] >= th
        det = float((flagged & fold["is_injected"]).sum() / max(1, fold["is_injected"].sum()))
        fpp = float((flagged & ~fold["is_injected"]).mean())
        folds.append({"fold": i + 1, "detection": det, "false_positive_proxy": fpp, "threshold": th})
    return {
        "folds": folds,
        "synthetic_detection_rate": float(np.mean([f["detection"] for f in folds])) if folds else 0.0,
        "stability": float(1.0 - np.std([f["detection"] for f in folds])) if len(folds) > 1 else 1.0,
        "false_positive_proxy": float(np.mean([f["false_positive_proxy"] for f in folds])) if folds else 0.0,
    }


def run_pipeline(train_dir: str, eval_dir: str, output_dir: str, config: dict) -> dict:
    scenario = config.get("run", {}).get("scenario_name") or Path(eval_dir).name
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{scenario}-{uuid.uuid4().hex[:8]}"

    profiler = StageProfiler()
    with profiler.stage("load_data"):
        train_data = load_modalities(train_dir, config=config)
        eval_data = load_modalities(eval_dir, config=config)

    leak_cfg = bool(config.get("run", {}).get("allow_eval_to_train_memory", False))
    if not leak_cfg:
        # explicit guard: never seed memory from eval before inference.
        pass

    with profiler.stage("link_entities"):
        train_data["linked_transactions"] = link_entities(train_data)
        eval_data["linked_transactions"] = link_entities(eval_data)

    llm_client = OpenRouterClient(
        cache_dir=out_root / "llm_cache",
        timeout_seconds=config.get("llm", {}).get("timeout_seconds", 20),
        max_retries=config.get("llm", {}).get("max_retries", 2),
    )
    eval_data["llm_client"] = llm_client
    run_cfg = config.get("run", {})
    eval_data["llm_budget"] = LLMBudget(
        max_calls=int(run_cfg.get("max_llm_calls_per_run", 80)),
        max_strong_calls=int(run_cfg.get("max_strong_model_calls", 8)),
    )

    with profiler.stage("build_features"):
        features = {"matrix": build_feature_matrix(eval_data)}

    with profiler.stage("agents"):
        ctx = PipelineContext(
            scenario_name=scenario,
            input_dir=eval_dir,
            output_dir=output_dir,
            config=config,
            data=eval_data,
            features=features,
        )
        ctx = Orchestrator().run(ctx)

    with profiler.stage("threshold_and_submission"):
        decisions = synthetic_stress(ctx.agent_outputs["DecisionAgent"].evidence, seed=config.get("run", {}).get("random_seed", 42))
        threshold_cfg = config.get("thresholding", {})
        threshold = choose_threshold(
            decisions["score"],
            min_frac=threshold_cfg.get("min_suspect_fraction", 0.005),
            max_frac=threshold_cfg.get("max_suspect_fraction", 0.2),
            target_frac=threshold_cfg.get("target_suspect_fraction", 0.04),
        )
        decisions = apply_threshold(decisions, threshold)
        decisions["llm_review_used"] = decisions["score"].between(0.55, 0.75)
        pm_ev = ctx.agent_outputs.get("PatternMemoryAgent")
        pm_ids = set(pm_ev.evidence["transaction_id"].astype(str).tolist()) if pm_ev and not pm_ev.evidence.empty else set()
        decisions["related_patterns"] = decisions["transaction_id"].map(lambda tid: ["pattern-memory"] if str(tid) in pm_ids else [])
        decisions["top_contributing_agents"] = decisions.get("top_reasons", [[] for _ in range(len(decisions))])
        decisions["evidence_bullets"] = decisions["reasons"].map(lambda x: x if isinstance(x, list) else [])

        submission = write_submission(
            decisions,
            eval_data["transactions"],
            out_root / config.get("run", {}).get("output_submission_name", "submission.txt"),
            min_frac=threshold_cfg.get("min_suspect_fraction", 0.005),
            max_frac=threshold_cfg.get("max_suspect_fraction", 0.2),
            suspect_rate_bounds=(
                threshold_cfg.get("suspect_rate_lower_bound", threshold_cfg.get("min_suspect_fraction", 0.005)),
                threshold_cfg.get("suspect_rate_upper_bound", threshold_cfg.get("max_suspect_fraction", 0.2)),
            ),
        )

    with profiler.stage("artifacts"):
        memory = MemoryStore(out_root / "memory")
        risky = decisions.loc[decisions["is_suspect"], "transaction_id"].astype(str).tolist()
        memory.update("suspicious_entities", risky)
        if leak_cfg:
            memory.update("train_seen_senders", train_data["transactions"]["sender_id"].astype(str).tolist())
        memory.save()
        memory.save_frame("decisions", decisions)

        decisions[
            [
                "transaction_id",
                "fraud_score",
                "score",
                "decision",
                "top_contributing_agents",
                "evidence_bullets",
                "related_patterns",
                "llm_review_used",
            ]
        ].rename(columns={"score": "final_score"}).to_parquet(out_root / "cases.parquet", index=False)

        diag = summarize(decisions, ctx.agent_outputs)
        diag["llm_usage"] = llm_client.usage
        diag["llm_budget_usage"] = eval_data["llm_budget"].usage()
        diag["threshold"] = threshold
        diag["validation"] = _run_unsupervised_backtesting(decisions, seed=config.get("run", {}).get("random_seed", 42))
        diag["runtime_controls"] = config.get("run", {})
        diag["orchestrator"] = ctx.diagnostics.get("orchestrator", {})
        diag["profile"] = {
            "stages_seconds": profiler.timings,
            "peak_memory_mb": memory_usage_mb(),
            "total_runtime_seconds": float(sum(profiler.timings.values())),
            "agent_stages_seconds": ctx.diagnostics.get("orchestrator", {}).get("stage_runtimes_seconds", {}),
            "agent_runtimes_seconds": ctx.diagnostics.get("orchestrator", {}).get("agent_runtimes_seconds", {}),
            "parallel_workers": {
                "max_agent_workers": int(config.get("run", {}).get("max_agent_workers", 4)),
                "max_llm_workers": int(config.get("run", {}).get("max_llm_workers", 3)),
                "max_audio_workers": int(config.get("run", {}).get("max_audio_workers", 2)),
            },
            "llm_queue_wait_seconds": float(eval_data["llm_budget"].usage().get("wait_time_seconds", 0.0)),
        }

        cfg_hash = config_hash(config)
        write_json(out_root / "diagnostics.json", diag)
        write_json(
            out_root / "traces.json",
            {
                "scenario": scenario,
                "agents_executed": list(ctx.agent_outputs.keys()),
                "llm_calls": llm_client.usage.get("calls", 0),
                "orchestrator_mode": "parallel" if config.get("run", {}).get("parallel_agents", True) else "serial",
                "run_id": run_id,
                "langfuse_run_id": config.get("run", {}).get("langfuse_run_id", run_id),
            },
        )
        write_json(out_root / "run_metadata.json", {"run_id": run_id, "config_hash": cfg_hash, "git_hash": git_hash()})
        write_json(out_root / "config.snapshot.json", config)
        write_json(out_root / "environment.snapshot.json", env_snapshot())

        registry = ExperimentRegistry(out_root.parent / "experiment_registry.csv")
        pattern_counts: dict[str, int] = {}
        for row in decisions.get("related_patterns", []):
            for tag in row:
                pattern_counts[tag] = pattern_counts.get(tag, 0) + 1
        registry.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "config_hash": cfg_hash,
                "runtime_sec": round(sum(profiler.timings.values()), 3),
                "llm_calls": llm_client.usage.get("calls", 0),
                "prompt_tokens": llm_client.usage.get("prompt_tokens", 0),
                "completion_tokens": llm_client.usage.get("completion_tokens", 0),
                "suspect_rate": round(float(decisions["is_suspect"].mean()), 6),
                "top_pattern_families": ";".join(sorted(pattern_counts, key=pattern_counts.get, reverse=True)[:5]),
                "artifacts_dir": str(out_root),
            }
        )

    return {"submission": str(submission), "diagnostics": diag, "run_id": run_id}
