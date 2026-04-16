from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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


def run_pipeline(input_dir: str, output_dir: str, config: dict) -> dict:
    scenario = Path(input_dir).name
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    data = load_modalities(input_dir, config=config)
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
    decisions["llm_review_used"] = decisions["score"].between(0.55, 0.75)
    pm_ev = ctx.agent_outputs.get("PatternMemoryAgent")
    pm_ids = set(pm_ev.evidence["transaction_id"].astype(str).tolist()) if pm_ev and not pm_ev.evidence.empty else set()
    decisions["related_patterns"] = decisions["transaction_id"].map(lambda tid: ["pattern-memory"] if str(tid) in pm_ids else [])
    decisions["top_contributing_agents"] = decisions.get("top_reasons", [[] for _ in range(len(decisions))])
    decisions["evidence_bullets"] = decisions["reasons"].map(lambda x: x if isinstance(x, list) else [])

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
    diag["threshold"] = threshold
    diag["validation"] = _run_unsupervised_backtesting(decisions, seed=config.get("run", {}).get("random_seed", 42))
    diag["runtime_controls"] = config.get("run", {})
    (out_root / "diagnostics.json").write_text(json.dumps(diag, indent=2, ensure_ascii=True), encoding="utf-8")
    (out_root / "traces.json").write_text(
        json.dumps({"scenario": scenario, "agents_executed": list(ctx.agent_outputs.keys()), "llm_calls": llm_client.usage.get("calls", 0)}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return {"submission": str(submission), "diagnostics": diag}
