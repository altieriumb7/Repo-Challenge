from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from mirror.agents.agents import (
    CaseManagerAgent,
    CommsRiskAgent,
    DecisionAgent,
    GeoRiskAgent,
    NetworkRiskAgent,
    PatternMemoryAgent,
    ProfilerAgent,
    RuleSynthesisAgent,
    TemporalBehaviorAgent,
)
from mirror.types import AgentResult, PipelineContext


@dataclass(frozen=True)
class _Stage:
    name: str
    agents: tuple
    allow_parallel: bool = True


class Orchestrator:
    def __init__(self):
        self.stages = [
            _Stage("stage_0_profile", (ProfilerAgent(),), allow_parallel=False),
            _Stage("stage_1_risk", (TemporalBehaviorAgent(), NetworkRiskAgent(), GeoRiskAgent(), CommsRiskAgent())),
            _Stage("stage_2_synthesis", (RuleSynthesisAgent(), PatternMemoryAgent())),
            _Stage("stage_3_decision", (CaseManagerAgent(), DecisionAgent()), allow_parallel=False),
        ]

    def _readonly_context(self, ctx: PipelineContext) -> PipelineContext:
        return PipelineContext(
            scenario_name=ctx.scenario_name,
            input_dir=ctx.input_dir,
            output_dir=ctx.output_dir,
            config=MappingProxyType(dict(ctx.config)),
            data=MappingProxyType(dict(ctx.data)),
            features=MappingProxyType(dict(ctx.features)),
            agent_outputs=MappingProxyType(dict(ctx.agent_outputs)),
            diagnostics=MappingProxyType(dict(ctx.diagnostics)),
        )

    def _prepare_shared_artifacts(self, ctx: PipelineContext) -> None:
        linked = ctx.data.get("linked_transactions", pd.DataFrame())
        if not linked.empty:
            tx = linked.sort_values("event_time").copy()
            ctx.data["shared_transactions"] = tx
            if "sender_id" in tx.columns:
                ctx.data["shared_sender_counts"] = tx.groupby("sender_id").size().to_dict()
            if {"sender_id", "recipient_id"}.issubset(tx.columns):
                ctx.data["shared_linked_entities"] = tx[["sender_id", "recipient_id"]].drop_duplicates().reset_index(drop=True)
            if {"event_time", "transaction_id"}.issubset(tx.columns):
                ctx.data["shared_time_features"] = tx[["transaction_id", "event_time"]].copy()
        sms = ctx.data.get("sms", pd.DataFrame())
        mails = ctx.data.get("mails", pd.DataFrame())
        audio = ctx.data.get("audio", pd.DataFrame())
        if not sms.empty or not mails.empty or not audio.empty:
            ctx.data["shared_comms_modalities"] = {"sms_rows": len(sms), "mail_rows": len(mails), "audio_rows": len(audio)}

    def _run_one(self, agent, ctx: PipelineContext) -> tuple[AgentResult, float]:
        started = time.perf_counter()
        result = agent.run(ctx)
        return result, time.perf_counter() - started

    def run(self, ctx: PipelineContext) -> PipelineContext:
        run_cfg = ctx.config.get("run", {})
        disabled = set(ctx.config.get("agents", {}).get("disable", []))
        parallel_agents = bool(run_cfg.get("parallel_agents", True))
        max_workers = int(run_cfg.get("max_agent_workers", 4))
        fail_fast = bool(run_cfg.get("fail_fast_parallel", True))

        self._prepare_shared_artifacts(ctx)

        agent_times: dict[str, float] = {}
        agent_mode: dict[str, str] = {}
        stage_times: dict[str, float] = {}
        stage_errors: dict[str, str] = {}
        failed_agents: set[str] = set()

        for stage in self.stages:
            stage_start = time.perf_counter()
            stage_agents = [a for a in stage.agents if a.name not in disabled]
            ctx.diagnostics.setdefault("trace", []).append(
                {
                    "event": "stage_start",
                    "stage": stage.name,
                    "agents": [a.name for a in stage_agents],
                    "parallel": bool(parallel_agents and stage.allow_parallel and len(stage_agents) > 1),
                }
            )
            if stage.name == "stage_3_decision" and failed_agents:
                raise RuntimeError(f"Cannot execute decision stage, upstream failures: {sorted(failed_agents)}")
            if not stage_agents:
                stage_times[stage.name] = time.perf_counter() - stage_start
                continue

            if parallel_agents and stage.allow_parallel and len(stage_agents) > 1:
                stage_results: dict[str, AgentResult] = {}
                with ThreadPoolExecutor(max_workers=min(max_workers, len(stage_agents))) as pool:
                    fut_map = {pool.submit(self._run_one, agent, self._readonly_context(ctx)): agent for agent in stage_agents}
                    for fut in as_completed(fut_map):
                        agent = fut_map[fut]
                        try:
                            result, elapsed = fut.result()
                            stage_results[agent.name] = result
                            agent_times[agent.name] = elapsed
                            agent_mode[agent.name] = "parallel"
                        except Exception as e:  # noqa: BLE001
                            failed_agents.add(agent.name)
                            stage_errors[agent.name] = str(e)
                            if fail_fast:
                                raise RuntimeError(f"Parallel stage failure in {agent.name}: {e}") from e
                for agent in stage_agents:
                    if agent.name in stage_results:
                        ctx.agent_outputs[agent.name] = stage_results[agent.name]
            else:
                for agent in stage_agents:
                    try:
                        result, elapsed = self._run_one(agent, self._readonly_context(ctx))
                        ctx.agent_outputs[result.name] = result
                        agent_times[agent.name] = elapsed
                        agent_mode[agent.name] = "serial"
                    except Exception as e:  # noqa: BLE001
                        failed_agents.add(agent.name)
                        stage_errors[agent.name] = str(e)
                        if fail_fast:
                            raise RuntimeError(f"Agent failure in {agent.name}: {e}") from e

            stage_times[stage.name] = time.perf_counter() - stage_start
            ctx.diagnostics.setdefault("trace", []).append(
                {
                    "event": "stage_end",
                    "stage": stage.name,
                    "elapsed_seconds": stage_times[stage.name],
                    "failures": sorted([a for a in failed_agents if a in {x.name for x in stage_agents}]),
                }
            )

        ordered_outputs: dict[str, AgentResult] = {}
        for stage in self.stages:
            for agent in stage.agents:
                if agent.name in ctx.agent_outputs:
                    ordered_outputs[agent.name] = ctx.agent_outputs[agent.name]
        ctx.agent_outputs = ordered_outputs
        ctx.diagnostics["orchestrator"] = {
            "parallel_agents": parallel_agents,
            "max_agent_workers": max_workers,
            "fail_fast_parallel": fail_fast,
            "stage_runtimes_seconds": stage_times,
            "agent_runtimes_seconds": agent_times,
            "agent_execution_mode": agent_mode,
            "failed_agents": sorted(failed_agents),
            "errors": stage_errors,
        }
        return ctx
