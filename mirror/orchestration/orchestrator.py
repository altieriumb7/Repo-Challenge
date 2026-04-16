from __future__ import annotations

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
from mirror.types import PipelineContext


class Orchestrator:
    def __init__(self):
        self.agents = [
            ProfilerAgent(),
            TemporalBehaviorAgent(),
            NetworkRiskAgent(),
            GeoRiskAgent(),
            CommsRiskAgent(),
            RuleSynthesisAgent(),
            PatternMemoryAgent(),
            CaseManagerAgent(),
            DecisionAgent(),
        ]

    def run(self, ctx: PipelineContext) -> PipelineContext:
        disabled = set(ctx.config.get("agents", {}).get("disable", []))
        for agent in self.agents:
            if agent.name in disabled:
                continue
            result = agent.run(ctx)
            ctx.agent_outputs[result.name] = result
        return ctx
