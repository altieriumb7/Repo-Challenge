from __future__ import annotations

import pandas as pd

from mirror.agents.base import Agent
from mirror.llm.prompts import PATTERN_SUMMARY_PROMPT
from mirror.types import AgentResult, PipelineContext


def _evidence_from_score(tx_ids: pd.Series, score: pd.Series, name: str, reason: str, confidence: float) -> AgentResult:
    ev = pd.DataFrame(
        {
            "transaction_id": tx_ids.astype(str),
            "score": score.astype(float).clip(0, 1),
            "confidence": confidence,
            "reasons": [[reason] for _ in range(len(score))],
        }
    )
    return AgentResult(name=name, evidence=ev)


class ProfilerAgent(Agent):
    name = "ProfilerAgent"

    def run(self, ctx: PipelineContext) -> AgentResult:
        tx = ctx.data["transactions"]
        missing = tx.isna().mean().to_dict()
        diagnostics = {
            "rows": len(tx),
            "time_span": [str(tx["event_time"].min()), str(tx["event_time"].max())],
            "modalities": {k: (not v.empty if hasattr(v, "empty") else bool(v)) for k, v in ctx.data.items() if k != "transactions"},
            "missingness": missing,
        }
        return AgentResult(name=self.name, evidence=pd.DataFrame(columns=["transaction_id", "score", "confidence", "reasons"]), diagnostics=diagnostics)


class TemporalBehaviorAgent(Agent):
    name = "TemporalBehaviorAgent"

    def run(self, ctx: PipelineContext) -> AgentResult:
        f = ctx.features["matrix"]
        score = (f["sender_amount_zproxy"].clip(lower=0) / 4 + f["burst_30m"] * 0.4 + f["off_hours"] * 0.2).clip(0, 1)
        return _evidence_from_score(f["transaction_id"], score, self.name, "temporal-drift", 0.78)


class NetworkRiskAgent(Agent):
    name = "NetworkRiskAgent"

    def run(self, ctx: PipelineContext) -> AgentResult:
        f = ctx.features["matrix"]
        novelty = f["edge_novelty"] * 0.4 + (f["sender_out_degree"] > 12).astype(float) * 0.4 + (f["recipient_in_degree"] > 30).astype(float) * 0.2
        return _evidence_from_score(f["transaction_id"], novelty.clip(0, 1), self.name, "network-anomaly", 0.72)


class GeoRiskAgent(Agent):
    name = "GeoRiskAgent"

    def run(self, ctx: PipelineContext) -> AgentResult:
        f = ctx.features["matrix"]
        return _evidence_from_score(f["transaction_id"], f["geo_risk"].clip(0, 1), self.name, "geo-inconsistency", 0.65)


class CommsRiskAgent(Agent):
    name = "CommsRiskAgent"

    def run(self, ctx: PipelineContext) -> AgentResult:
        f = ctx.features["matrix"]
        score = (f["comms_score"] / 3).clip(0, 1)
        return _evidence_from_score(f["transaction_id"], score, self.name, "phishing-signals", 0.69)


class RuleSynthesisAgent(Agent):
    name = "RuleSynthesisAgent"

    def run(self, ctx: PipelineContext) -> AgentResult:
        f = ctx.features["matrix"]
        suspicious = f.nlargest(30, "sender_amount_zproxy")
        cluster_dump = suspicious[["transaction_id", "sender_id", "recipient_id", "amount", "off_hours", "edge_novelty"]].to_dict("records")
        llm = ctx.data.get("llm_client")
        summary = "LLM disabled"
        if llm and ctx.config.get("run", {}).get("llm_enabled", True):
            summary = llm.complete(PATTERN_SUMMARY_PROMPT.format(cluster_dump=cluster_dump), model=ctx.config.get("model_fast", "openai/gpt-4o-mini"))
        ev = pd.DataFrame({"transaction_id": f["transaction_id"], "score": 0.0, "confidence": 0.5, "reasons": [["rule-synthesis"] for _ in range(len(f))]})
        return AgentResult(name=self.name, evidence=ev, diagnostics={"pattern_summary": summary})


class CaseManagerAgent(Agent):
    name = "CaseManagerAgent"

    def run(self, ctx: PipelineContext) -> AgentResult:
        merged = []
        for name, result in ctx.agent_outputs.items():
            if result.evidence.empty:
                continue
            d = result.evidence[["transaction_id", "score", "confidence", "reasons"]].copy()
            d = d.rename(columns={"score": f"score_{name}", "confidence": f"conf_{name}", "reasons": f"reasons_{name}"})
            merged.append(d)
        base = ctx.features["matrix"][["transaction_id"]].copy()
        for d in merged:
            base = base.merge(d, on="transaction_id", how="left")
        base = base.fillna(0)
        reason_cols = [c for c in base.columns if c.startswith("reasons_")]
        base["reasons"] = base[reason_cols].apply(lambda r: [x for x in r if x != 0], axis=1)
        score_cols = [c for c in base.columns if c.startswith("score_")]
        base["score"] = base[score_cols].mean(axis=1) if score_cols else 0.0
        base["confidence"] = 0.75
        return AgentResult(name=self.name, evidence=base[["transaction_id", "score", "confidence", "reasons"]])


class DecisionAgent(Agent):
    name = "DecisionAgent"

    def run(self, ctx: PipelineContext) -> AgentResult:
        cases = ctx.agent_outputs["CaseManagerAgent"].evidence.copy()
        score = cases["score"].clip(0, 1)
        cases["score"] = score
        return AgentResult(name=self.name, evidence=cases)
