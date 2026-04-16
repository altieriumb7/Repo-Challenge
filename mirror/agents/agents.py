from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

from mirror.agents.base import Agent
from mirror.llm.prompts import ARBITRATION_PROMPT, COMMUNICATION_ANALYSIS_PROMPT, PATTERN_SUMMARY_PROMPT
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
        tx = ctx.data["linked_transactions"].sort_values("event_time").copy()
        tx["payment_method"] = tx["payment_method"].astype(str) if "payment_method" in tx.columns else "unknown"
        tx["transaction_type"] = tx["transaction_type"].astype(str) if "transaction_type" in tx.columns else "unknown"
        tx["hour"] = tx["event_time"].dt.hour.fillna(0).astype(int)
        tx["dow"] = tx["event_time"].dt.dayofweek.fillna(0).astype(int)
        tx["log_amount"] = np.log1p(tx["amount"].clip(lower=0))
        tx["global_time_idx"] = np.arange(len(tx))

        user_mean = tx.groupby("sender_id")["log_amount"].transform("mean").replace(0, 1e-6)
        user_std = tx.groupby("sender_id")["log_amount"].transform("std").fillna(0.2).replace(0, 0.2)
        rolling_mean = tx.groupby("sender_id")["log_amount"].transform(lambda s: s.rolling(8, min_periods=2).mean()).fillna(user_mean)
        decay_mean = tx.groupby("sender_id")["log_amount"].transform(lambda s: s.ewm(alpha=0.35, adjust=False).mean())
        amount_novelty = ((tx["log_amount"] - rolling_mean).abs() / (user_std + 1e-6)).clip(0, 6) / 6

        tx["recipient_novelty"] = (tx.groupby(["sender_id", "recipient_id"]).cumcount() == 0).astype(float)
        tx["tx_type_novelty"] = (tx.groupby(["sender_id", "transaction_type"]).cumcount() == 0).astype(float)
        tx["payment_method_novelty"] = (tx.groupby(["sender_id", "payment_method"]).cumcount() == 0).astype(float)
        hour_prob = tx.groupby(["sender_id", "hour"]).cumcount() / tx.groupby("sender_id").cumcount().clip(lower=1)
        dow_prob = tx.groupby(["sender_id", "dow"]).cumcount() / tx.groupby("sender_id").cumcount().clip(lower=1)
        hour_day_novelty = (1 - 0.5 * (hour_prob.fillna(0) + dow_prob.fillna(0))).clip(0, 1)

        drift_user = (tx["log_amount"] - decay_mean).abs().clip(0, 5) / 5
        global_roll = tx["log_amount"].rolling(50, min_periods=8).mean().fillna(tx["log_amount"].expanding().mean())
        drift_global = (tx["log_amount"] - global_roll).abs().clip(0, 6) / 6
        last_time = tx.groupby("sender_id")["event_time"].shift(1)
        mins = ((tx["event_time"] - last_time).dt.total_seconds() / 60).fillna(1e6)
        burst = (mins <= 10).astype(float)
        sequence_count = tx.groupby(["sender_id", "recipient_id"]).cumcount()
        sequence_anomaly = ((sequence_count == 0) & (tx.groupby("sender_id").cumcount() > 5)).astype(float)

        score = (
            amount_novelty * 0.2
            + tx["recipient_novelty"] * 0.14
            + tx["tx_type_novelty"] * 0.08
            + tx["payment_method_novelty"] * 0.08
            + hour_day_novelty * 0.12
            + drift_user * 0.16
            + drift_global * 0.08
            + burst * 0.1
            + sequence_anomaly * 0.04
        ).clip(0, 1)
        return _evidence_from_score(tx["transaction_id"], score, self.name, "temporal-drift-novelty", 0.81)


class NetworkRiskAgent(Agent):
    name = "NetworkRiskAgent"

    def run(self, ctx: PipelineContext) -> AgentResult:
        tx = ctx.data["linked_transactions"].sort_values("event_time").copy()
        sender_deg = tx.groupby("sender_id")["recipient_id"].nunique()
        recipient_deg = tx.groupby("recipient_id")["sender_id"].nunique()
        edge_first = (tx.groupby(["sender_id", "recipient_id"]).cumcount() == 0).astype(float)
        comp_size = tx.groupby("recipient_id")["sender_id"].transform("nunique")
        rare_component = (1 / comp_size.clip(lower=1)).clip(0, 1)
        fan_out = tx["sender_id"].map(sender_deg).fillna(0)
        fan_in = tx["recipient_id"].map(recipient_deg).fillna(0)
        unusualness = ((edge_first * np.log1p(fan_out + fan_in)) / np.log1p(fan_out.mean() + fan_in.mean() + 1)).clip(0, 1)
        concentration = tx.groupby("sender_id")["recipient_id"].transform(lambda s: s.value_counts(normalize=True).iloc[0] if len(s) else 0)
        concentration_spike = (concentration > 0.8).astype(float)
        shared_recipient = (fan_in > 5).astype(float) * (fan_out <= 2).astype(float)
        local_cluster_anomaly = ((fan_in > fan_in.quantile(0.95)) | (fan_out > fan_out.quantile(0.95))).astype(float)
        score = (
            edge_first * 0.2
            + rare_component * 0.12
            + unusualness * 0.18
            + concentration_spike * 0.16
            + shared_recipient * 0.16
            + local_cluster_anomaly * 0.18
        ).clip(0, 1)
        return _evidence_from_score(tx["transaction_id"], score, self.name, "network-anomaly-graph", 0.76)


class GeoRiskAgent(Agent):
    name = "GeoRiskAgent"

    def run(self, ctx: PipelineContext) -> AgentResult:
        tx = ctx.data["linked_transactions"].sort_values("event_time").copy()
        loc = ctx.data.get("locations", pd.DataFrame()).copy()
        tx["geo_score"] = 0.0
        tx["geo_conf"] = 0.45
        if loc.empty or "user_id" not in loc.columns:
            return _evidence_from_score(tx["transaction_id"], tx["geo_score"] + 0.2, self.name, "geo-weak-evidence", 0.4)
        loc["event_time"] = pd.to_datetime(loc.get("event_time"), utc=True, errors="coerce")
        for c in ["lat", "lon"]:
            if c not in loc.columns:
                loc[c] = np.nan
        rows = []
        for sender, grp in tx.groupby("sender_id"):
            sender_loc = loc[loc["user_id"].astype(str) == str(sender)].sort_values("event_time")
            if sender_loc.empty:
                rows.extend([{"transaction_id": tid, "score": 0.3, "confidence": 0.35} for tid in grp["transaction_id"]])
                continue
            base_lat = sender_loc["lat"].median()
            base_lon = sender_loc["lon"].median()
            prev_t, prev_lat, prev_lon = None, None, None
            for row in grp.itertuples(index=False):
                if pd.isna(base_lat) or pd.isna(base_lon):
                    rows.append({"transaction_id": row.transaction_id, "score": 0.25, "confidence": 0.35})
                    continue
                jump = abs(float(getattr(row, "lat", base_lat) if hasattr(row, "lat") else base_lat) - base_lat) + abs(float(getattr(row, "lon", base_lon) if hasattr(row, "lon") else base_lon) - base_lon)
                geo_shift = min(1.0, jump / 8.0)
                impossible = 0.0
                if prev_t is not None and row.event_time is not pd.NaT:
                    dt_h = max((row.event_time - prev_t).total_seconds() / 3600, 1e-3)
                    speed_like = (abs(base_lat - prev_lat) + abs(base_lon - prev_lon)) / dt_h if prev_lat is not None else 0
                    impossible = float(speed_like > 15)
                in_person = str(getattr(row, "payment_method", "unknown")).lower() in {"cash", "card_present", "pos", "in_person"}
                in_person_consistency = 0.0 if in_person and geo_shift < 0.2 else (0.2 if in_person else 0.0)
                rows.append(
                    {"transaction_id": row.transaction_id, "score": min(1.0, geo_shift * 0.5 + impossible * 0.35 + in_person_consistency), "confidence": 0.7 if not pd.isna(base_lat) else 0.35}
                )
                prev_t, prev_lat, prev_lon = row.event_time, base_lat, base_lon
        geo = pd.DataFrame(rows)
        return AgentResult(
            name=self.name,
            evidence=pd.DataFrame(
                {
                    "transaction_id": geo["transaction_id"].astype(str),
                    "score": geo["score"].astype(float),
                    "confidence": geo["confidence"].astype(float),
                    "reasons": [["geo-mobility-drift"] for _ in range(len(geo))],
                }
            ),
        )


class CommsRiskAgent(Agent):
    name = "CommsRiskAgent"

    def run(self, ctx: PipelineContext) -> AgentResult:
        tx = ctx.features["matrix"][["transaction_id", "sender_id", "burst_30m"]].copy()
        sms = ctx.data.get("sms", pd.DataFrame())
        mails = ctx.data.get("mails", pd.DataFrame())
        audio = ctx.data.get("audio", pd.DataFrame())
        text_blobs: dict[str, list[str]] = defaultdict(list)
        comms_records_considered = 0
        for df in [sms, mails, audio]:
            if df.empty:
                continue
            text_col = "text" if "text" in df.columns else None
            user_col = "user_id" if "user_id" in df.columns else ("sender_id" if "sender_id" in df.columns else None)
            if not text_col or not user_col:
                continue
            for uid, grp in df.groupby(user_col):
                messages = grp[text_col].fillna("").astype(str).tolist()
                comms_records_considered += len(messages)
                text_blobs[str(uid)].append(" ".join(messages).lower())
        patterns = {
            "urgency": r"\b(urgent|immediately|asap|act now)\b",
            "credential": r"\b(password|otp|verify account|pin)\b",
            "payment_pressure": r"\b(wire|transfer|gift card|crypto|payment due)\b",
            "short_link": r"(bit\.ly|tinyurl|t\.co|goo\.gl)",
            "invoice_scare": r"\b(invoice|tax|refund|bill|arrears|penalty)\b",
            "marketplace_delivery": r"\b(marketplace|delivery|parcel|shipment)\b",
        }
        cheap_scores = {}
        hit_counts: dict[str, int] = {}
        for uid, blobs in text_blobs.items():
            txt = " ".join(blobs)
            hits = sum(1 for pat in patterns.values() if re.search(pat, txt))
            hit_counts[uid] = hits
            cheap_scores[uid] = min(1.0, hits / 5)

        llm_cfg = ctx.config.get("llm", {})
        run_cfg = ctx.config.get("run", {})
        llm = ctx.data.get("llm_client")
        llm_budget = int(run_cfg.get("max_messages_for_llm_review", llm_cfg.get("max_messages_for_llm_review", 25)))
        llm_workers = max(1, int(run_cfg.get("max_llm_workers", 3)))
        llm_skip_reason = ""
        should_use_llm = bool(llm and run_cfg.get("llm_enabled", True) and not run_cfg.get("disable_llm", False))
        if run_cfg.get("disable_llm", False) or not run_cfg.get("llm_enabled", True):
            llm_skip_reason = "llm disabled by config"
        elif not llm:
            llm_skip_reason = "llm client unavailable"
        elif not getattr(llm, "api_key", ""):
            llm_skip_reason = "no API key"
            should_use_llm = False
        risky_users = tx.loc[tx["burst_30m"] > 0, "sender_id"].astype(str).unique().tolist()
        ranked = sorted(cheap_scores.items(), key=lambda x: x[1], reverse=True)[:llm_budget]
        selected_users = sorted(uid for uid, user_score in ranked if uid in risky_users or user_score >= 0.35 or hit_counts.get(uid, 0) >= 2)
        llm_budget_obj = ctx.data.get("llm_budget")
        if llm_budget_obj and llm_budget_obj.max_calls <= 0:
            selected_users = []
            llm_skip_reason = "budget exhausted"
        if not selected_users and not llm_skip_reason:
            llm_skip_reason = "no suspicious clusters selected"
        llm_map: dict[str, float] = {}
        llm_diag: dict[str, dict] = {}
        if should_use_llm:
            model = ctx.config.get("model_fast", "openai/gpt-4o-mini")
            prompt_cache: dict[str, dict] = {}
            prompt_lock = Lock()

            def _review(uid: str) -> tuple[str, float, dict]:
                payload = {"messages": text_blobs.get(uid, [])[:6]}
                prompt = COMMUNICATION_ANALYSIS_PROMPT.format(summary=json.dumps(payload, ensure_ascii=True))
                with prompt_lock:
                    cached = prompt_cache.get(prompt)
                if cached is not None:
                    return uid, float(cached["score"]), dict(cached["parsed"])
                wait_start = time.perf_counter()
                if llm_budget_obj and not llm_budget_obj.try_acquire(model=model, wait_time_seconds=time.perf_counter() - wait_start):
                    return uid, 0.0, {"skipped": "llm_budget_exhausted"}
                raw = llm.complete(prompt, model=model)
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = {"scam_probability": 0.55, "urgency_score": 0.5, "reasoning_summary": str(raw)[:220]}
                llm_score = float(parsed.get("scam_probability", 0.5)) * 0.7 + float(parsed.get("urgency_score", 0.5)) * 0.3
                bounded = min(1.0, max(0.0, llm_score))
                with prompt_lock:
                    prompt_cache[prompt] = {"score": bounded, "parsed": parsed}
                return uid, bounded, parsed

            with ThreadPoolExecutor(max_workers=min(llm_workers, len(selected_users) or 1)) as pool:
                futures = [pool.submit(_review, uid) for uid in selected_users]
                for fut in as_completed(futures):
                    uid, score_val, parsed = fut.result()
                    llm_map[uid] = score_val
                    llm_diag[uid] = parsed
        successful_reviews = len([v for v in llm_diag.values() if not v.get("skipped")])
        if successful_reviews == 0 and any(v.get("skipped") == "llm_budget_exhausted" for v in llm_diag.values()) and not llm_skip_reason:
            llm_skip_reason = "budget exhausted"
        score = tx["sender_id"].astype(str).map(lambda u: cheap_scores.get(u, 0.0) * 0.7 + llm_map.get(u, 0.0) * 0.3).fillna(0.0)
        return AgentResult(
            name=self.name,
            evidence=pd.DataFrame(
                {
                    "transaction_id": tx["transaction_id"].astype(str),
                    "score": score.astype(float).clip(0, 1),
                    "confidence": np.where(score > 0, 0.72, 0.55),
                    "reasons": [["comms-social-engineering"] for _ in range(len(tx))],
                }
            ),
            diagnostics={
                "llm_reviews": successful_reviews,
                "llm_selected_users": len(selected_users),
                "llm_escalated_users": len(selected_users),
                "llm_workers": llm_workers,
                "comms_records_considered": comms_records_considered,
                "comms_users_with_text": len(text_blobs),
                "llm_skip_reason": llm_skip_reason if successful_reviews == 0 else "",
                "llm_structured": {k: llm_diag[k] for k in sorted(llm_diag)},
            },
        )


class RuleSynthesisAgent(Agent):
    name = "RuleSynthesisAgent"

    def run(self, ctx: PipelineContext) -> AgentResult:
        f = ctx.features["matrix"].copy()
        suspicious = f.nlargest(40, "sender_amount_zproxy")
        cluster_dump = suspicious[["transaction_id", "sender_id", "recipient_id", "amount", "off_hours", "edge_novelty"]].to_dict("records")
        llm = ctx.data.get("llm_client")
        llm_budget_obj = ctx.data.get("llm_budget")
        summary = "LLM disabled"
        model = ctx.config.get("model_fast", "openai/gpt-4o-mini")
        if llm and ctx.config.get("run", {}).get("llm_enabled", True) and not ctx.config.get("run", {}).get("disable_llm", False):
            if llm_budget_obj and not llm_budget_obj.try_acquire(model=model):
                summary = "LLM budget exhausted"
            else:
                summary = llm.complete(PATTERN_SUMMARY_PROMPT.format(cluster_dump=cluster_dump), model=model)
        ev = pd.DataFrame({"transaction_id": f["transaction_id"], "score": 0.0, "confidence": 0.5, "reasons": [["rule-synthesis"] for _ in range(len(f))]})
        return AgentResult(name=self.name, evidence=ev, diagnostics={"pattern_summary": summary, "clusters": cluster_dump[:10]})


class PatternMemoryAgent(Agent):
    name = "PatternMemoryAgent"

    def run(self, ctx: PipelineContext) -> AgentResult:
        f = ctx.features["matrix"].copy()
        top = f.sort_values(["sender_amount_zproxy", "off_hours", "edge_novelty"], ascending=False).head(30)
        motifs = []
        for row in top.itertuples(index=False):
            motif = f"{'late-night' if row.off_hours else 'daytime'}-{'new-recipient' if row.edge_novelty else 'known-recipient'}-amt-{int(min(max(row.amount, 0), 99999))}"
            motifs.append({"transaction_id": str(row.transaction_id), "motif": motif, "score_hint": float(min(1.0, max(0.0, row.sender_amount_zproxy / 5)))})
        motif_scores = {m["transaction_id"]: m["score_hint"] for m in motifs}
        out_dir = Path(ctx.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "patterns.json").write_text(json.dumps({"motifs": motifs[:200]}, indent=2, ensure_ascii=True), encoding="utf-8")
        evidence = pd.DataFrame(
            {
                "transaction_id": f["transaction_id"].astype(str),
                "score": f["transaction_id"].astype(str).map(motif_scores).fillna(0.0),
                "confidence": 0.67,
                "reasons": [["pattern-memory-match"] for _ in range(len(f))],
            }
        )
        return AgentResult(name=self.name, evidence=evidence, diagnostics={"pattern_cards": motifs[:10]})


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
        conf_cols = [c for c in base.columns if c.startswith("conf_")]
        base["confidence"] = base[conf_cols].mean(axis=1) if conf_cols else 0.7
        return AgentResult(name=self.name, evidence=base[["transaction_id", "score", "confidence", "reasons"]])


class DecisionAgent(Agent):
    name = "DecisionAgent"

    def run(self, ctx: PipelineContext) -> AgentResult:
        cases = ctx.agent_outputs["CaseManagerAgent"].evidence.copy()
        mode = ctx.config.get("run", {}).get("decision_mode", "conservative")
        weight_map = ctx.config.get("ensemble", {}).get(
            mode,
            {
                "TemporalBehaviorAgent": 0.24,
                "NetworkRiskAgent": 0.2,
                "GeoRiskAgent": 0.16,
                "CommsRiskAgent": 0.18,
                "PatternMemoryAgent": 0.14,
                "RuleSynthesisAgent": 0.08,
            },
        )
        weighted = []
        reasons = []
        for name, w in weight_map.items():
            ev = ctx.agent_outputs.get(name)
            if ev is None or ev.evidence.empty:
                continue
            part = ev.evidence[["transaction_id", "score"]].rename(columns={"score": f"s_{name}"})
            weighted.append((w, part))
            reasons.append(name)
        for _, part in weighted:
            cases = cases.merge(part, on="transaction_id", how="left")
        score_cols = [c for c in cases.columns if c.startswith("s_")]
        denom = sum(w for w, _ in weighted) or 1.0
        agg = sum(cases[c].fillna(0) * w for w, c in [(w, f"s_{n}") for n, w in weight_map.items() if f"s_{n}" in cases.columns]) / denom
        cases["score"] = agg.clip(0, 1)
        sparse = (cases[[c for c in cases.columns if c.startswith("s_")]].fillna(0).sum(axis=1) <= 0.05)
        cases.loc[sparse, "score"] = cases.loc[sparse, "score"] * 0.65
        cases["fraud_score"] = cases["score"]
        cases["evidence_confidence"] = cases["confidence"].clip(0, 1)
        cases["decision"] = np.where(cases["score"] >= ctx.config.get("thresholding", {}).get("prelim_decision_threshold", 0.72), "flag", "review")
        cases["top_reasons"] = [reasons[:3] for _ in range(len(cases))]
        return AgentResult(name=self.name, evidence=cases)
