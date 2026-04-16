#!/usr/bin/env python3
from __future__ import annotations

import copy
import csv
from pathlib import Path

from mirror.pipeline import run_pipeline
from mirror.utils.config import load_config

ABLATIONS = {
    "full_system": {},
    "no_llm": {"run": {"disable_llm": True, "llm_enabled": False}},
    "no_comms": {"agents": {"disable": ["CommsRiskAgent"]}},
    "no_geo": {"agents": {"disable": ["GeoRiskAgent"]}},
    "no_graph": {"agents": {"disable": ["NetworkRiskAgent"]}},
    "no_audio": {"run": {"disable_audio": True, "audio_enabled": False}},
}


def merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", required=True)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output-root", default="outputs/ablations")
    p.add_argument("--scenario", default="scenario")
    args = p.parse_args()

    base = load_config(args.config, scenario_name=args.scenario)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, patch in ABLATIONS.items():
        cfg = merge(base, patch)
        cfg.setdefault("run", {})["scenario_name"] = args.scenario
        result = run_pipeline(args.train_dir, args.eval_dir, str(out_root / name), cfg)
        rows.append(
            {
                "ablation": name,
                "run_id": result["run_id"],
                "threshold": result["diagnostics"].get("threshold"),
                "suspect_rate": result["diagnostics"].get("suspect_rate"),
                "llm_calls": result["diagnostics"].get("llm_usage", {}).get("calls", 0),
            }
        )

    table = out_root / "ablation_results.csv"
    with table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
