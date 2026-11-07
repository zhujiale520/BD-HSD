#!/usr/bin/env python3
"""Analyze the frozen four-way BD-HSD budget ablation JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from budget_analysis import (
    POLICIES,
    summarize,
)


def aggregate_policy_telemetry(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for policy in POLICIES:
        selected = [run for run in runs if run["policy"] == policy]
        metrics = [
            metric
            for run in selected
            for metric in run["spec_metrics_by_prompt"].values()
        ]
        num_drafts = sum(int(metric["num_drafts"]) for metric in metrics)
        drafted = sum(int(metric["num_draft_tokens"]) for metric in metrics)
        accepted = sum(int(metric["num_accepted_tokens"]) for metric in metrics)
        routing_keys = {
            key for run in selected for key in run["hybrid_routing"]
        }
        routing = {
            key: sum(int(run["hybrid_routing"].get(key, 0)) for run in selected)
            for key in sorted(routing_keys)
        }
        result[policy] = {
            "num_drafts": num_drafts,
            "num_draft_tokens": drafted,
            "num_accepted_tokens": accepted,
            "mean_acceptance_length": 1 + accepted / num_drafts,
            "acceptance_rate": accepted / drafted,
            "routing": routing,
            "peak_allocated_bytes": max(
                int(run["peak_memory"]["max_allocated_bytes"])
                for run in selected
            ),
            "peak_reserved_bytes": max(
                int(run["peak_memory"]["max_reserved_bytes"])
                for run in selected
            ),
        }
    return result


def analyze(input_path: Path) -> dict[str, Any]:
    raw_bytes = input_path.read_bytes()
    raw = json.loads(raw_bytes)
    return {
        "analysis": "BD-HSD four-way same-engine budget-factor ablation",
        "source_path": str(input_path),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "protocol": raw["protocol"],
        "model": raw["model"],
        "draft_model": raw["draft_model"],
        "dataset": raw["dataset"],
        "prompt_ids": raw["prompt_ids"],
        "max_tokens": raw["max_tokens"],
        "policy_config": raw["policy_config"],
        "summary": summarize(raw["runs"]),
        "telemetry_by_policy": aggregate_policy_telemetry(raw["runs"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
