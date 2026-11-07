#!/usr/bin/env python3
"""Aggregate marker-delimited BD-HSD GPU stage traces into cost components."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXCLUSIVE_KEYS = (
    "shape_or_graph_ns",
    "verify_ns",
    "rejection_ns",
    "route_ns",
    "eagle_first_pass_ns",
    "eagle_draft_decode_ns",
    "eagle_subbatch_compact_ns",
    "hybrid_merge_ns",
    "proposal_control_residual_ns",
    "other_model_runner_ns",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_policy(policy: str) -> tuple[int, int]:
    try:
        retrieval, eagle = policy.removeprefix("k").split("_e", maxsplit=1)
        return int(retrieval), int(eagle)
    except (AttributeError, ValueError) as error:
        raise ValueError(f"invalid policy name: {policy}") from error


def read_segments(path: Path) -> list[dict[str, Any]]:
    segments = []
    current: dict[str, Any] | None = None
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        record = json.loads(line)
        if record.get("record_type") == "budget_scan_marker":
            if record["event"] == "start":
                if current is not None:
                    raise ValueError(f"nested marker at line {line_number}")
                current = {**record, "records": []}
            elif record["event"] == "end":
                if current is None:
                    raise ValueError(f"unmatched end marker at line {line_number}")
                for key in ("phase", "policy", "prompt_id"):
                    if record[key] != current[key]:
                        raise ValueError(f"marker mismatch at line {line_number}: {key}")
                if current["phase"] == "measure":
                    segments.append(current)
                current = None
            else:
                raise ValueError(f"unknown marker event at line {line_number}")
            continue
        if current is not None:
            current["records"].append(record)
    if current is not None:
        raise ValueError("unterminated stage-trace segment")
    return segments


def summarize_policy(
    segments: list[dict[str, Any]], expected_budget: tuple[int, int]
) -> dict[str, Any]:
    stage_ns: Counter[str] = Counter()
    exclusive_ns: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    route_eagle_ns: Counter[str] = Counter()
    route_first_pass_ns: Counter[str] = Counter()
    route_first_pass_tokens: Counter[str] = Counter()
    debt_before_tokens: Counter[str] = Counter()
    num_rounds = 0
    num_scheduled_tokens = 0

    for segment in segments:
        for record in segment["records"]:
            if record.get("method") != "hybrid_ngram_eagle3":
                continue
            hybrid = record.get("hybrid_state")
            if not hybrid:
                continue
            observed_budget = (hybrid.get("retrieval_k"), hybrid.get("eagle_k"))
            if observed_budget != expected_budget:
                raise ValueError(
                    f"trace budget {observed_budget} != marker {expected_budget}"
                )
            timing = record["timing_ns"]
            stage_ns.update({key: int(value) for key, value in timing.items()})
            eagle_ns = int(timing.get("eagle_sync_ns", 0))
            first_pass_ns = int(timing.get("eagle_first_pass_ns", 0))
            draft_decode_ns = int(timing.get("eagle_draft_decode_ns", 0))
            residual_ns = max(
                0,
                int(timing.get("propose_ns", 0))
                - int(timing.get("route_ns", 0))
                - eagle_ns
                - int(timing.get("eagle_subbatch_compact_ns", 0))
                - int(timing.get("hybrid_merge_ns", 0)),
            )
            exclusive = {
                "shape_or_graph_ns": int(timing.get("shape_or_graph_ns", 0)),
                "verify_ns": int(timing.get("verify_ns", 0)),
                "rejection_ns": int(timing.get("rejection_ns", 0)),
                "route_ns": int(timing.get("route_ns", 0)),
                "eagle_first_pass_ns": first_pass_ns,
                "eagle_draft_decode_ns": draft_decode_ns,
                "eagle_subbatch_compact_ns": int(
                    timing.get("eagle_subbatch_compact_ns", 0)
                ),
                "hybrid_merge_ns": int(timing.get("hybrid_merge_ns", 0)),
                "proposal_control_residual_ns": residual_ns,
                "other_model_runner_ns": int(
                    timing.get("other_model_runner_ns", 0)
                ),
            }
            exclusive_ns.update(exclusive)
            route = str(hybrid.get("route", "unknown"))
            route_counts[route] += 1
            route_eagle_ns[route] += eagle_ns
            route_first_pass_ns[route] += first_pass_ns
            route_first_pass_tokens[route] += int(
                record.get("eagle_first_pass_input_tokens", 0)
            )
            debt_before_tokens[route] += sum(
                int(value)
                for value in hybrid.get("debt_tokens_before_by_req", {}).values()
            )
            num_rounds += 1
            num_scheduled_tokens += int(record.get("num_scheduled_tokens", 0))

    if num_rounds == 0:
        raise ValueError("policy has no measured hybrid stage records")
    exclusive_total = sum(exclusive_ns.values())
    model_runner_total = stage_ns["model_runner_total_ns"]
    return {
        "request_count": len(segments),
        "model_round_count": num_rounds,
        "num_scheduled_tokens": num_scheduled_tokens,
        "stage_total_s": {
            key.removesuffix("_ns"): value / 1e9
            for key, value in sorted(stage_ns.items())
        },
        "exclusive_stage_total_s": {
            key.removesuffix("_ns"): exclusive_ns[key] / 1e9
            for key in EXCLUSIVE_KEYS
        },
        "exclusive_stage_share_percent": {
            key.removesuffix("_ns"): 100 * exclusive_ns[key] / exclusive_total
            for key in EXCLUSIVE_KEYS
        },
        "exclusive_sum_vs_model_runner_percent": (
            100 * exclusive_total / model_runner_total
        ),
        "mean_model_runner_ms_per_round": model_runner_total / num_rounds / 1e6,
        "route_counts": dict(sorted(route_counts.items())),
        "route_eagle_total_s": {
            key: value / 1e9 for key, value in sorted(route_eagle_ns.items())
        },
        "route_eagle_first_pass_total_s": {
            key: value / 1e9
            for key, value in sorted(route_first_pass_ns.items())
        },
        "route_eagle_first_pass_input_tokens": dict(
            sorted(route_first_pass_tokens.items())
        ),
        "route_debt_before_tokens": dict(sorted(debt_before_tokens.items())),
    }


def analyze(trace_path: Path, benchmark_path: Path) -> dict[str, Any]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    segments = read_segments(trace_path)
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        by_policy[segment["policy"]].append(segment)
    if "policy_budgets" in benchmark:
        budget_by_policy = {
            policy: tuple(budget)
            for policy, budget in benchmark["policy_budgets"].items()
        }
    else:
        budget_by_policy = {
            f"k{retrieval}_e{eagle}": (retrieval, eagle)
            for retrieval, eagle in benchmark["budgets"]
        }
    expected_policies = set(budget_by_policy)
    if set(by_policy) != expected_policies:
        raise ValueError(
            f"trace policies {set(by_policy)} != benchmark policies {expected_policies}"
        )
    benchmark_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in benchmark["runs"]:
        benchmark_runs[run["policy"]].append(run)
    policies = {}
    for policy in sorted(by_policy):
        summary = summarize_policy(by_policy[policy], budget_by_policy[policy])
        runs = benchmark_runs[policy]
        summary["request_wall_total_s"] = sum(
            run["request_duration_total_s"] for run in runs
        )
        summary["equal_reference_count"] = sum(
            len(run["equal_reference_prompt_ids"]) for run in runs
        )
        summary["mismatch_reference_prompt_ids"] = sorted(
            {
                prompt_id
                for run in runs
                for prompt_id in run["mismatch_reference_prompt_ids"]
            }
        )
        policies[policy] = summary
    return {
        "analysis": "Marker-delimited BD-HSD component cost attribution",
        "measurement_note": (
            "CUDA synchronization is enabled at stage boundaries. Exclusive stages "
            "sum to the traced model-runner time, but proposal_control_residual is "
            "proposal preparation plus control/state bookkeeping, not pure state cost."
        ),
        "trace_source": {"path": str(trace_path), "sha256": sha256(trace_path)},
        "benchmark_source": {
            "path": str(benchmark_path),
            "sha256": sha256(benchmark_path),
        },
        "debt_mode": benchmark.get("debt_mode"),
        "prompt_ids": benchmark["prompt_ids"],
        "policies": policies,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--benchmark-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace, args.benchmark_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    compact = {
        policy: {
            "wall_s": data["request_wall_total_s"],
            "model_runner_s": data["stage_total_s"]["model_runner_total"],
            "verify_s": data["stage_total_s"]["verify"],
            "route_s": data["stage_total_s"]["route"],
            "eagle_s": data["stage_total_s"]["eagle_sync"],
            "routes": data["route_counts"],
        }
        for policy, data in result["policies"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
