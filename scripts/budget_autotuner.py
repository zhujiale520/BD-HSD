#!/usr/bin/env python3
"""Measured-cost budget calibration and separate held-out replay evaluation.

MVP: select only observed profiles, no prediction of unmeasured joint budgets.
Historical replay is exploratory, not a new prospective GPU experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from analyze_stage_cost import read_segments, summarize_policy


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid {name}")
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"invalid {name}")
    return value


def ids(values, name):
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be nonempty and unique")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in values):
        raise ValueError(f"invalid {name}")
    return set(values)


def select_profiles(profiles):
    eligible = []
    for row in profiles:
        if row["eligible"]:
            positive(row["cost_s"], "cost")
            positive(row["output_tokens"], "tokens")
            row = {**row, "seconds_per_output_token": row["cost_s"] / row["output_tokens"]}
            eligible.append(row)
    if not eligible:
        raise ValueError("no fully gated calibration candidate")
    key = lambda row: (row["seconds_per_output_token"], row["retrieval_k"], row["eagle_k"])
    shared = [row for row in eligible if row["retrieval_k"] == row["eagle_k"]]
    if not shared:
        raise ValueError("no fully gated shared candidate")
    return min(eligible, key=key), min(shared, key=key), sorted(eligible, key=key)


def calibrate(benchmark_path, trace_path, calibration_ids, holdout_ids):
    calibration = ids(calibration_ids, "calibration IDs")
    holdout = ids(holdout_ids, "holdout IDs")
    if calibration & holdout:
        raise ValueError("calibration/holdout overlap")
    benchmark = json.loads(benchmark_path.read_text())
    cohort = ids(benchmark["prompt_ids"], "benchmark IDs")
    if not calibration <= cohort:
        raise ValueError("calibration IDs not covered")
    if benchmark.get("debt_mode") not in (0, 1):
        raise ValueError("unsupported debt mode")
    segments = read_segments(trace_path)
    by_key = {}
    for segment in segments:
        key = (segment["policy"], segment["prompt_id"])
        if key in by_key:
            raise ValueError("duplicate measured segment")
        by_key[key] = segment
    budgets = {f"k{kr}_e{ke}": (kr, ke) for kr, ke in benchmark["budgets"]}
    if len(budgets) != len(benchmark["budgets"]):
        raise ValueError("duplicate budget")
    runs = {row["policy"]: row for row in benchmark["runs"]}
    if len(runs) != len(benchmark["runs"]) or set(runs) != set(budgets):
        raise ValueError("incomplete/duplicate calibration policies")
    if set(by_key) != {(policy, prompt) for policy in budgets for prompt in cohort}:
        raise ValueError("incomplete or unexpected trace matrix")
    profiles = []
    calibration_wall = 0.0
    for policy, (kr, ke) in sorted(budgets.items()):
        run = runs[policy]
        if (run["retrieval_k"], run["eagle_k"]) != (kr, ke) or not 1 <= ke <= kr:
            raise ValueError("invalid budget metadata")
        selected = [by_key[policy, prompt] for prompt in sorted(calibration)]
        summary = summarize_policy(selected, (kr, ke))
        wall = sum(positive(run["request_durations_s"][str(prompt)], "wall")
                   for prompt in calibration)
        tokens = sum(positive(run["output_lengths_by_prompt"][str(prompt)], "tokens")
                     for prompt in calibration)
        components = summary["exclusive_stage_total_s"]
        cost = sum(positive(value, "component", allow_zero=True) for value in components.values())
        positive(cost, "total component cost")
        calibration_wall += wall
        profiles.append({"policy": policy, "retrieval_k": kr, "eagle_k": ke,
                         "eligible": calibration <= set(run["equal_reference_prompt_ids"]),
                         "mismatch_ids": sorted(calibration - set(run["equal_reference_prompt_ids"])),
                         "cost_s": cost, "output_tokens": tokens,
                         "request_wall_s": wall, "components_s": components,
                         "model_round_count": summary["model_round_count"]})
    best, shared, ranked = select_profiles(profiles)
    return {"protocol": "measured-component-budget-calibration-v1",
            "status": "historical exploratory replay; not prospective validation",
            "selection_objective": "exclusive model-runner seconds / finalized output tokens",
            "limitations": ["Only observed budgets; no separable-cost prediction",
                            "Includes prefill; stage synchronization perturbs execution",
                            "Calibration wall excludes warmup, engine startup and offline analysis",
                            "Debt mode frozen; not a budget-only synchronization ablation",
                            "No SLO goodput objective"],
            "model": benchmark["model"], "draft_model": benchmark["draft_model"],
            "dataset": benchmark["dataset"], "max_tokens": benchmark["max_tokens"],
            "debt_mode": benchmark["debt_mode"], "engine_capacity_k": benchmark["engine_capacity_k"],
            "calibration_ids": sorted(calibration), "holdout_ids": sorted(holdout),
            "candidate_budgets": benchmark["budgets"],
            "selected_budget": [best["retrieval_k"], best["eagle_k"]],
            "selected_shared_budget": [shared["retrieval_k"], shared["eagle_k"]],
            "measured_calibration_wall_s": calibration_wall,
            "eligible_ranking": ranked, "profiles": profiles,
            "sources": [{"path": str(path), "sha256": digest(path)}
                        for path in (benchmark_path, trace_path)]}


def evaluate(manifest_path, validation_paths, validation_kind="historical-replay"):
    if validation_kind not in ("historical-replay", "fresh-gpu"):
        raise ValueError("invalid validation kind")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != "measured-component-budget-calibration-v1":
        raise ValueError("unsupported manifest")
    calibration = ids(manifest["calibration_ids"], "calibration IDs")
    holdout = ids(manifest["holdout_ids"], "holdout IDs")
    if calibration & holdout:
        raise ValueError("calibration/holdout overlap")
    selected = tuple(manifest["selected_budget"])
    shared = tuple(manifest["selected_shared_budget"])
    requested = {"autotuner": selected, "best_shared_calibration": shared, "fixed_k10_e3": (10, 3)}
    if not validation_paths or len(set(map(str, validation_paths))) != len(validation_paths):
        raise ValueError("missing/duplicate validation paths")
    rows = []
    source_hashes = set()
    for path in validation_paths:
        fingerprint = digest(path)
        if fingerprint in source_hashes:
            raise ValueError("duplicate validation content")
        source_hashes.add(fingerprint)
        data = json.loads(path.read_text())
        if ids(data["prompt_ids"], "validation IDs") != holdout:
            raise ValueError("validation cohort differs from frozen holdout")
        for key in ("model", "draft_model", "dataset", "max_tokens", "debt_mode", "engine_capacity_k"):
            if data.get(key) != manifest[key]:
                raise ValueError(f"validation protocol mismatch: {key}")
        runs = {(row["retrieval_k"], row["eagle_k"]): row for row in data["runs"]}
        if len(runs) != len(data["runs"]):
            raise ValueError("duplicate validation budget")
        if not set(requested.values()) <= set(runs):
            raise ValueError("frozen selected/fixed/shared budget not evaluated")
        metrics = {}
        for budget, run in runs.items():
            duration = sum(positive(run["request_durations_s"][str(prompt)], "wall") for prompt in holdout)
            count = sum(positive(run["output_lengths_by_prompt"][str(prompt)], "tokens") for prompt in holdout)
            metrics[budget] = {"budget": list(budget), "wall_s": duration,
                               "output_tokens": count, "tokens_per_second": count / duration,
                               "fully_gated": holdout <= set(run["equal_reference_prompt_ids"])}
        eligible = [row for row in metrics.values() if row["fully_gated"]]
        if not eligible:
            raise ValueError("no fully gated validation candidate")
        oracle = max(eligible, key=lambda row: row["tokens_per_second"])
        methods = {label: metrics[budget] for label, budget in requested.items()}
        selected_row = methods["autotuner"]
        rows.append({"source": {"path": str(path), "sha256": fingerprint}, "methods": methods,
                     "oracle_evaluated_subset": oracle,
                     "oracle_budget_count": len(eligible),
                     "strict_comparison_eligible": all(row["fully_gated"] for row in methods.values()),
                     "throughput_gap_to_subset_oracle_percent":
                     100 * (1 - selected_row["tokens_per_second"] / oracle["tokens_per_second"]),
                     "autotuner_vs_fixed_wall_reduction_percent":
                     100 * (1 - selected_row["wall_s"] / methods["fixed_k10_e3"]["wall_s"]),
                     "autotuner_vs_shared_wall_reduction_percent":
                     100 * (1 - selected_row["wall_s"] / methods["best_shared_calibration"]["wall_s"])})
    strict = [row for row in rows if row["strict_comparison_eligible"]]
    return {"protocol": "frozen-budget-heldout-replay-v1",
            "validation_kind": validation_kind,
            "manifest": {"path": str(manifest_path), "sha256": digest(manifest_path)},
            "limitations": ["Calibration uses existing development telemetry; not a preregistered new calibration study",
                            "Single validation run is descriptive; more runs required for confirmation",
                            "Oracle only over validation budgets actually evaluated, not full calibration grid",
                            "Descriptive medians; no confirmatory CI or test"],
            "run_count": len(rows), "fully_gated_run_count": len(strict), "runs": rows,
            "strict_median_shared_wall_reduction_percent": statistics.median(
                row["autotuner_vs_shared_wall_reduction_percent"] for row in strict) if strict else None,
            "strict_median_fixed_wall_reduction_percent": statistics.median(
                row["autotuner_vs_fixed_wall_reduction_percent"] for row in strict) if strict else None}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--benchmark", type=Path, required=True)
    calibration.add_argument("--trace", type=Path, required=True)
    calibration.add_argument("--calibration-ids", type=int, nargs="+", required=True)
    calibration.add_argument("--holdout-ids", type=int, nargs="+", required=True)
    validation = commands.add_parser("evaluate")
    validation.add_argument("--manifest", type=Path, required=True)
    validation.add_argument("--validation", type=Path, nargs="+", required=True)
    validation.add_argument("--validation-kind", choices=("historical-replay", "fresh-gpu"),
                            default="historical-replay")
    for command in (calibration, validation):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; refusing to overwrite frozen artifact")
    result = (calibrate(args.benchmark, args.trace, args.calibration_ids, args.holdout_ids)
              if args.command == "calibrate" else evaluate(args.manifest, args.validation,
                                                           args.validation_kind))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("selected_budget", "selected_shared_budget",
                       "run_count", "fully_gated_run_count", "strict_median_shared_wall_reduction_percent",
                       "strict_median_fixed_wall_reduction_percent") if key in result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
