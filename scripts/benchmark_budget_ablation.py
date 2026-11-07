#!/usr/bin/env python3
"""Same-engine request-interleaved BD-HSD budget-factor ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_V1", "1")

import torch
from vllm import LLM, SamplingParams

from benchmark_spec_decode import (
    counter_dict_delta,
    hybrid_routing_snapshot,
    load_prompts,
    metric_delta,
    metric_snapshot,
)
from benchmark_helpers import (
    add_counter_delta,
    gpu_snapshot,
    matches_reference,
    token_hash,
)


POLICIES = (
    "shared_k3_e3_shadow0",
    "shared_k10_e10_shadow0",
    "decoupled_k10_e3_shadow0",
    "bsdh_k10_e3_debt1",
)
POLICY_CONFIG = {
    "shared_k3_e3_shadow0": {"retrieval_k": 3, "eagle_k": 3, "debt": 0},
    "shared_k10_e10_shadow0": {"retrieval_k": 10, "eagle_k": 10, "debt": 0},
    "decoupled_k10_e3_shadow0": {"retrieval_k": 10, "eagle_k": 3, "debt": 0},
    "bsdh_k10_e3_debt1": {"retrieval_k": 10, "eagle_k": 3, "debt": 1},
}
COMPARISONS = {
    "shared_scale_k10e10_vs_k3e3": (
        "shared_k3_e3_shadow0",
        "shared_k10_e10_shadow0",
    ),
    "budget_decoupling_k10e3_vs_k10e10": (
        "shared_k10_e10_shadow0",
        "decoupled_k10_e3_shadow0",
    ),
    "bounded_debt_vs_shadow_k10e3": (
        "decoupled_k10_e3_shadow0",
        "bsdh_k10_e3_debt1",
    ),
    "full_bsdh_vs_k3e3": (
        "shared_k3_e3_shadow0",
        "bsdh_k10_e3_debt1",
    ),
}


def read_reference(path: Path) -> dict[int, list[int]]:
    reference = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if int(row["batch_size"]) != 1:
            continue
        for prompt_id, token_ids in zip(
            row["prompt_ids"], row["output_token_ids"], strict=True
        ):
            reference[int(prompt_id)] = token_ids
    return reference


def balanced_orders(num_prompts: int, seed: int) -> list[list[str]]:
    """Return randomized orders balanced across ordinal positions."""
    rng = random.Random(seed)
    base = list(POLICIES)
    rng.shuffle(base)
    rotations = [base[offset:] + base[:offset] for offset in range(len(base))]
    orders = [rotations[index % len(rotations)] for index in range(num_prompts)]
    rng.shuffle(orders)
    return orders


def _sample_summary(samples: list[float], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    bootstrap_medians = sorted(
        statistics.median(rng.choices(samples, k=len(samples)))
        for _ in range(100_000)
    )
    result = {
        "samples": samples,
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "positive_pairs": sum(value > 0 for value in samples),
        "bootstrap_median_95_ci": [
            bootstrap_medians[round(0.025 * (len(bootstrap_medians) - 1))],
            bootstrap_medians[round(0.975 * (len(bootstrap_medians) - 1))],
        ],
        "bootstrap_seed": seed,
        "bootstrap_resamples": 100_000,
    }
    try:
        from scipy.stats import wilcoxon

        result["wilcoxon_two_sided_p"] = float(
            wilcoxon(samples, alternative="two-sided", method="exact").pvalue
        )
        result["wilcoxon_greater_p"] = float(
            wilcoxon(samples, alternative="greater", method="exact").pvalue
        )
    except ImportError:
        result["wilcoxon_two_sided_p"] = None
        result["wilcoxon_greater_p"] = None
    return result


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    details = []
    for pair in sorted({int(run["pair"]) for run in runs}):
        pair_runs = {run["policy"]: run for run in runs if run["pair"] == pair}
        strict_ids = sorted(
            set.intersection(
                *(set(pair_runs[policy]["equal_reference_prompt_ids"])
                  for policy in POLICIES)
            )
        )
        totals = {
            policy: sum(
                pair_runs[policy]["request_durations_s"][str(prompt_id)]
                for prompt_id in strict_ids
            )
            for policy in POLICIES
        }
        reductions = {
            name: 100 * (1 - totals[new] / totals[base])
            for name, (base, new) in COMPARISONS.items()
        }
        details.append(
            {
                "pair": pair,
                "strict_prompt_count": len(strict_ids),
                "strict_prompt_ids": strict_ids,
                "strict_total_s": totals,
                "latency_reduction_percent": reductions,
            }
        )
    result: dict[str, Any] = {
        "num_pairs": len(details),
        "pair_details": details,
        "comparisons": {},
    }
    for index, name in enumerate(COMPARISONS):
        samples = [item["latency_reduction_percent"][name] for item in details]
        result["comparisons"][name] = _sample_summary(samples, 20260913 + index)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--canitedit-path", type=Path, required=True)
    parser.add_argument("--reference-eagle", type=Path, required=True)
    parser.add_argument("--prompt-indices", type=int, nargs="+", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--randomization-seed", type=int, default=20260913)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt_args = SimpleNamespace(
        dataset="canitedit",
        model=args.model,
        prompt_indices=args.prompt_indices,
        num_prompts=len(args.prompt_indices),
        instructcoder_path=None,
        prompt_source_results=None,
        canitedit_path=args.canitedit_path,
        canitedit_instruction="descriptive",
    )
    prompts = load_prompts(prompt_args)
    reference = read_reference(args.reference_eagle)
    prompt_ids = [prompt_id for prompt_id, _, _ in prompts]
    if not set(prompt_ids).issubset(reference):
        raise ValueError("reference does not cover the benchmark cohort")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        speculative_config={
            "method": "hybrid_ngram_eagle3",
            "model": args.draft_model,
            "num_speculative_tokens": 10,
            "hybrid_eagle_num_speculative_tokens": 3,
            "prompt_lookup_min": 3,
            "prompt_lookup_max": 7,
            "disable_padded_drafter_batch": True,
            "hybrid_max_draft_skip_steps": 1,
        },
        disable_log_stats=False,
        enable_prefix_caching=False,
        seed=42,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens, ignore_eos=False
    )

    def configure(policy: str) -> None:
        config = POLICY_CONFIG[policy]
        llm.collective_rpc("set_hybrid_retrieval_only", args=(False,))
        llm.collective_rpc("set_hybrid_ngram_window", args=(3, 7))
        llm.collective_rpc(
            "set_hybrid_runtime_budgets",
            args=(config["retrieval_k"], config["eagle_k"]),
        )
        llm.collective_rpc(
            "set_hybrid_max_draft_skip_steps",
            args=(config["debt"],),
        )

    for policy in POLICIES:
        configure(policy)
        llm.generate([prompts[0][1]], sampling, use_tqdm=False)

    runs = []
    for pair in range(args.pairs):
        pair_data = {
            policy: {
                "request_durations_s": {},
                "output_token_sha256_by_prompt": {},
                "output_lengths_by_prompt": {},
                "equal_reference_prompt_ids": [],
                "hybrid_routing": {},
                "spec_metrics_by_prompt": {},
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
            }
            for policy in POLICIES
        }
        prompt_orders = balanced_orders(
            len(prompts), args.randomization_seed + pair
        )
        order_by_prompt = {}
        gpu_before = gpu_snapshot()
        for (prompt_id, prompt, _), order in zip(
            prompts, prompt_orders, strict=True
        ):
            order_by_prompt[str(prompt_id)] = order
            for policy in order:
                configure(policy)
                routing_before = hybrid_routing_snapshot("hybrid_ngram_eagle3")
                metrics_before = metric_snapshot(llm)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = llm.generate([prompt], sampling, use_tqdm=False)[0]
                torch.cuda.synchronize()
                duration = time.perf_counter() - start
                metrics_after = metric_snapshot(llm)
                routing_after = hybrid_routing_snapshot("hybrid_ngram_eagle3")
                assert routing_before is not None and routing_after is not None
                token_ids = list(output.outputs[0].token_ids)
                data = pair_data[policy]
                data["request_durations_s"][str(prompt_id)] = duration
                data["output_token_sha256_by_prompt"][str(prompt_id)] = token_hash(
                    token_ids
                )
                data["output_lengths_by_prompt"][str(prompt_id)] = len(token_ids)
                if matches_reference(token_ids, reference[prompt_id], args.max_tokens):
                    data["equal_reference_prompt_ids"].append(prompt_id)
                add_counter_delta(
                    data["hybrid_routing"],
                    counter_dict_delta(routing_before, routing_after),
                )
                data["spec_metrics_by_prompt"][str(prompt_id)] = metric_delta(
                    metrics_before,
                    metrics_after,
                    POLICY_CONFIG[policy]["retrieval_k"],
                )
                data["peak_allocated_bytes"] = max(
                    data["peak_allocated_bytes"], torch.cuda.max_memory_allocated()
                )
                data["peak_reserved_bytes"] = max(
                    data["peak_reserved_bytes"], torch.cuda.max_memory_reserved()
                )
        gpu_after = gpu_snapshot()
        for policy in POLICIES:
            data = pair_data[policy]
            equal_ids = sorted(data["equal_reference_prompt_ids"])
            runs.append(
                {
                    "pair": pair,
                    "policy": policy,
                    "request_duration_total_s": sum(
                        data["request_durations_s"].values()
                    ),
                    "request_durations_s": data["request_durations_s"],
                    "output_token_sha256_by_prompt": data[
                        "output_token_sha256_by_prompt"
                    ],
                    "output_lengths_by_prompt": data["output_lengths_by_prompt"],
                    "equal_reference_prompt_ids": equal_ids,
                    "mismatch_reference_prompt_ids": sorted(
                        set(prompt_ids) - set(equal_ids)
                    ),
                    "hybrid_routing": data["hybrid_routing"],
                    "spec_metrics_by_prompt": data["spec_metrics_by_prompt"],
                    "peak_memory": {
                        "max_allocated_bytes": data["peak_allocated_bytes"],
                        "max_reserved_bytes": data["peak_reserved_bytes"],
                    },
                    "order_by_prompt": order_by_prompt,
                    "gpu_snapshot_before": gpu_before,
                    "gpu_snapshot_after": gpu_after,
                }
            )
            print(
                f"pair={pair} policy={policy} "
                f"total={runs[-1]['request_duration_total_s']:.3f}s "
                f"equal={len(equal_ids)}/{len(prompt_ids)}",
                flush=True,
            )

    report = {
        "protocol": (
            "One loaded K10-capacity engine; every request runs all four "
            "configurations in randomized orders balanced across ordinal "
            "positions. Shadow-sync policies isolate proposal budgets; the "
            "final policy adds accepted-token debt and one-step catch-up."
        ),
        "policy_config": POLICY_CONFIG,
        "comparisons": COMPARISONS,
        "model": args.model,
        "draft_model": args.draft_model,
        "dataset": "canitedit",
        "dataset_path": str(args.canitedit_path),
        "reference_eagle": str(args.reference_eagle),
        "prompt_ids": prompt_ids,
        "prompt_sha256_by_prompt": {
            str(prompt_id): hashlib.sha256(prompt.encode()).hexdigest()
            for prompt_id, prompt, _ in prompts
        },
        "max_tokens": args.max_tokens,
        "pairs": args.pairs,
        "randomization_seed": args.randomization_seed,
        "runs": runs,
    }
    report["summary"] = summarize(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
