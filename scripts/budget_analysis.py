"""Pure-Python summaries for the BD-HSD budget ablation."""

from __future__ import annotations

import random
import statistics
from typing import Any


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


def sample_summary(samples: list[float], seed: int) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one paired sample is required")
    rng = random.Random(seed)
    bootstrap_medians = sorted(
        statistics.median(rng.choices(samples, k=len(samples)))
        for _ in range(10_000)
    )
    result: dict[str, Any] = {
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
        "bootstrap_resamples": 10_000,
    }
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        result["wilcoxon_two_sided_p"] = None
        result["wilcoxon_greater_p"] = None
    else:
        result["wilcoxon_two_sided_p"] = float(
            wilcoxon(samples, alternative="two-sided", method="exact").pvalue
        )
        result["wilcoxon_greater_p"] = float(
            wilcoxon(samples, alternative="greater", method="exact").pvalue
        )
    return result


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    details = []
    for pair in sorted({int(run["pair"]) for run in runs}):
        pair_runs = {run["policy"]: run for run in runs if run["pair"] == pair}
        missing = set(POLICIES) - set(pair_runs)
        if missing:
            raise ValueError(f"pair {pair} is missing policies: {sorted(missing)}")
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
        result["comparisons"][name] = sample_summary(samples, 20260913 + index)
    return result
