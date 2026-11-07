#!/usr/bin/env python3
"""Deterministic reproduction harness for the paper's vLLM fork."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any

# The paper scripts require these before importing vLLM.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_V1", "1")

import torch
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.v1.metrics.reader import Counter, Vector


CODE_TASKS = [
    """You are editing Python code. Return only the complete corrected function. /no_think

Task: add a `limit` argument and stop after collecting that many even squares.

```python
def even_squares(values):
    result = []
    for value in values:
        if value % 2 == 0:
            result.append(value * value)
    return result
```
""",
    """Complete the function below. Return only Python code. /no_think

```python
def group_by_extension(paths):
    \"\"\"Return a dict mapping a lowercase extension to sorted paths.\"\"\"
    groups = {}
    for path in paths:
        # fill this section
```
""",
    """Fix the off-by-one bug and return only the corrected Java method. /no_think

```java
static int binarySearch(int[] values, int target) {
    int left = 0;
    int right = values.length;
    while (left <= right) {
        int middle = left + (right - left) / 2;
        if (values[middle] == target) return middle;
        if (values[middle] < target) left = middle + 1;
        else right = middle - 1;
    }
    return -1;
}
```
""",
    """Implement this TypeScript function. Return only the function. /no_think

```typescript
type User = { id: number; name: string; active: boolean };

function indexActiveUsers(users: User[]): Map<number, string> {
    // create a map containing only active users
}
```
""",
    """Refactor the loop into a safe context-manager implementation. Return only code. /no_think

```python
def count_nonempty_lines(filenames):
    total = 0
    for filename in filenames:
        handle = open(filename, "r", encoding="utf-8")
        for line in handle:
            if line.strip():
                total += 1
        handle.close()
    return total
```
""",
    """Write the missing Rust body. Return only the complete function. /no_think

```rust
fn deduplicate_sorted(values: &[i64]) -> Vec<i64> {
    // preserve ascending order and remove duplicates
}
```
""",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument(
        "--dataset",
        choices=("synthetic", "instructcoder", "canitedit", "negative_control"),
        default="synthetic",
    )
    parser.add_argument(
        "--instructcoder-path",
        type=Path,
        help=(
            "Optional pinned local InstructCoder JSON/JSONL/Parquet file. "
            "Rows must contain input and instruction fields."
        ),
    )
    parser.add_argument(
        "--negative-control-path",
        type=Path,
        default=Path(__file__).parent
        / "data"
        / "negative_control"
        / "general_instruction_qa_v1.json",
        help="Pinned general-instruction/QA prompt file for the P2 negative control.",
    )
    parser.add_argument(
        "--canitedit-path",
        type=Path,
        default=Path(__file__).parent
        / "data"
        / "canitedit"
        / "test-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--prompt-source-results",
        type=Path,
        help=(
            "Optional local paper-harness JSONL containing InstructCoder "
            "prompts. This avoids a network-dependent streaming reload."
        ),
    )
    parser.add_argument(
        "--canitedit-instruction",
        choices=("descriptive", "lazy"),
        default="descriptive",
    )
    parser.add_argument(
        "--method",
        choices=(
            "none",
            "ngram",
            "copyspec",
            "suffix",
            "eagle3",
            "draft_model",
            "hybrid_ngram_draft",
            "hybrid_ngram_eagle3",
        ),
        required=True,
    )
    parser.add_argument("--num-spec-tokens", type=int, default=3)
    parser.add_argument("--prompt-lookup-min", type=int, default=3)
    parser.add_argument("--prompt-lookup-max", type=int, default=7)
    parser.add_argument(
        "--ngram-retention-q",
        type=float,
        default=1.0,
        help=(
            "Retain probability for already-found natural N-gram hits. "
            "Used only for controlled hit-retention ablations; q never "
            "creates a hit."
        ),
    )
    parser.add_argument(
        "--copyspec-gamma",
        type=int,
        default=3,
        help=(
            "CopySpec's fixed suffix-key length. The same-engine port uses "
            "vLLM's KMP lookup instead of Python hashes but preserves the "
            "paper repository's first-occurrence candidate policy."
        ),
    )
    parser.add_argument(
        "--copyspec-copy-tokens",
        type=int,
        default=10,
        help="Maximum copied continuation length used by CopySpec.",
    )
    parser.add_argument("--suffix-cache-max-depth", type=int, default=16)
    parser.add_argument("--suffix-max-spec-factor", type=float, default=1.0)
    parser.add_argument("--suffix-min-token-prob", type=float, default=0.1)
    parser.add_argument("--hybrid-max-draft-skip-steps", type=int, default=0)
    parser.add_argument("--hybrid-selective-draft-subbatch", action="store_true")
    parser.add_argument("--hybrid-selective-min-saved-requests", type=int, default=1)
    parser.add_argument(
        "--hybrid-eagle-num-spec-tokens",
        type=int,
        default=3,
        help=(
            "Fixed EAGLE fallback K for hybrid_ngram_eagle3; "
            "--num-spec-tokens controls retrieval K."
        ),
    )
    parser.add_argument(
        "--profile-force-eagle-subbatch-size",
        type=int,
        help=(
            "Diagnostic only: force the first K requests through EAGLE while "
            "the remainder use N-gram (or no draft on a miss). Requires "
            "--hybrid-selective-draft-subbatch."
        ),
    )
    parser.add_argument(
        "--disable-padded-drafter-batch",
        action="store_true",
        help="Use the existing per-request CPU-list drafter input path.",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--batch-layout",
        choices=("replicate", "mixed"),
        default="replicate",
        help=(
            "replicate repeats one prompt within a batch, matching the paper's "
            "latency harness; mixed packs distinct consecutive prompts and is "
            "required for batch-aware routing studies."
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument(
        "--prompt-indices",
        type=int,
        nargs="+",
        help=(
            "Run only these zero-based prompt indices. For InstructCoder, "
            "sampling is still performed from index zero so IDs remain stable."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--warmup-batch-size",
        type=int,
        help=(
            "Optional explicit warm-up shape. Use the measured batch size for "
            "scheduler experiments whose first large batch may initialize "
            "shape-dependent runtime state."
        ),
    )
    parser.add_argument(
        "--respect-eos",
        action="store_true",
        help="Stop at EOS like the paper scripts; default forces fixed output length.",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--decision-round-trace", type=Path,
        help="Optional EngineCore round-level JSONL trace (profiling runs only).",
    )
    parser.add_argument(
        "--gpu-stage-trace", type=Path,
        help="Optional synchronized GPUModelRunner stage JSONL trace (profiling runs only).",
    )
    parser.add_argument(
        "--ngram-proposer-trace",
        type=Path,
        help="Optional N-gram lookup trace including natural and q-retained proposals.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[tuple[int, str, int | None]]:
    requested = args.prompt_indices
    sample_count = max(requested) + 1 if requested else args.num_prompts
    if requested and (min(requested) < 0 or len(set(requested)) != len(requested)):
        raise ValueError("--prompt-indices must be unique non-negative integers")
    if args.dataset == "synthetic":
        if not 1 <= sample_count <= len(CODE_TASKS):
            raise ValueError(
                f"requested synthetic prompts must be within 0..{len(CODE_TASKS) - 1}"
            )
        indices = requested or list(range(sample_count))
        return [(idx, CODE_TASKS[idx], None) for idx in indices]

    tokenizer = get_tokenizer(
        args.model, tokenizer_mode="auto", trust_remote_code=False
    )
    if args.dataset == "canitedit":
        import pyarrow.parquet as pq

        table = pq.read_table(args.canitedit_path)
        items = table.to_pylist()
        if sample_count > len(items):
            raise ValueError(
                f"CanItEdit contains {len(items)} rows, requested {sample_count}"
            )
        instruction_key = f"instruction_{args.canitedit_instruction}"
        indices = requested or list(range(sample_count))
        prompts = []
        for idx in indices:
            item = items[idx]
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are PythonEditGPT. You will be provided the original "
                        "code snippet and an instruction that specifies the changes "
                        "you need to make. You will produce the changed code, based "
                        "on the original code and the instruction given. Only "
                        "produce the code, do not include any additional prose."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"## Code Before\n```py\n{item['before']}\n```\n\n"
                        f"## Instruction\n{item[instruction_key]}\n\n## Code After"
                    ),
                },
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            prompts.append(
                (idx, prompt, len(tokenizer(prompt, add_special_tokens=False).input_ids))
            )
        return prompts

    if args.prompt_source_results is not None:
        rows = [
            json.loads(line)
            for line in args.prompt_source_results.read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        ]
        rows = [row for row in rows if row.get("batch_size") == 1]
        if len(rows) < sample_count:
            raise ValueError(
                f"prompt source has {len(rows)} batch=1 rows, need {sample_count}"
            )
        indices = requested or list(range(sample_count))
        return [
            (
                idx,
                rows[idx]["prompt"],
                len(
                    tokenizer(
                        rows[idx]["prompt"], add_special_tokens=False
                    ).input_ids
                ),
            )
            for idx in indices
        ]

    if args.instructcoder_path is not None or args.dataset == "negative_control":
        path = args.instructcoder_path or args.negative_control_path
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            items = pq.read_table(path).to_pylist()
        else:
            raw = path.read_text(encoding="utf-8")
            if path.suffix == ".jsonl":
                items = [json.loads(line) for line in raw.splitlines() if line]
            else:
                items = json.loads(raw)
        if not isinstance(items, list) or any(
            not isinstance(item, dict)
            or "input" not in item
            or "instruction" not in item
            for item in items
        ):
            raise ValueError(
                "local InstructCoder data must be a list of rows containing "
                "input and instruction"
            )
        random.Random(42).shuffle(items)
        if sample_count > len(items):
            raise ValueError(
                f"InstructCoder contains {len(items)} rows, requested {sample_count}"
            )
        indices = requested or list(range(sample_count))
        prompts = []
        for idx in indices:
            item = items[idx]
            content = (
                f"{item['input']}\n\n{item['instruction']} Just output "
                "the code, do not include any explanation."
            )
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            prompts.append(
                (
                    idx,
                    prompt,
                    len(tokenizer(prompt, add_special_tokens=False).input_ids),
                )
            )
        return prompts

    from vllm.benchmarks.datasets_bench import InstructCoderDataset

    dataset = InstructCoderDataset(
        dataset_path="likaixin/InstructCoder",
        dataset_subset=None,
        dataset_split="train",
        random_seed=42,
    )
    requests = dataset.sample(
        num_requests=sample_count,
        tokenizer=tokenizer,
        output_len=None,
    )
    indices = requested or list(range(sample_count))
    return [
        (idx, requests[idx].prompt, requests[idx].prompt_len) for idx in indices
    ]


def get_speculative_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.method == "none":
        return None
    if args.method == "ngram":
        return {
            "method": "ngram",
            "num_speculative_tokens": args.num_spec_tokens,
            "prompt_lookup_min": args.prompt_lookup_min,
            "prompt_lookup_max": args.prompt_lookup_max,
        }
    if args.method == "copyspec":
        if args.copyspec_gamma < 1:
            raise ValueError("--copyspec-gamma must be positive")
        if args.copyspec_copy_tokens < 1:
            raise ValueError("--copyspec-copy-tokens must be positive")
        # CopySpec uses the first previous occurrence of the fixed-length
        # suffix and copies its following tokens. vLLM's N-gram proposer has
        # the same candidate semantics when min=max=gamma; only the lookup
        # data structure differs (KMP rather than Python's salted hash).
        return {
            "method": "ngram",
            "num_speculative_tokens": args.copyspec_copy_tokens,
            "prompt_lookup_min": args.copyspec_gamma,
            "prompt_lookup_max": args.copyspec_gamma,
        }
    if args.method == "suffix":
        return {
            "method": "suffix",
            "suffix_cache_max_depth": args.suffix_cache_max_depth,
            "suffix_speculative_tokens": args.suffix_cache_max_depth,
            "suffix_max_spec_factor": args.suffix_max_spec_factor,
            "suffix_min_token_prob": args.suffix_min_token_prob,
        }
    if not args.draft_model:
        raise ValueError(f"--draft-model is required for {args.method}")
    if args.method == "eagle3":
        return {
            "method": "eagle3",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_spec_tokens,
            "disable_padded_drafter_batch": args.disable_padded_drafter_batch,
        }
    if args.method == "hybrid_ngram_eagle3":
        allowed_retrieval_k = (3, 6, 8, 10, 16)
        if args.num_spec_tokens not in allowed_retrieval_k:
            raise ValueError(
                "hybrid retrieval --num-spec-tokens must be one of "
                f"{allowed_retrieval_k}, got {args.num_spec_tokens}"
            )
        return {
            "method": "hybrid_ngram_eagle3",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_spec_tokens,
            "prompt_lookup_min": args.prompt_lookup_min,
            "prompt_lookup_max": args.prompt_lookup_max,
            # N-gram lookup consumes per-request CPU token lists. EAGLE-3
            # supports this existing non-padded path as well.
            "disable_padded_drafter_batch": True,
            "hybrid_max_draft_skip_steps": args.hybrid_max_draft_skip_steps,
            "hybrid_selective_draft_subbatch": (
                args.hybrid_selective_draft_subbatch
            ),
            "hybrid_selective_min_saved_requests": (
                args.hybrid_selective_min_saved_requests
            ),
            "hybrid_eagle_num_speculative_tokens": (
                args.hybrid_eagle_num_spec_tokens
            ),
        }
    return {
        "method": args.method,
        "model": args.draft_model,
        "num_speculative_tokens": args.num_spec_tokens,
        "disable_padded_drafter_batch": True,
        "max_model_len": args.max_model_len,
        **(
            {
                "hybrid_max_draft_skip_steps": args.hybrid_max_draft_skip_steps
            }
            if args.method == "hybrid_ngram_draft"
            else {}
        ),
    }


def metric_key(metric: Any) -> tuple[str, tuple[tuple[str, str], ...]]:
    return metric.name, tuple(sorted(metric.labels.items()))


def metric_snapshot(llm: LLM) -> dict[tuple[str, tuple], Any]:
    return {metric_key(metric): metric for metric in llm.get_metrics()}


def metric_delta(before: dict, after: dict, num_spec_tokens: int) -> dict[str, Any]:
    counters = {
        "vllm:spec_decode_num_drafts": "num_drafts",
        "vllm:spec_decode_num_draft_tokens": "num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens": "num_accepted_tokens",
    }
    result: dict[str, Any] = {name: 0 for name in counters.values()}
    accepted_per_pos = [0] * num_spec_tokens
    for key, current in after.items():
        previous = before.get(key)
        if current.name in counters and isinstance(current, Counter):
            old_value = previous.value if isinstance(previous, Counter) else 0
            result[counters[current.name]] += current.value - old_value
        elif (
            current.name == "vllm:spec_decode_num_accepted_tokens_per_pos"
            and isinstance(current, Vector)
        ):
            old_values = previous.values if isinstance(previous, Vector) else []
            for pos, value in enumerate(current.values[:num_spec_tokens]):
                old_value = old_values[pos] if pos < len(old_values) else 0
                accepted_per_pos[pos] += value - old_value
    drafts = result["num_drafts"]
    draft_tokens = result["num_draft_tokens"]
    accepted = result["num_accepted_tokens"]
    result.update(
        {
            "accepted_per_pos": accepted_per_pos,
            "acceptance_rate": accepted / draft_tokens if draft_tokens else 0.0,
            "mean_acceptance_length": 1 + accepted / drafts if drafts else 1.0,
        }
    )
    return result


def hybrid_routing_snapshot(method: str) -> dict[str, int] | None:
    if method not in ("hybrid_ngram_draft", "hybrid_ngram_eagle3"):
        return None
    from vllm.v1.spec_decode.hybrid_proposer import get_hybrid_routing_stats

    return get_hybrid_routing_stats()


def counter_dict_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before.get(key, 0) for key in after}


def build_workloads(
    prompts: list[tuple[int, str, int | None]], batch_size: int, layout: str
) -> list[tuple[int, list[tuple[int, str, int | None]]]]:
    """Build paper-compatible replicated or real heterogeneous batches."""
    if layout == "replicate":
        return [
            (prompt_id, [(prompt_id, prompt, prompt_len)] * batch_size)
            for prompt_id, prompt, prompt_len in prompts
        ]
    if len(prompts) % batch_size:
        raise ValueError(
            "mixed batches require --num-prompts (or selected --prompt-indices) "
            "to be divisible by every requested --batch-sizes value; refusing "
            "a partial final batch because it changes the comparison."
        )
    return [
        (batch_id, prompts[start : start + batch_size])
        for batch_id, start in enumerate(range(0, len(prompts), batch_size))
    ]


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.ngram_retention_q <= 1.0:
        raise ValueError("--ngram-retention-q must be within [0, 1]")
    metric_spec_tokens = (
        args.suffix_cache_max_depth
        if args.method == "suffix"
        else (
            args.copyspec_copy_tokens
            if args.method == "copyspec"
            else args.num_spec_tokens
        )
    )
    if args.profile_force_eagle_subbatch_size is not None:
        if not args.hybrid_selective_draft_subbatch:
            raise ValueError(
                "--profile-force-eagle-subbatch-size requires "
                "--hybrid-selective-draft-subbatch"
            )
        if args.profile_force_eagle_subbatch_size < 0:
            raise ValueError("forced EAGLE sub-batch size must be non-negative")
        os.environ["VLLM_PROFILE_FORCE_EAGLE_SUBBATCH_SIZE"] = str(
            args.profile_force_eagle_subbatch_size
        )
    for path, variable in (
        (args.decision_round_trace, "VLLM_DECISION_ROUND_TRACE_PATH"),
        (args.gpu_stage_trace, "VLLM_GPU_STAGE_TRACE_PATH"),
        (args.ngram_proposer_trace, "VLLM_NGRAM_PROPOSER_TRACE_PATH"),
    ):
        if path is None:
            continue
        if path.exists() and path.stat().st_size:
            raise ValueError(f"trace already contains data: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path.resolve())
    # Set before vLLM constructs NgramProposer.  It is harmless for methods
    # that do not use retrieval and is recorded in the output environment.
    os.environ["VLLM_NGRAM_RETENTION_Q"] = str(args.ngram_retention_q)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        speculative_config=get_speculative_config(args),
        disable_log_stats=False,
        enable_prefix_caching=False,
        seed=42,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=not args.respect_eos,
    )

    # Exclude model load and graph warm-up from all timed measurements.
    warmup_batch_size = (
        args.warmup_batch_size
        if args.warmup_batch_size is not None
        else max(1, args.profile_force_eagle_subbatch_size or 0)
    )
    if warmup_batch_size <= 0:
        raise ValueError("warmup batch size must be positive")
    llm.generate(
        [prompts[0][1]] * warmup_batch_size, sampling, use_tqdm=False
    )
    # Profiling files intentionally describe measured workloads only.  The
    # warm-up uses the same hooks but is removed before the timed loop.
    for trace_path in (
        args.decision_round_trace,
        args.gpu_stage_trace,
        args.ngram_proposer_trace,
    ):
        if trace_path is not None:
            trace_path.write_text("", encoding="utf-8")

    with args.output.open("w", encoding="utf-8") as handle:
        for batch_size in args.batch_sizes:
            workloads = build_workloads(prompts, batch_size, args.batch_layout)
            for workload_id, members in workloads:
                prompt_ids = [member[0] for member in members]
                batch_prompts = [member[1] for member in members]
                prompt_sha256s = [
                    hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    for prompt in batch_prompts
                ]
                prompt_lens = [member[2] for member in members]
                durations: list[float] = []
                last_outputs = None
                metrics = None
                hybrid_routing = None
                for _ in range(args.repeats):
                    before = metric_snapshot(llm) if args.method != "none" else {}
                    routing_before = hybrid_routing_snapshot(args.method)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    last_outputs = llm.generate(batch_prompts, sampling, use_tqdm=False)
                    torch.cuda.synchronize()
                    durations.append(time.perf_counter() - start)
                    if args.method != "none":
                        metrics = metric_delta(
                            before, metric_snapshot(llm), metric_spec_tokens
                        )
                    if routing_before is not None:
                        routing_after = hybrid_routing_snapshot(args.method)
                        assert routing_after is not None
                        hybrid_routing = counter_dict_delta(
                            routing_before, routing_after
                        )

                assert last_outputs is not None
                output_token_ids = [
                    list(output.outputs[0].token_ids) for output in last_outputs
                ]
                output_lengths = [len(ids) for ids in output_token_ids]
                duration = statistics.median(durations)
                record = {
                    "method": args.method,
                    "method_provenance": (
                        {
                            "paper": "CopySpec",
                            "official_repository": (
                                "https://github.com/RazvanDu/CopySpec"
                            ),
                            "official_commit": (
                                "9b133970838de26cd8a33fb844349f44a0a1a050"
                            ),
                            "port_scope": (
                                "candidate-equivalent fixed-gamma first-occurrence "
                                "lookup on the common vLLM verifier"
                            ),
                            "gamma": args.copyspec_gamma,
                            "copy_tokens": args.copyspec_copy_tokens,
                        }
                        if args.method == "copyspec"
                        else None
                    ),
                    "num_spec_tokens": (
                        metric_spec_tokens if args.method != "none" else None
                    ),
                    "hybrid_eagle_num_spec_tokens": (
                        args.hybrid_eagle_num_spec_tokens
                        if args.method == "hybrid_ngram_eagle3"
                        else None
                    ),
                    "ngram_retention_q": args.ngram_retention_q,
                    "ngram_trace_path": (
                        str(args.ngram_proposer_trace.resolve())
                        if args.ngram_proposer_trace is not None
                        else None
                    ),
                    # `prompt_id` is preserved for the replicated paper layout.
                    # For mixed layout it is the batch/workload index; consumers
                    # must use prompt_ids to identify its distinct members.
                    "prompt_id": workload_id,
                    "prompt_ids": prompt_ids,
                    "prompt_sha256s": prompt_sha256s,
                    "dataset": args.dataset,
                    "canitedit_instruction": (
                        args.canitedit_instruction
                        if args.dataset == "canitedit"
                        else None
                    ),
                    "batch_layout": args.batch_layout,
                    "respect_eos": args.respect_eos,
                    "prompt_len": prompt_lens[0] if args.batch_layout == "replicate" else None,
                    "prompt_lens": prompt_lens,
                    "batch_size": batch_size,
                    "duration_s_samples": durations,
                    "duration_s_median": duration,
                    "output_lengths": output_lengths,
                    "output_tokens_per_second": sum(output_lengths) / duration,
                    "output_token_ids": output_token_ids,
                    "generated_text": last_outputs[0].outputs[0].text,
                    "generated_texts": [
                        output.outputs[0].text for output in last_outputs
                    ],
                    "finish_reasons": [
                        output.outputs[0].finish_reason for output in last_outputs
                    ],
                    "metrics": metrics,
                    "hybrid_routing": hybrid_routing,
                    "environment": {
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "cuda_wheel": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0),
                        "gpu_capability": torch.cuda.get_device_capability(0),
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"method={args.method} layout={args.batch_layout} "
                    f"workload={workload_id} prompts={prompt_ids} batch={batch_size} "
                    f"median={duration:.4f}s throughput="
                    f"{record['output_tokens_per_second']:.2f} tok/s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
