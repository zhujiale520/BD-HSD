# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for composing lossless speculative token proposers."""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch

from vllm.v1.attention.backends.utils import CommonAttentionMetadata


@dataclass
class HybridRoutingStats:
    routing_rounds: int = 0
    all_ngram_hit_rounds: int = 0
    partial_ngram_hit_rounds: int = 0
    zero_ngram_hit_rounds: int = 0
    active_request_rounds: int = 0
    selective_draft_subbatch_rounds: int = 0
    selective_draft_full_batch_rounds: int = 0
    selective_draft_requests_avoided: int = 0
    ngram_requests: int = 0
    draft_model_requests: int = 0
    ngram_proposed_tokens: int = 0
    draft_model_proposed_tokens: int = 0
    draft_model_executed_requests: int = 0
    draft_model_skipped_requests: int = 0
    compact_plan_cache_hits: int = 0
    compact_plan_cache_misses: int = 0
    compact_plan_cache_evictions: int = 0
    compact_plan_uncached_builds: int = 0


_routing_stats = HybridRoutingStats()


def get_hybrid_routing_stats() -> dict[str, int]:
    """Return a process-local cumulative routing snapshot."""
    return asdict(_routing_stats)


def record_draft_model_execution(num_requests: int) -> None:
    _routing_stats.draft_model_executed_requests += num_requests


def record_ngram_batch_availability(
    ngram_token_ids: Sequence[Sequence[int]],
) -> None:
    """Record whether the active batch has all, some, or no N-gram hits."""
    num_requests = len(ngram_token_ids)
    num_hits = sum(bool(token_ids) for token_ids in ngram_token_ids)
    _routing_stats.routing_rounds += 1
    _routing_stats.active_request_rounds += num_requests
    if num_hits == num_requests:
        _routing_stats.all_ngram_hit_rounds += 1
    elif num_hits:
        _routing_stats.partial_ngram_hit_rounds += 1
    else:
        _routing_stats.zero_ngram_hit_rounds += 1


def record_selective_draft_batch(active_requests: int, draft_requests: int) -> None:
    """Record the shape chosen by the selective draft cost gate."""
    if not 0 < draft_requests <= active_requests:
        raise ValueError("draft request count must be within the active batch")
    if draft_requests < active_requests:
        _routing_stats.selective_draft_subbatch_rounds += 1
        _routing_stats.selective_draft_requests_avoided += (
            active_requests - draft_requests
        )
    else:
        _routing_stats.selective_draft_full_batch_rounds += 1


def select_draft_request_indices(
    ngram_token_ids: Sequence[Sequence[int]],
    draft_debt_mask: Sequence[bool],
) -> list[int]:
    """Return requests that miss N-gram or must catch up draft state."""
    if len(ngram_token_ids) != len(draft_debt_mask):
        raise ValueError("N-gram proposals and draft debt mask must have same size")
    return [
        index
        for index, (token_ids, has_debt) in enumerate(
            zip(ngram_token_ids, draft_debt_mask)
        )
        if not token_ids or has_debt
    ]


def record_skipped_state_debt(
    debt_tokens_by_req: dict[str, int],
    skipped_req_ids: set[str],
    committed_tokens_by_req: dict[str, int],
) -> dict[str, int]:
    """Charge skipped EAGLE state by tokens actually committed this round.

    ``committed_tokens_by_req`` is produced by rejection sampling before the
    next proposal is routed, so it is the exact target-prefix advance that the
    skipped EAGLE first pass would otherwise consume. Requests that have left
    the active batch are pruned at the same time.
    """
    return {
        req_id: debt_tokens_by_req.get(req_id, 0)
        + (committed if req_id in skipped_req_ids else 0)
        for req_id, committed in committed_tokens_by_req.items()
    }


def gather_ragged_token_indices(
    query_start_loc: Sequence[int], request_indices: Sequence[int]
) -> list[int]:
    """Expand selected request rows into their flattened ragged token indices."""
    if len(query_start_loc) < 2:
        raise ValueError("query_start_loc must contain at least [0, end]")
    if query_start_loc[0] != 0 or any(
        right < left for left, right in zip(query_start_loc, query_start_loc[1:])
    ):
        raise ValueError("query_start_loc must be monotonic and start at zero")
    num_requests = len(query_start_loc) - 1
    if any(index < 0 or index >= num_requests for index in request_indices):
        raise ValueError("request index is outside query_start_loc")
    if len(set(request_indices)) != len(request_indices):
        raise ValueError("request indices must be unique")
    return [
        token_index
        for request_index in request_indices
        for token_index in range(
            query_start_loc[request_index], query_start_loc[request_index + 1]
        )
    ]


@dataclass(frozen=True)
class _CompactAttentionPlan:
    """Immutable index topology reused by selective EAGLE compaction."""

    cpu_request_indices: torch.Tensor
    device_request_indices: torch.Tensor
    token_indices: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    query_start_loc: torch.Tensor
    last_token_indices: torch.Tensor
    max_query_len: int


CompactAttentionPlanCache = OrderedDict[tuple, _CompactAttentionPlan]


def _get_compact_attention_plan(
    metadata: CommonAttentionMetadata,
    request_indices: list[int],
    plan_cache: CompactAttentionPlanCache | None,
    max_cached_plans: int,
) -> _CompactAttentionPlan:
    query_start_values = tuple(metadata.query_start_loc_cpu.tolist())
    cache_key = (tuple(request_indices), query_start_values)
    if plan_cache is not None and cache_key in plan_cache:
        _routing_stats.compact_plan_cache_hits += 1
        plan = plan_cache.pop(cache_key)
        plan_cache[cache_key] = plan
        return plan

    if plan_cache is None:
        _routing_stats.compact_plan_uncached_builds += 1
    else:
        _routing_stats.compact_plan_cache_misses += 1
    ragged_indices = gather_ragged_token_indices(
        query_start_values, request_indices
    )
    cpu_request_indices = torch.tensor(request_indices, dtype=torch.long)
    device_request_indices = cpu_request_indices.to(metadata.seq_lens.device)
    token_indices = torch.tensor(
        ragged_indices, dtype=torch.long, device=metadata.slot_mapping.device
    )
    query_lengths_cpu = (
        metadata.query_start_loc_cpu[1:] - metadata.query_start_loc_cpu[:-1]
    ).index_select(0, cpu_request_indices)
    query_start_loc_cpu = torch.zeros(
        len(request_indices) + 1, dtype=metadata.query_start_loc_cpu.dtype
    )
    torch.cumsum(query_lengths_cpu, dim=0, out=query_start_loc_cpu[1:])
    query_start_loc = query_start_loc_cpu.to(metadata.query_start_loc.device)
    plan = _CompactAttentionPlan(
        cpu_request_indices=cpu_request_indices,
        device_request_indices=device_request_indices,
        token_indices=token_indices,
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc=query_start_loc,
        last_token_indices=query_start_loc[1:] - 1,
        max_query_len=int(query_lengths_cpu.max().item()),
    )
    if plan_cache is not None and max_cached_plans > 0:
        plan_cache[cache_key] = plan
        if len(plan_cache) > max_cached_plans:
            plan_cache.popitem(last=False)
            _routing_stats.compact_plan_cache_evictions += 1
    return plan


def compact_common_attention_metadata(
    metadata: CommonAttentionMetadata,
    request_indices: Sequence[int],
    plan_cache: CompactAttentionPlanCache | None = None,
    max_cached_plans: int = 128,
) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
    """Compact common attention metadata to a request sub-batch.

    Returns the compact metadata, flattened token indices into tensors ordered
    like the original metadata, and last-token indices in compact coordinates.
    The first prototype deliberately excludes cross attention and DCP.
    """
    request_indices = list(request_indices)
    if not request_indices:
        raise ValueError("selective draft sub-batch must contain a request")
    if (
        metadata.encoder_seq_lens is not None
        or metadata.dcp_local_seq_lens is not None
    ):
        raise ValueError(
            "selective draft sub-batches do not support cross attention or DCP"
        )
    plan = _get_compact_attention_plan(
        metadata, request_indices, plan_cache, max_cached_plans
    )
    seq_lens_cpu = metadata.seq_lens_cpu.index_select(
        0, plan.cpu_request_indices
    )
    seq_lens = metadata.seq_lens.index_select(0, plan.device_request_indices)
    num_computed_tokens_cpu = metadata.num_computed_tokens_cpu.index_select(
        0, plan.cpu_request_indices
    )

    compact = CommonAttentionMetadata(
        query_start_loc=plan.query_start_loc,
        query_start_loc_cpu=plan.query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=len(request_indices),
        num_actual_tokens=plan.token_indices.numel(),
        max_query_len=plan.max_query_len,
        max_seq_len=int(seq_lens_cpu.max().item()),
        block_table_tensor=metadata.block_table_tensor.index_select(
            0, plan.device_request_indices
        ),
        slot_mapping=metadata.slot_mapping.index_select(0, plan.token_indices),
        causal=metadata.causal,
    )
    return compact, plan.token_indices, plan.last_token_indices


def merge_ngram_and_compact_draft_proposals(
    ngram_token_ids: Sequence[Sequence[int]],
    draft_request_indices: Sequence[int],
    compact_draft_token_ids: Sequence[Sequence[int]] | torch.Tensor,
) -> list[list[int]]:
    """Scatter a compact draft sub-batch and prefer N-gram proposals on hits."""
    if isinstance(compact_draft_token_ids, torch.Tensor):
        compact_draft_token_ids = compact_draft_token_ids.tolist()
    if len(draft_request_indices) != len(compact_draft_token_ids):
        raise ValueError(
            "draft request indices and compact proposals must have same size"
        )
    if len(set(draft_request_indices)) != len(draft_request_indices):
        raise ValueError("draft request indices must be unique")
    compact_by_request = dict(zip(draft_request_indices, compact_draft_token_ids))
    if any(index < 0 or index >= len(ngram_token_ids) for index in compact_by_request):
        raise ValueError("draft request index is outside the N-gram batch")

    merged = []
    for index, ngram_ids in enumerate(ngram_token_ids):
        if ngram_ids:
            proposal = list(ngram_ids)
            _routing_stats.ngram_requests += 1
            _routing_stats.ngram_proposed_tokens += len(proposal)
            merged.append(proposal)
        elif index in compact_by_request:
            proposal = list(compact_by_request[index])
            _routing_stats.draft_model_requests += 1
            _routing_stats.draft_model_proposed_tokens += len(proposal)
            merged.append(proposal)
        else:
            raise ValueError(f"request {index} has neither N-gram nor draft proposal")
    return merged


def select_ngram_without_draft(
    ngram_token_ids: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Record and return an all-N-gram batch when the draft step is skipped."""
    if not all(ngram_token_ids):
        raise ValueError("draft can be skipped only when every request has an N-gram")
    proposals = [list(token_ids) for token_ids in ngram_token_ids]
    _routing_stats.ngram_requests += len(proposals)
    _routing_stats.ngram_proposed_tokens += sum(map(len, proposals))
    _routing_stats.draft_model_skipped_requests += len(proposals)
    return proposals


def select_retrieval_only(
    ngram_token_ids: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Return retrieval hits and empty proposals without executing fallback."""
    proposals = [list(token_ids) for token_ids in ngram_token_ids]
    hits = [proposal for proposal in proposals if proposal]
    _routing_stats.ngram_requests += len(hits)
    _routing_stats.ngram_proposed_tokens += sum(map(len, hits))
    _routing_stats.draft_model_skipped_requests += len(proposals)
    return proposals


def merge_ngram_and_draft_proposals(
    ngram_token_ids: Sequence[Sequence[int]],
    draft_token_ids: Sequence[Sequence[int]] | torch.Tensor,
) -> list[list[int]]:
    """Prefer an N-gram proposal and fall back to the draft-model proposal.

    The draft proposer is intentionally executed for every request in the
    initial shadow-synchronized implementation. This keeps its KV cache valid
    while the integration and acceptance behavior are measured end to end.
    """
    if isinstance(draft_token_ids, torch.Tensor):
        draft_token_ids = draft_token_ids.tolist()
    if len(ngram_token_ids) != len(draft_token_ids):
        raise ValueError(
            "N-gram and draft proposal batches must have the same size: "
            f"{len(ngram_token_ids)} != {len(draft_token_ids)}"
        )
    merged = []
    for ngram_ids, draft_ids in zip(ngram_token_ids, draft_token_ids):
        if ngram_ids:
            proposal = list(ngram_ids)
            _routing_stats.ngram_requests += 1
            _routing_stats.ngram_proposed_tokens += len(proposal)
        else:
            proposal = list(draft_ids)
            _routing_stats.draft_model_requests += 1
            _routing_stats.draft_model_proposed_tokens += len(proposal)
        merged.append(proposal)
    return merged
