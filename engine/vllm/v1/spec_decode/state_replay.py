# SPDX-License-Identifier: Apache-2.0
"""Eager request-indexed EAGLE target-feature replay, experimental.

Positions describe target hidden-state source rows. EAGLE consumes the shifted
next token at each such position. Only accepted rows may be retained.
"""
from dataclasses import replace
import torch


def get_replay_consume_reason(ready_flags, retrieval_hit_flags, kv_only_hits):
    """Choose state-only consumption without omitting needed neural candidates."""
    if not ready_flags or len(ready_flags) != len(retrieval_hit_flags):
        raise ValueError('replay readiness/hits must have the same nonempty size')
    if not all(ready_flags):
        return 'incomplete_prefix'
    if kv_only_hits and all(retrieval_hit_flags):
        return 'retrieval_hits'
    return None


class EagleReplay:
    def __init__(self, rebuild_until=0):
        self.rebuild_until = rebuild_until
        self.chunks = []
        self.consumed_end = None
        self.rounds = 0
        self.debt = 0

    def append(self, token_ids, positions, hidden_states, next_token_ids, committed,
               *, is_decode=True, round_id=None):
        if positions.ndim != 1 or next_token_ids.numel() != 1:
            raise ValueError('state replay supports one text request only')
        if token_ids.numel() == 0 or token_ids.numel() != positions.numel():
            raise ValueError('empty or misaligned replay input')
        start, end = int(positions[0].item()), int(positions[-1].item()) + 1
        if not torch.equal(positions, torch.arange(start, end, device=positions.device)):
            raise RuntimeError('noncontiguous source positions')
        if self.consumed_end is None and not self.chunks and start != 0:
            raise RuntimeError('new replay requires prefix reconstruction from zero')
        expected = self.chunks[-1][4] if self.chunks else self.consumed_end
        if expected is not None and start != expected:
            raise RuntimeError(f'replay prefix gap/overlap: {start} != {expected}')
        if hidden_states.shape[0] != token_ids.numel():
            raise ValueError('hidden-state row mismatch')
        if is_decode and self.consumed_end is not None and token_ids.numel() != committed:
            raise RuntimeError('accepted rows disagree with committed count')
        # clone: runner input/hidden-state buffers are overwritten next round.
        self.chunks.append((token_ids.clone(), positions.clone(), hidden_states.clone(),
                            next_token_ids.clone(), end, round_id))
        self.debt += committed

    def can_defer(self, hit, round_bound, token_bound, capacity):
        rows = sum(c[0].numel() for c in self.chunks)
        return (self.consumed_end is not None
                and self.consumed_end >= self.rebuild_until and hit
                and (round_bound == -1 or self.rounds < round_bound)
                and (token_bound is None or self.debt <= token_bound)
                # Reserve maximum next-round accepted rows before deferring.
                and rows + capacity[1] <= capacity[0])

    def inputs(self, cad, block_size):
        ids = torch.cat([c[0] for c in self.chunks])
        positions = torch.cat([c[1] for c in self.chunks])
        features = torch.cat([c[2] for c in self.chunks])
        # Validate shifted boundary tokens: last correction becomes next input.
        for left, right in zip(self.chunks, self.chunks[1:]):
            if left[3].item() != right[0][0].item():
                raise RuntimeError('shifted token boundary mismatch')
        n = ids.numel()
        end = self.chunks[-1][4]
        start = int(positions[0].item())
        if int(cad.seq_lens_cpu[0]) != end:
            raise RuntimeError('replay endpoint disagrees with target sequence')
        q_cpu = torch.tensor([0, n], dtype=torch.int32)
        seq_cpu = torch.tensor([end], dtype=cad.seq_lens_cpu.dtype)
        block_ids = cad.block_table_tensor[0, positions // block_size]
        slots = block_ids * block_size + positions % block_size
        cad = replace(cad, query_start_loc=q_cpu.to(ids.device),
                      query_start_loc_cpu=q_cpu, seq_lens=seq_cpu.to(ids.device),
                      seq_lens_cpu=seq_cpu, num_computed_tokens_cpu=torch.tensor([start]),
                      num_actual_tokens=n, max_query_len=n, max_seq_len=end,
                      slot_mapping=slots, logits_indices_padded=None,
                      num_logits_indices=None)
        return ids, positions, features, self.chunks[-1][3], cad

    def acknowledge(self, input_rows):
        if input_rows != sum(c[0].numel() for c in self.chunks):
            raise RuntimeError('EAGLE first-pass consumption mismatch')
        self.consumed_end = self.chunks[-1][4]
        self.chunks.clear()
        self.rounds = self.debt = 0


def append_replay_batch(states, req_ids, token_ids, positions, features, next_ids,
                        cad, committed, prompt_lengths, rebuild_until=None, *, round_id=None):
    """Split accepted source rows by current request order, never stale indices."""
    offsets = cad.query_start_loc_cpu.tolist()
    if len(offsets) != len(req_ids) + 1 or offsets[-1] != token_ids.numel():
        raise RuntimeError('replay batch query offsets disagree with accepted rows')
    active = []
    source = {}
    for i, req_id in enumerate(req_ids):
        lo, hi = offsets[i:i + 2]
        if hi <= lo:
            raise RuntimeError('replay does not support empty scheduled queries')
        boundary = max(prompt_lengths[req_id], (rebuild_until or {}).get(req_id, 0))
        state = states.setdefault(req_id, EagleReplay(rebuild_until=boundary))
        start = int(positions[lo].item())
        state.append(token_ids[lo:hi], positions[lo:hi], features[lo:hi],
                     next_ids[i:i + 1], committed[req_id],
                     is_decode=start >= state.rebuild_until, round_id=round_id)
        source[req_id] = dict(start=start, end=int(positions[hi - 1].item()) + 1,
                              current_rows=hi - lo, rebuild_until=state.rebuild_until)
        active.append(state)
    return active, source


def can_defer_batch(states, hits, round_bound, token_bound, max_rows, lookahead):
    """Conservative whole-batch policy; any miss/bound/init forces all to run."""
    if len(states) != len(hits) or not states:
        raise ValueError('replay states/hits batch mismatch')
    rows = sum(sum(c[0].numel() for c in s.chunks) for s in states)
    return (rows + len(states) * lookahead <= max_rows
            and all(s.can_defer(bool(hit), round_bound, token_bound,
                                (max_rows, 0)) for s, hit in zip(states, hits)))


def prepare_replay_batch(states, req_ids, cad, block_size):
    """Rebuild flattened queries in current order with current per-request KV."""
    if cad.dcp_local_seq_lens is not None:
        raise ValueError('state replay does not support decode context parallelism')
    parts = []
    audit = {}
    offsets = [0]
    for i, (req_id, state) in enumerate(zip(req_ids, states)):
        local = replace(cad, seq_lens=cad.seq_lens[i:i + 1],
                        seq_lens_cpu=cad.seq_lens_cpu[i:i + 1],
                        block_table_tensor=cad.block_table_tensor[i:i + 1],
                        num_reqs=1)
        ids, pos, hidden, next_id, local = state.inputs(local, block_size)
        parts.append((ids, pos, hidden, next_id, local))
        offsets.append(offsets[-1] + ids.numel())
        audit[req_id] = dict(start=int(pos[0].item()), end=int(pos[-1].item()) + 1,
                             rows=ids.numel(), pending_debt=state.debt)
    if not parts or len(parts) != len(req_ids):
        raise RuntimeError('replay batch metadata mismatch')
    q_cpu = torch.tensor(offsets, dtype=torch.int32)
    rebuilt = replace(cad, query_start_loc=q_cpu.to(cad.query_start_loc.device),
        query_start_loc_cpu=q_cpu,
        seq_lens=torch.cat([p[4].seq_lens for p in parts]),
        seq_lens_cpu=torch.cat([p[4].seq_lens_cpu for p in parts]),
        num_computed_tokens_cpu=torch.cat([p[4].num_computed_tokens_cpu for p in parts]),
        num_actual_tokens=offsets[-1], max_query_len=max(p[0].numel() for p in parts),
        max_seq_len=max(int(p[4].seq_lens_cpu[0]) for p in parts),
        slot_mapping=torch.cat([p[4].slot_mapping for p in parts]),
        logits_indices_padded=None, num_logits_indices=None)
    return (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]),
            torch.cat([p[2] for p in parts]), torch.cat([p[3] for p in parts]),
            rebuilt, audit)


def acknowledge_replay_batch(states, input_rows):
    """Validate total first-pass consumption before clearing any request."""
    counts = [sum(c[0].numel() for c in s.chunks) for s in states]
    if sum(counts) != input_rows:
        raise RuntimeError('EAGLE batch first-pass consumption mismatch')
    for state, count in zip(states, counts):
        state.acknowledge(count)


def prepare_replay_round_batches(states, req_ids, cad, block_size):
    """Yield retained source rounds in time order, using current live-request KV.

    A new arrival need not have rows in an earlier round. Finished/cancelled
    requests are absent; no stale block table is retained or read. Reordering is
    resolved by current request ID, not by historical batch indices. Serial
    recovery preserves source-round boundaries among surviving requests, not a
    bitwise promise for the original batch shape after cancellation/reordering.
    """
    if not states or len(states) != len(req_ids):
        raise ValueError('serial replay states/request mismatch')
    by_round = {}
    latest = []
    for index, state in enumerate(states):
        tags = [chunk[5] for chunk in state.chunks]
        if (not tags or any(type(tag) is not int for tag in tags)
                or any(a >= b for a, b in zip(tags, tags[1:]))):
            raise RuntimeError('serial replay requires increasing recorded round IDs')
        for left, right in zip(state.chunks, state.chunks[1:]):
            if left[3].item() != right[0][0].item():
                raise RuntimeError('shifted token boundary mismatch')
        latest.append(tags[-1])
        for tag, chunk in zip(tags, state.chunks):
            by_round.setdefault(tag, {})[index] = chunk
    if len(set(latest)) != 1:
        raise RuntimeError('serial replay latest round must cover all active requests')
    for tag in sorted(by_round):
        entries = by_round[tag]
        indices = sorted(entries)
        local_states = []
        for index in indices:
            local = EagleReplay(states[index].rebuild_until)
            local.chunks = [entries[index]]
            local_states.append(local)
        ends = torch.tensor([entries[i][4] for i in indices], dtype=cad.seq_lens_cpu.dtype)
        local_cad = replace(cad, num_reqs=len(indices), seq_lens_cpu=ends,
            seq_lens=ends.to(cad.seq_lens.device),
            block_table_tensor=cad.block_table_tensor[indices])
        active_ids = [req_ids[i] for i in indices]
        yield tag, active_ids, prepare_replay_batch(local_states, active_ids, local_cad, block_size)
