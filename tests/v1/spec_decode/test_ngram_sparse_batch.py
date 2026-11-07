# SPDX-License-Identifier: Apache-2.0

import io
import json
import unittest

import numpy as np

from vllm.v1.spec_decode.ngram_proposer import (
    NgramProposer,
    batch_propose_numba,
    retain_natural_ngram_hits,
)


class TestNgramSparseBatch(unittest.TestCase):
    def test_retention_q_only_suppresses_natural_hits(self) -> None:
        req_ids = ["request-0", "request-1"]
        positions = [10, 11]
        proposals = [[7, 8], []]
        self.assertEqual(
            retain_natural_ngram_hits(req_ids, positions, proposals, 0.0),
            [[], []],
        )
        self.assertEqual(
            retain_natural_ngram_hits(req_ids, positions, proposals, 1.0),
            proposals,
        )
        # Stable across calls; the exact outcome is hash-defined, not a
        # process-randomized Python hash.
        self.assertEqual(
            retain_natural_ngram_hits(req_ids, positions, proposals, 0.5),
            retain_natural_ngram_hits(req_ids, positions, proposals, 0.5),
        )

    def test_writes_proposal_to_original_request_index(self) -> None:
        token_ids = np.zeros((2, 16), dtype=np.int32)
        token_ids[1, :5] = [1, 2, 3, 1, 2]
        num_tokens = np.asarray([0, 5], dtype=np.int32)
        drafts = np.zeros((2, 1), dtype=np.int32)
        num_drafts = np.zeros(2, dtype=np.int32)

        batch_propose_numba(
            [1],
            num_tokens,
            token_ids,
            2,
            2,
            16,
            1,
            drafts,
            num_drafts,
        )

        self.assertEqual(num_drafts.tolist(), [0, 1])
        self.assertEqual(drafts[1, 0], 3)

    def test_opt_in_trace_records_lookup_contract(self) -> None:
        proposer = NgramProposer.__new__(NgramProposer)
        proposer.max_model_len = 16
        proposer._trace_handle = io.StringIO()
        proposer._trace_call_index = 0
        proposer.backoff_miss_threshold = 0
        proposer.backoff_probe_interval = 1
        proposer.enabled = True
        proposer._miss_counts = {}
        proposer._next_probe_positions = {}
        proposer.batch_propose = lambda *args: [[7, 8], []]

        result = proposer.propose(
            [[4], [5]],
            ["request-0", "request-1"],
            np.asarray([5, 6], dtype=np.int32),
            np.zeros((2, 16), dtype=np.int32),
            set(),
        )

        self.assertEqual(result, [[7, 8], []])
        trace = json.loads(proposer._trace_handle.getvalue())
        self.assertEqual(trace["call_index"], 0)
        self.assertEqual(trace["request_ids"], ["request-0", "request-1"])
        self.assertEqual(trace["eligible_request_indices"], [0, 1])
        self.assertEqual(trace["proposal_lengths"], [2, 0])
        self.assertEqual(trace["proposal_request_count"], 1)
        self.assertGreaterEqual(trace["elapsed_s"], 0.0)

    def test_retention_aligns_to_sampled_rows_when_batch_has_bookkeeping_rows(self) -> None:
        proposer = NgramProposer.__new__(NgramProposer)
        proposer.max_model_len = 16
        proposer.retention_q = 0.0
        proposer._trace_handle = None
        proposer._trace_call_index = 0
        proposer.backoff_miss_threshold = 0
        proposer.backoff_probe_interval = 1
        proposer.enabled = True
        proposer._miss_counts = {}
        proposer._next_probe_positions = {}
        proposer.batch_propose = lambda *args: [[7, 8]]
        result = proposer.propose(
            [[4]],
            ["request-0", "bookkeeping-1"],
            np.asarray([5, 6], dtype=np.int32),
            np.zeros((2, 16), dtype=np.int32),
            set(),
        )
        assert result == [[]]

    def test_backoff_skips_then_probes_again(self) -> None:
        proposer = NgramProposer.__new__(NgramProposer)
        proposer.max_model_len = 32
        proposer._trace_handle = None
        proposer._trace_call_index = 0
        proposer.backoff_miss_threshold = 2
        proposer.backoff_probe_interval = 2
        proposer.enabled = True
        proposer._miss_counts = {}
        proposer._next_probe_positions = {}
        probed_indices = []

        def batch_propose(_count, valid, *_args):
            probed_indices.append(list(valid))
            return [[]]

        proposer.batch_propose = batch_propose
        token_ids = np.zeros((1, 32), dtype=np.int32)
        for position in (10, 11, 12, 13):
            proposer.propose(
                [[4]],
                ["request"],
                np.asarray([position], dtype=np.int32),
                token_ids,
                set(),
            )

        self.assertEqual(probed_indices, [[0], [0], [], [0]])

    def test_runtime_disable_returns_empty_without_lookup(self) -> None:
        proposer = NgramProposer.__new__(NgramProposer)
        proposer.max_model_len = 32
        proposer._trace_handle = None
        proposer._trace_call_index = 0
        proposer.backoff_miss_threshold = 0
        proposer.backoff_probe_interval = 1
        proposer.enabled = False
        proposer._miss_counts = {}
        proposer._next_probe_positions = {}
        probed_indices = []

        def batch_propose(_count, valid, *_args):
            probed_indices.append(list(valid))
            return [[]]

        proposer.batch_propose = batch_propose
        result = proposer.propose(
            [[4]],
            ["request"],
            np.asarray([10], dtype=np.int32),
            np.zeros((1, 32), dtype=np.int32),
            set(),
        )

        self.assertEqual(result, [[]])
        self.assertEqual(probed_indices, [[]])


if __name__ == "__main__":
    unittest.main()
