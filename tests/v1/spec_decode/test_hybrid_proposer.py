# SPDX-License-Identifier: Apache-2.0

import unittest
from collections import OrderedDict

import torch

from vllm.v1.spec_decode.hybrid_proposer import (
    compact_common_attention_metadata,
    gather_ragged_token_indices,
    get_hybrid_routing_stats,
    merge_ngram_and_compact_draft_proposals,
    merge_ngram_and_draft_proposals,
    record_draft_model_execution,
    record_ngram_batch_availability,
    record_skipped_state_debt,
    record_selective_draft_batch,
    select_draft_request_indices,
    select_ngram_without_draft,
    select_retrieval_only,
)


class TestMergeNgramAndDraftProposals(unittest.TestCase):
    def test_state_debt_uses_actual_committed_tokens_and_prunes(self) -> None:
        self.assertEqual(
            record_skipped_state_debt(
                {"hit": 2, "synced": 4, "finished": 9},
                {"hit"},
                {"hit": 5, "synced": 1, "new": 3},
            ),
            {"hit": 7, "synced": 4, "new": 0},
        )

    def test_prefers_ngram_and_falls_back_to_draft(self) -> None:
        before = get_hybrid_routing_stats()
        ngram = [[1, 2, 3], [], [7]]
        draft = torch.tensor([[4, 5, 6], [8, 9, 10], [11, 12, 13]])

        self.assertEqual(
            merge_ngram_and_draft_proposals(ngram, draft),
            [[1, 2, 3], [8, 9, 10], [7]],
        )
        after = get_hybrid_routing_stats()
        self.assertEqual(after["ngram_requests"] - before["ngram_requests"], 2)
        self.assertEqual(
            after["draft_model_requests"] - before["draft_model_requests"], 1
        )
        self.assertEqual(
            after["ngram_proposed_tokens"] - before["ngram_proposed_tokens"], 4
        )
        self.assertEqual(
            after["draft_model_proposed_tokens"]
            - before["draft_model_proposed_tokens"],
            3,
        )

    def test_rejects_different_batch_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same size"):
            merge_ngram_and_draft_proposals([[1]], [[2], [3]])

    def test_records_execution_and_all_ngram_skip(self) -> None:
        before = get_hybrid_routing_stats()
        record_draft_model_execution(2)
        self.assertEqual(select_ngram_without_draft([[1, 2], [3]]), [[1, 2], [3]])
        after = get_hybrid_routing_stats()
        self.assertEqual(
            after["draft_model_executed_requests"]
            - before["draft_model_executed_requests"],
            2,
        )
        self.assertEqual(
            after["draft_model_skipped_requests"]
            - before["draft_model_skipped_requests"],
            2,
        )

    def test_rejects_skip_with_ngram_miss(self) -> None:
        with self.assertRaisesRegex(ValueError, "every request"):
            select_ngram_without_draft([[1], []])

    def test_retrieval_only_preserves_misses_without_fallback(self) -> None:
        before = get_hybrid_routing_stats()
        self.assertEqual(select_retrieval_only([[1, 2], [], [3]]), [[1, 2], [], [3]])
        after = get_hybrid_routing_stats()
        self.assertEqual(after["ngram_requests"] - before["ngram_requests"], 2)
        self.assertEqual(
            after["draft_model_skipped_requests"]
            - before["draft_model_skipped_requests"],
            3,
        )

    def test_merges_eagle_tensor_without_changing_shape_contract(self) -> None:
        ngram = [[1, 2, 3], []]
        eagle = torch.tensor([[4, 5, 6], [7, 8, 9]])
        self.assertEqual(
            merge_ngram_and_draft_proposals(ngram, eagle),
            [[1, 2, 3], [7, 8, 9]],
        )

    def test_records_batch_availability_categories(self) -> None:
        before = get_hybrid_routing_stats()
        record_ngram_batch_availability([[1], [2]])
        record_ngram_batch_availability([[1], []])
        record_ngram_batch_availability([[], []])
        after = get_hybrid_routing_stats()
        self.assertEqual(after["routing_rounds"] - before["routing_rounds"], 3)
        self.assertEqual(
            after["active_request_rounds"] - before["active_request_rounds"], 6
        )
        self.assertEqual(
            after["all_ngram_hit_rounds"] - before["all_ngram_hit_rounds"], 1
        )
        self.assertEqual(
            after["partial_ngram_hit_rounds"]
            - before["partial_ngram_hit_rounds"],
            1,
        )
        self.assertEqual(
            after["zero_ngram_hit_rounds"] - before["zero_ngram_hit_rounds"], 1
        )

    def test_selects_misses_and_debt_requests_for_compact_draft(self) -> None:
        self.assertEqual(
            select_draft_request_indices(
                [[1, 2, 3], [], [4, 5, 6], []],
                [False, False, True, True],
            ),
            [1, 2, 3],
        )
        with self.assertRaisesRegex(ValueError, "same size"):
            select_draft_request_indices([[1]], [False, True])

    def test_gathers_ragged_token_ranges_in_subbatch_order(self) -> None:
        self.assertEqual(
            gather_ragged_token_indices([0, 2, 5, 6, 10], [3, 1]),
            [6, 7, 8, 9, 2, 3, 4],
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            gather_ragged_token_indices([0, 1, 2], [1, 1])

    def test_scatters_compact_draft_and_keeps_ngram_on_debt_hit(self) -> None:
        # Request 2 enters the compact batch only to catch up its draft state;
        # its N-gram proposal still wins when proposals are merged.
        self.assertEqual(
            merge_ngram_and_compact_draft_proposals(
                [[10, 11], [], [20, 21], []],
                [1, 2, 3],
                torch.tensor([[30, 31], [40, 41], [50, 51]]),
            ),
            [[10, 11], [30, 31], [20, 21], [50, 51]],
        )
        with self.assertRaisesRegex(ValueError, "neither"):
            merge_ngram_and_compact_draft_proposals([[1], []], [], [])

    def test_compacts_common_attention_metadata(self) -> None:
        from vllm.v1.attention.backends.utils import CommonAttentionMetadata

        metadata = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 2, 5, 6], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 2, 5, 6], dtype=torch.int32),
            seq_lens=torch.tensor([12, 23, 31], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([12, 23, 31], dtype=torch.int32),
            num_computed_tokens_cpu=torch.tensor([10, 20, 30], dtype=torch.int32),
            num_reqs=3,
            num_actual_tokens=6,
            max_query_len=3,
            max_seq_len=31,
            block_table_tensor=torch.tensor([[1, 2], [3, 4], [5, 6]]),
            slot_mapping=torch.tensor([10, 11, 20, 21, 22, 30]),
        )
        compact, token_indices, last_indices = compact_common_attention_metadata(
            metadata, [2, 0]
        )
        self.assertEqual(token_indices.tolist(), [5, 0, 1])
        self.assertEqual(last_indices.tolist(), [0, 2])
        self.assertEqual(compact.query_start_loc_cpu.tolist(), [0, 1, 3])
        self.assertEqual(compact.seq_lens_cpu.tolist(), [31, 12])
        self.assertEqual(compact.block_table_tensor.tolist(), [[5, 6], [1, 2]])
        self.assertEqual(compact.slot_mapping.tolist(), [30, 10, 11])

    def test_reuses_compact_index_plan_but_refreshes_dynamic_metadata(self) -> None:
        from vllm.v1.attention.backends.utils import CommonAttentionMetadata

        def make_metadata(seq_offset: int) -> CommonAttentionMetadata:
            return CommonAttentionMetadata(
                query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
                query_start_loc_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
                seq_lens=torch.tensor(
                    [12 + seq_offset, 23 + seq_offset], dtype=torch.int32
                ),
                seq_lens_cpu=torch.tensor(
                    [12 + seq_offset, 23 + seq_offset], dtype=torch.int32
                ),
                num_computed_tokens_cpu=torch.tensor(
                    [10 + seq_offset, 20 + seq_offset], dtype=torch.int32
                ),
                num_reqs=2,
                num_actual_tokens=5,
                max_query_len=3,
                max_seq_len=23 + seq_offset,
                block_table_tensor=torch.tensor(
                    [[1 + seq_offset, 2], [3 + seq_offset, 4]]
                ),
                slot_mapping=torch.tensor(
                    [10 + seq_offset, 11, 20 + seq_offset, 21, 22]
                ),
            )

        cache = OrderedDict()
        before = get_hybrid_routing_stats()
        first, first_tokens, _ = compact_common_attention_metadata(
            make_metadata(0), [1], cache
        )
        second, second_tokens, _ = compact_common_attention_metadata(
            make_metadata(5), [1], cache
        )
        after = get_hybrid_routing_stats()

        self.assertIs(first_tokens, second_tokens)
        self.assertEqual(len(cache), 1)
        self.assertEqual(
            after["compact_plan_cache_misses"]
            - before["compact_plan_cache_misses"],
            1,
        )
        self.assertEqual(
            after["compact_plan_cache_hits"] - before["compact_plan_cache_hits"],
            1,
        )
        self.assertEqual(first.seq_lens_cpu.tolist(), [23])
        self.assertEqual(second.seq_lens_cpu.tolist(), [28])
        self.assertEqual(first.block_table_tensor.tolist(), [[3, 4]])
        self.assertEqual(second.block_table_tensor.tolist(), [[8, 4]])
        self.assertEqual(first.slot_mapping.tolist(), [20, 21, 22])
        self.assertEqual(second.slot_mapping.tolist(), [25, 21, 22])

    def test_records_selective_subbatch_savings(self) -> None:
        before = get_hybrid_routing_stats()
        record_selective_draft_batch(4, 2)
        record_selective_draft_batch(4, 4)
        after = get_hybrid_routing_stats()
        self.assertEqual(
            after["selective_draft_subbatch_rounds"]
            - before["selective_draft_subbatch_rounds"],
            1,
        )
        self.assertEqual(
            after["selective_draft_full_batch_rounds"]
            - before["selective_draft_full_batch_rounds"],
            1,
        )
        self.assertEqual(
            after["selective_draft_requests_avoided"]
            - before["selective_draft_requests_avoided"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
