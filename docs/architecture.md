# Architecture

BD-HSD is implemented as a vLLM speculative-decoding overlay. The target model remains the single verifier. Candidate generation is split into a retrieval path and a neural path so their budgets and execution costs can be measured independently.

## Request path

For each active request and decode round:

1. The N-gram proposer searches the current target token history and emits up to `K_r` retrieval tokens.
2. A hit is retained as the request proposal. A miss, or a request whose EAGLE state needs catch-up, is selected for the neural draft path.
3. The EAGLE runner executes only the selected requests when selective sub-batching is enabled. The compact batch keeps original request order through explicit ragged token indices and refreshed KV metadata.
4. Proposals are scattered back to the original active batch. The target model verifies the complete proposal batch and rejection sampling commits the accepted prefix plus the correction/bonus token according to the upstream contract.
5. The runner updates target KV, EAGLE state and routing counters. Finished request IDs are removed from all per-request state maps.

## Budget decoupling

`num_speculative_tokens` is the retrieval capacity `K_r`. `hybrid_eagle_num_speculative_tokens` controls the EAGLE fallback length `K_e`. This makes it possible to compare, for example, `K_r=10, K_e=3` with `K_r=10, K_e=10` while keeping the target verification path and prompt cohort fixed.

The four-way ablation in `configs/budget_ablation.json` separates:

- shared budget scaling;
- retrieval/neural budget decoupling;
- bounded state-debt skipping versus shadow synchronization.

## State debt and KV Cache

Skipping a neural draft execution does not erase the work needed to keep its hidden state aligned. The implementation charges debt from the number of target tokens actually committed in the round. A request with debt is sent through EAGLE catch-up before it is allowed to defer another neural step. This turns a heuristic skip into a bounded state machine.

The selective path rebuilds `CommonAttentionMetadata` from the current request rows and physical KV mapping. It does not reuse stale flattened indices after batch compaction. The optional `state_replay.py` module stores accepted target feature rows by request ID and reconstructs an eager catch-up batch; it is disabled by default and has a narrower support envelope than the main route.

## Runtime observability

The overlay exposes process-local routing counters for N-gram hits, EAGLE fallback requests, actual draft execution, skipped execution, selective sub-batch shape and compact-plan cache behavior. The benchmark harness adds speculative decoding counters, token hashes, stage traces and peak memory snapshots so candidate quality and systems cost can be inspected together.

