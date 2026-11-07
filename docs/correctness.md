# Correctness and state invariants

BD-HSD treats correctness as a gate for performance analysis. A faster run with a changed output is kept as a diagnostic record and is excluded from strict latency comparisons.

## Candidate and verification invariants

- N-gram and EAGLE only propose token IDs; the target model performs the normal verification step.
- A retrieval hit is preferred for that request. An empty retrieval proposal falls back to the EAGLE row for the same request.
- The merge operation preserves active-batch order and proposal lengths.
- A selective EAGLE sub-batch is represented by an explicit request-index list and ragged token indices. Scatter restores the original request positions before verification.

## State invariants

- Only target-committed rows can enter persistent EAGLE state.
- Rejected future tokens never enter the retained state frontier.
- Debt is charged from actual committed target tokens, not from the requested speculation length.
- A debt request is eligible for another skip only after the configured bound permits it; otherwise it must catch up.
- Request completion and preemption remove stale state. KV slot mappings are rebuilt from current scheduler metadata after compaction or recovery.

## Verification layers

The focused CPU tests cover proposal merge behavior, all-hit and mixed-hit routing, sparse request-index writes, compact metadata construction, cache-plan reuse, stable N-gram retention and state-debt accounting. GPU runs add:

1. a warm-up outside measured intervals;
2. fixed prompt and sampling metadata;
3. output token IDs and hashes;
4. exact reference-prefix comparison with explicit EOS/length handling;
5. acceptance, draft and routing counters;
6. mismatch IDs retained in the result record.

This separation prevents acceptance rate, prompt length or a raw wall-time reduction from being mistaken for a correctness proof.

