# Experiment protocol

The benchmark harness is designed around paired, same-engine measurements. It loads the target and EAGLE-3 models once, warms up the requested shape, then switches only the BD-HSD runtime policy between requests or paired cohorts.

## Fixed factors

- target model and EAGLE checkpoint;
- tokenizer and prompt text;
- greedy sampling settings;
- maximum output length and model length;
- GPU visibility, dtype and eager mode;
- retrieval window and neural draft length;
- prompt order and randomization seed;
- correctness reference and token comparison rule.

## Main measurements

Each result record contains request latency, output length, token hash, finish reason, target-gated reference status, speculative-decoding counters, hybrid routing counters and peak allocated/reserved GPU memory. Optional traces split the runner into target verification, rejection sampling, route control, EAGLE first pass, draft decode and metadata preparation.

The primary comparisons are:

| Question | Comparison |
| --- | --- |
| Does a larger shared capacity help? | `K3/E3` vs `K10/E10` |
| Does budget decoupling reduce neural work? | `K10/E10` vs `K10/E3` |
| Does bounded debt reduce redundant draft execution? | `K10/E3/shadow` vs `K10/E3/debt=1` |
| What is the complete BD-HSD path? | `K3/E3` vs `K10/E3/debt=1` |

Latency summaries are computed only over the intersection of reference-gated prompt IDs for all policies in a pair. The analysis keeps the full mismatch set and reports memory/routing counters alongside the paired effect.

## Reproduction boundaries

The repository keeps scripts and contracts, while large model weights, datasets and raw run directories stay external. A result should be published with its source revision, model identifiers, prompt hashes, policy JSON, environment summary and analysis hash. The project reports measured operating points and their support envelope; it does not assume that a positive result on one workload generalizes to every batch shape or serving load.

