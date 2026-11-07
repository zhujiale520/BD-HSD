# Benchmark outputs

The repository now includes a curated, public result snapshot:

- [BD-HSD results](BD-HSD-results.md): readable tables, figures and experiment scope.
- [summary.json](summary.json): machine-readable primary, A100, ablation and cost-attribution results.
- [figures/](figures/): selected plots from the verified analyses.

Large benchmark outputs are still generated outside the source tree. Keep a
run directory local and pass its JSON/JSONL files to the analysis scripts:

```bash
python scripts/analyze_budget_ablation.py \
  --input results/runs/budget_ablation.json \
  --output results/summary.json
```

The run record should contain the model identities, prompt hashes, policy
configuration, per-request latency, generated token IDs or hashes, strict
reference-gate IDs, speculative-decoding counters, routing counters and peak
GPU memory. Do not commit checkpoints, prompts containing private data, or raw
machine logs. The checked-in snapshot contains rounded aggregates and does not
claim to replace the raw audit trail.
