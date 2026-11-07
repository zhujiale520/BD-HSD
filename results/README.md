# Benchmark outputs

Benchmark outputs are intentionally generated outside the source tree. Keep a
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
machine logs.

