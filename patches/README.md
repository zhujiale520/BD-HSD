# vLLM overlay

`vllm-bd-hsd.patch` is the reviewable runtime patch. It is generated against
the base revision recorded in [`BASE_COMMIT.txt`](BASE_COMMIT.txt) and contains
only the BD-HSD runtime changes and their focused tests.

Apply it to a clean checkout of the same vLLM revision:

```bash
git apply --check /path/to/vllm-bd-hsd.patch
git apply /path/to/vllm-bd-hsd.patch
pip install -e .
```

The patch changes vLLM's speculative configuration, N-gram proposer, EAGLE
runner, optional engine trace, GPU worker controls and correctness-facing
runtime logic. It does not include model weights, virtual environments,
generated logs, request-regrouping experiments or the rest of the upstream
vLLM tree.

The focused tests can be run directly from this repository with `unittest`;
this avoids loading the full upstream test configuration:

```bash
PYTHONPATH=/path/to/patched-vllm \
  python tests/v1/spec_decode/test_hybrid_proposer.py
PYTHONPATH=/path/to/patched-vllm \
  python tests/v1/spec_decode/test_ngram_sparse_batch.py
```
