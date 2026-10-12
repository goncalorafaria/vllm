# Compact logprob extraction

This experimental fork adds opt-in logprob extraction to the vLLM 0.29.0 V2 GPU
sampler. Token sampling is unchanged. FlashInfer's deterministic radix lookup
finds the highest-valued entries to record; Triton computes normalization over
the full processed distribution and the sampled token's rank.

## Enable

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_COMPACT_LOGPROBS=1
```

Use `logprobs_mode="processed_logprobs"`. For normal sampling with top-128 logprob
recording, use `SamplingParams(top_k=-1, top_p=1.0, logprobs=128)` with your chosen
temperature. The no-extraction baseline is `logprobs=None`; `logprobs=0` still
returns the sampled token's logprob. The output retains the sampled token even
when it is outside the recorded head.

The optimized path supports CUDA FP32/FP16/BF16 logits with a contiguous vocabulary
axis and recording widths 0–128. Custom token lists, logits output and larger
heads use native extraction. Small vocabularies and strided rows retain PyTorch's
lookup with parallel normalization. Equal-valued entries can have a different
deterministic order or boundary subset from native extraction.

## Reproduce

With the fork installed in a CUDA environment:

```bash
VLLM_USE_COMPACT_LOGPROBS=1 uv run --python .venv/bin/python \
  benchmarks/kernels/benchmark_compact_logprobs.py results.jsonl
uv run --python .venv/bin/python -m pytest \
  tests/v1/sample/test_logprobs.py -k compact
```

The benchmark compares full sampler calls with no logprobs, native top-128 and
compact top-128. It uses vocabulary size 151936, temperature 0.8, no top-k/top-p
sampling truncation, and batches 1, 8, 64 and 256. It checks outputs, warms up each
path, rotates measurement order, and records wall and CUDA event timings.

The saved results in `benchmarks/kernels/compact_logprobs_results.json` came from
Jupiter H100 runs of these kernels through the Prime RL runtime adapter. They
exclude model execution, host transfer and serialization. The direct fork
integration has not yet been GPU-tested; these measurements do not establish
whole-model throughput gains.
