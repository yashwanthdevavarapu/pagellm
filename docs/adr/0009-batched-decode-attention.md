# 0009. Batched decode attention and float32 discipline

- Status: Accepted
- Date: 2026-11-08

## Context

Continuous batching only pays off in wall time if the cost per token falls as the batch grows.
Profiling a 16-sequence decode step showed two problems.

1. Attention ran as a Python loop: one `causal_attention` call per sequence per layer. That loop was about 60% of step time, so per-call overhead grew linearly with batch size.
2. The attention scale was written `x * (1.0 / np.sqrt(d))`. `np.sqrt` returns a numpy float64 scalar, and under NumPy 2 promotion rules that silently turns the float32 scores into float64. The following `probs @ v` then multiplies float64 by float32, which numpy cannot hand to BLAS. Measured on a 32-sequence attention: 1822 us versus 42 us for float32 by float32, a 43x slowdown on that matmul.

## Decision

- Decode rows (one new token per sequence) of all sequences in a step are attended in one call, `batched_decode_attention`. Each sequence's context is gathered through its block table and padded to the longest context. Because a decode query sits at the last position of its own context, the causal mask reduces to a padding mask. Prefill rows keep the per-sequence path.
- Index math (slot mappings, padding masks) is computed once per step, not once per layer.
- Scalars that touch float32 arrays are Python floats (`math.sqrt`), never numpy float64 scalars.
- `EngineConfig.batch_decode` can turn batching off, so the benchmark and the tests can compare both paths.

## Consequences

- Both paths are tested to produce the same logits (`allclose`, including a mixed prefill and decode step), and garbage in the padding is tested not to leak into results.
- `pagellm bench scaling` measures cost per token against batch size with the batched path on and off.
- Padding wastes compute when context lengths in a batch differ widely. Bucketing by length would reduce it; at this model size the per-call saving dominates.
