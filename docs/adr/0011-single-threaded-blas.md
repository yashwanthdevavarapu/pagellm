# 0011. Single-threaded BLAS by default

- Status: Accepted
- Date: 2026-11-08

## Context

Profiling a chunked-prefill step showed that most of its time went to plain matmuls in the transformer, about 10 ms per step for under 100 tokens of a 128-wide model. That work should take microseconds.
A microbenchmark isolated the cause. numpy's bundled OpenBLAS starts a thread pool and splits each matmul across it. For this model's small matrices, waking and synchronising those threads costs far more than the arithmetic, and on a machine where other applications are busy the threads also wait to be scheduled.

Measured on the development laptop (8 logical CPUs), for `(n, 128) @ (128, 352)` in float32:

| rows | default thread pool | 1 thread |
|---|---|---|
| 16 | 23 us | 17 us |
| 80 | 1944 us | 205 us |
| 256 | 459 us | 477 us |

## Decision

`import pagellm` sets `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 with `os.environ.setdefault`, before numpy is imported.

- An explicit user setting always wins, because `setdefault` never overwrites.
- `PAGELLM_BLAS_THREADS` changes the default without touching the BLAS-specific variables.
- If numpy is imported before pagellm, the setting has no effect. The README documents this.

## Consequences

- Step time becomes much lower and far more stable. Before this change, wall-clock benchmark results drifted by about 2x between runs.
- Continuous batching and chunked prefill show their wall-time benefits instead of having them hidden by thread start-up cost.
- A much larger model, where each matmul is big enough to benefit from parallelism, should raise `PAGELLM_BLAS_THREADS`.
- A test checks that the default is applied and that a user's value is respected.
