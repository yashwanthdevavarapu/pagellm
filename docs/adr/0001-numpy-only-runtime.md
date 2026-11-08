# 0001. Pure numpy runtime, no deep learning framework

- Status: Accepted
- Date: 2026-11-08

## Context

The project exists to make the systems side of LLM serving readable: cache layout, batching, scheduling and speculation.
In production servers those ideas sit behind CUDA kernels, autograd frameworks and tens of thousands of lines of glue.
A reader should be able to follow any token from the HTTP request to the sampled output without leaving the repository.

## Decision

The model, the KV caches, attention, sampling, training and the server are written against numpy and the Python standard library only.
The single runtime dependency is `numpy>=1.24`.
Training uses a hand-derived backward pass that is checked against finite differences.

## Consequences

- Installation is one `pip install` on Linux, macOS or Windows, Python 3.11 and newer, with no GPU or compiler.
- Every mechanism can be tested against a dense reference implementation with `allclose`.
- Absolute speed is low: a step is dominated by Python and per-call overhead, not FLOPs. Benchmarks therefore report hardware-independent ratios (engine steps, KV slots, tokens per target forward) next to wall time.
- Training is limited to small models (under 1M parameters). That is enough to produce text that makes cache bugs visible.
