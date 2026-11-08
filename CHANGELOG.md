# Changelog

All notable changes are listed here. The format follows Keep a Changelog, and the project uses semantic versioning.

## [1.0.0] - 2026-11-08

### Added

- Chunked prefill with decode-first scheduling: `pagellm serve --chunked-prefill --max-num-batched-tokens N` (ADR 0010).
- `pagellm loadtest`: concurrent streaming HTTP clients reporting TTFT, inter-token latency and end-to-end percentiles.
- `pagellm bench chunked` and `pagellm bench http` (real server subprocess, chunked prefill off and on).
- BLAS pinned to one thread on import, overridable with `PAGELLM_BLAS_THREADS` (ADR 0011).
- Package split into `model`, `cache`, `scheduling`, `engine`, `serving`, `training` and `benchmarks` subpackages with a stable top-level API.
- Batched decode attention: all single-token decode rows in a step are attended in one padded, masked matmul (ADR 0009).
- `serving/protocol.py`: strict, unit-tested validation of every request field.
- `GET /metrics` in Prometheus text format, and queue depth plus engine liveness in `GET /health`.
- Engine fault isolation: an exception in a step fails only the in-flight requests with a 500, reclaims their KV blocks, and the server keeps serving.
- `pagellm bench scaling` (decode cost per token vs batch size) and `--quick` mode; results record their environment.
- `pagellm --version`, `serve --model-name`, `serve --log-level`, and clean shutdown on SIGTERM.
- Test tiers: unit, integration and end-to-end.
- CI on Linux, macOS and Windows with Python 3.11 to 3.13; benchmark, release and CodeQL workflows; Dependabot.
- Dockerfile with a non-root user and a health check.
- Eleven architecture decision records, chart and screenshot generators, Makefile, pre-commit configuration.

### Fixed

- OpenBLAS thread start-up made mid-size matmuls up to 9.5x slower and wall-clock timings unstable.
- A non-integer `seed` raised inside the engine thread, which stopped all serving. Requests are now validated before they reach the engine.
- Attention scores were silently promoted from float32 to float64 by a numpy scalar, which pushed the following matmul off BLAS (43x slower on that operation).
- Engine statistics kept a per-step list that grew without bound in a long-running server; replaced with constant-size counters.
- `usage.prompt_tokens` now counts tokenizer tokens instead of characters.
- Aborted requests release their engine bookkeeping.
