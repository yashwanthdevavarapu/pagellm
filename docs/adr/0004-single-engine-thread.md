# 0004. One engine thread behind a queue

- Status: Accepted
- Date: 2026-11-08

## Context

The HTTP server handles each connection on its own thread.
Continuous batching only works if one loop sees every in-flight request, and the scheduler, block manager and KV pool are shared mutable state.

## Decision

Exactly one thread (`EngineLoop`) ever calls into the engine.
Handler threads validate the request, put a submission on a queue, and block on a per-request queue of deltas.
The engine thread drains submissions, runs `engine.step()` (which advances every in-flight request at once) and fans the deltas out.
Client disconnects and engine faults travel through the same queues as messages (`abort`, `Failure`).

## Consequences

- No locks around engine state. Correctness reasoning stays single-threaded.
- The loop is a single point of failure, so it is defended in depth:
  - requests are validated before submission (ADR 0007);
  - an `add_request` error becomes a 400 or 500 for that client only;
  - an exception inside `step()` fails the in-flight requests with a 500, frees their blocks, and the loop keeps serving. An integration test injects a fault to prove it.
- `/health` returns 503 if the engine thread is ever not alive, so an orchestrator can restart the process.
- Prefill and decode share the GIL with HTTP handling. That is acceptable at this scale and is listed as a limitation.
