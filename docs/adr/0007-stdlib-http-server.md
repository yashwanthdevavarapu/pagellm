# 0007. Standard library HTTP server with strict request validation

- Status: Accepted
- Date: 2026-11-08

## Context

The server must speak the OpenAI `/v1/completions` format, including Server-Sent Events streaming, so that existing clients work unchanged.
A web framework would add dependencies and hide the threading model that ADR 0004 relies on.
Every value a client sends eventually reaches the single engine thread.

## Decision

Use `http.server.ThreadingHTTPServer` with a listen backlog of 128.
All request parsing lives in `serving/protocol.py`, which has no sockets or threads and is unit tested on its own.
Types are checked strictly:

- booleans are not accepted where integers are expected;
- numbers must be finite;
- `seed` must be a non-negative integer;
- `n` is capped at 16;
- `stop` accepts up to 4 non-empty strings;
- bodies over 1 MiB are rejected with 413.

Operational endpoints are `/health` (engine liveness and queue depth) and `/metrics` (Prometheus text format, no client library).

## Consequences

- Zero extra dependencies, and the whole request path is readable in three small files.
- An earlier version accepted `{"seed": "abc"}`. That value raised inside the engine thread and stopped all serving. Strict validation plus a regression test closes that class of bug.
- There is no TLS, authentication or rate limiting. The server is meant to run behind a reverse proxy, as the README states.
