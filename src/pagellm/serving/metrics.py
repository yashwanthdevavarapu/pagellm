"""Prometheus text-format metrics for the server (no client library needed)."""

from __future__ import annotations

import threading
from collections import Counter

from ..engine.llm_engine import LLMEngine


class ServerMetrics:
    """Counters written by handler threads; a lock keeps increments exact."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: Counter[tuple[str, int]] = Counter()  # (route, status) -> count
        self.streamed_chunks = 0
        self.aborted = 0

    def record(self, route: str, status: int) -> None:
        with self._lock:
            self.requests[(route, status)] += 1

    def chunk(self) -> None:
        with self._lock:
            self.streamed_chunks += 1

    def abort(self) -> None:
        with self._lock:
            self.aborted += 1


def render(engine: LLMEngine, metrics: ServerMetrics) -> str:
    """Snapshot engine and server state. Reads are lock-free: every value is a
    plain int or a container length, each of which is atomic under the GIL, so
    the worst case is a snapshot that is one step stale."""
    st = engine.stats
    alloc = engine.bm.alloc
    sched = engine.scheduler
    lines: list[str] = []

    def metric(name: str, kind: str, help_: str, samples: list[tuple[str, float]]) -> None:
        lines.append(f"# HELP pagellm_{name} {help_}")
        lines.append(f"# TYPE pagellm_{name} {kind}")
        for labels, value in samples:
            lines.append(f"pagellm_{name}{labels} {value}")

    with metrics._lock:
        reqs = sorted(metrics.requests.items())
        chunks, aborted = metrics.streamed_chunks, metrics.aborted
    metric(
        "http_requests_total",
        "counter",
        "HTTP requests by route and status.",
        [(f'{{route="{r}",status="{s}"}}', n) for (r, s), n in reqs],
    )
    metric("stream_chunks_total", "counter", "SSE chunks written.", [("", chunks)])
    metric(
        "aborted_requests_total",
        "counter",
        "Requests aborted by client disconnect.",
        [("", aborted)],
    )
    metric("engine_steps_total", "counter", "Engine iterations (forward passes).", [("", st.steps)])
    metric("generated_tokens_total", "counter", "Tokens sampled.", [("", st.generated_tokens)])
    metric("prefill_tokens_total", "counter", "Prompt tokens computed.", [("", st.prefill_tokens)])
    metric(
        "prefix_cache_tokens_total",
        "counter",
        "Prompt tokens served from the prefix cache.",
        [("", st.cached_prompt_tokens)],
    )
    metric("cow_copies_total", "counter", "Copy-on-write block copies.", [("", st.cow_copies)])
    metric(
        "preemptions_total",
        "counter",
        "Sequences preempted for recompute.",
        [("", sched.num_preemptions)],
    )
    metric(
        "running_sequences", "gauge", "Sequences in the running batch.", [("", len(sched.running))]
    )
    metric(
        "waiting_sequences", "gauge", "Sequences queued for admission.", [("", len(sched.waiting))]
    )
    metric("kv_blocks_used", "gauge", "KV cache blocks in use.", [("", alloc.num_used)])
    metric("kv_blocks_total", "gauge", "KV cache blocks in the pool.", [("", alloc.num_blocks)])
    return "\n".join(lines) + "\n"
