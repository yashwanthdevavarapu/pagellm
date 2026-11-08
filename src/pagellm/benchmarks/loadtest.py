"""HTTP load test: concurrent streaming clients against a live server.

    pagellm loadtest --url http://127.0.0.1:8000 --requests 64 --concurrency 16

Each client sends ``stream: true`` completions and timestamps every SSE chunk, so
the report covers what a user actually experiences: time to first token (TTFT),
inter-token latency (ITL), end-to-end latency, and aggregate throughput. Works
against any server that speaks the OpenAI completions SSE format.
"""

from __future__ import annotations

import http.client
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import numpy as np

from ..model.weights import DATA_PATH


@dataclass
class RequestResult:
    ok: bool
    sent: float
    first: float | None = None
    end: float | None = None
    chunk_times: list[float] = field(default_factory=list)
    tokens: int = 0
    error: str | None = None


def make_workload(n: int, seed: int = 0, prompt_len=(32, 128), out_len=(32, 96)) -> list[dict]:
    rng = np.random.default_rng(seed)
    text = DATA_PATH.read_text(encoding="utf-8")
    bodies = []
    for i in range(n):
        L = int(rng.integers(*prompt_len))
        s = int(rng.integers(0, len(text) - L))
        bodies.append(
            {
                "prompt": text[s : s + L],
                "max_tokens": int(rng.integers(*out_len)),
                "temperature": 0.8,
                "seed": i,
                "stream": True,
            }
        )
    return bodies


def _one(host: str, port: int, body: dict, timeout: float) -> RequestResult:
    res = RequestResult(ok=False, sent=time.perf_counter())
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(
            "POST",
            "/v1/completions",
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        if resp.status != 200:
            res.error = f"HTTP {resp.status}"
            return res
        while line := resp.readline():
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                res.ok = True
                break
            now = time.perf_counter()
            if json.loads(data)["choices"][0]["text"]:
                res.chunk_times.append(now)
                res.tokens += 1
        res.end = time.perf_counter()
        res.first = res.chunk_times[0] if res.chunk_times else None
    except OSError as e:
        res.error = repr(e)
    finally:
        conn.close()
    return res


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def run_load(
    url: str,
    n_requests: int = 64,
    concurrency: int = 16,
    seed: int = 0,
    timeout: float = 300.0,
    workload: list[dict] | None = None,
) -> dict[str, Any]:
    u = urlparse(url)
    host, port = u.hostname or "127.0.0.1", u.port or 80
    bodies = workload or make_workload(n_requests, seed)
    todo: queue.Queue = queue.Queue()
    for b in bodies:
        todo.put(b)
    results: list[RequestResult] = []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                body = todo.get_nowait()
            except queue.Empty:
                return
            r = _one(host, port, body, timeout)
            with lock:
                results.append(r)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r.ok]
    ttft = [(r.first - r.sent) * 1e3 for r in ok if r.first is not None]
    e2e = [(r.end - r.sent) * 1e3 for r in ok if r.end is not None]
    itl = [float(g) * 1e3 for r in ok for g in np.diff(r.chunk_times)]
    tokens = sum(r.tokens for r in ok)
    return {
        "requests": len(bodies),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        "concurrency": concurrency,
        "wall_s": wall,
        "requests_per_s": len(ok) / wall,
        "output_tok_per_s": tokens / wall,
        "ttft_ms": {"p50": _pct(ttft, 50), "p95": _pct(ttft, 95), "p99": _pct(ttft, 99)},
        "itl_ms": {
            "p50": _pct(itl, 50),
            "p95": _pct(itl, 95),
            "p99": _pct(itl, 99),
            "max": max(itl) if itl else float("nan"),
        },
        "e2e_ms": {"p50": _pct(e2e, 50), "p95": _pct(e2e, 95), "p99": _pct(e2e, 99)},
    }


def print_report(r: dict, title: str = "HTTP load test") -> None:
    print(f"\n## {title} ({r['requests']} streaming requests, {r['concurrency']} concurrent)\n")
    print(
        f"succeeded {r['succeeded']}, failed {r['failed']}, wall {r['wall_s']:.2f} s, "
        f"{r['requests_per_s']:.2f} req/s, {r['output_tok_per_s']:.0f} output tok/s\n"
    )
    print("| metric (ms) | p50 | p95 | p99 |")
    print("|---|---|---|---|")
    for key, name in (
        ("ttft_ms", "time to first token"),
        ("itl_ms", "inter-token latency"),
        ("e2e_ms", "end-to-end latency"),
    ):
        m = r[key]
        print(f"| {name} | {m['p50']:.1f} | {m['p95']:.1f} | {m['p99']:.1f} |")
