"""Benchmarks. Every number they print is measured on the machine they run on.

    pagellm bench batching   static vs continuous batching on an arrival trace
    pagellm bench memory     KV waste: contiguous reservation vs paged blocks
    pagellm bench prefix     prefix caching on requests sharing a system prompt
    pagellm bench spec       speculative decoding acceptance and speedup
    pagellm bench scaling    decode cost per token vs batch size

Timing method for the batching benchmark: arrivals follow a seeded Poisson
trace in *simulated* time, and the simulated clock advances by the *measured*
wall time of each engine step (and jumps forward when the engine is idle).
This keeps the trace reproducible while every unit of compute in the result is
real - it is the same trick as replaying a production trace against a server.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..cache.contiguous import KVCache, forward_with_caches
from ..engine.llm_engine import EngineConfig, LLMEngine
from ..engine.sampling import SamplingParams
from ..engine.speculative import SpeculativeDecoder, autoregressive
from ..model.weights import DATA_PATH, load_pretrained


@dataclass
class TraceRequest:
    arrival: float
    prompt: str
    max_tokens: int


def make_trace(
    n: int = 64, rate: float = 40.0, seed: int = 0, prompt_len=(16, 96), out_len=(8, 160)
) -> list[TraceRequest]:
    """Poisson arrivals; prompts are random slices of the corpus; output lengths
    are log-uniform - most answers short, a few long, as in real traffic. The
    long tail is exactly what hurts static batching."""
    rng = np.random.default_rng(seed)
    text = DATA_PATH.read_text(encoding="utf-8")
    t, out = 0.0, []
    for _ in range(n):
        t += rng.exponential(1.0 / rate) if rate > 0 else 0.0
        L = int(rng.integers(*prompt_len))
        s = int(rng.integers(0, len(text) - L))
        mt = int(np.exp(rng.uniform(np.log(out_len[0]), np.log(out_len[1]))))
        out.append(TraceRequest(t, text[s : s + L], mt))
    return out


def run_trace(
    engine: LLMEngine,
    trace: list[TraceRequest],
    temperature: float = 0.0,
    sample_memory: bool = False,
) -> dict:
    sim = 0.0
    engine.clock = lambda: sim
    pending = sorted(trace, key=lambda r: r.arrival)
    i = 0
    arrival: dict[str, float] = {}
    first: dict[str, float] = {}
    finish: dict[str, float] = {}
    last: dict[str, float] = {}
    gaps: list[float] = []  # inter-token latency: time between a request's tokens
    step_s: list[float] = []
    gen = 0
    mem: list[tuple[int, int, int]] = []
    bs = engine.config.block_size
    while i < len(pending) or engine.has_unfinished():
        while i < len(pending) and pending[i].arrival <= sim:
            r = pending[i]
            rid = engine.add_request(
                r.prompt,
                SamplingParams(max_tokens=r.max_tokens, temperature=temperature, seed=i),
                arrival_time=r.arrival,
            )
            arrival[rid] = r.arrival
            i += 1
        if not engine.has_unfinished():
            sim = pending[i].arrival  # idle: jump to next arrival
            continue
        t0 = time.perf_counter()
        outs = engine.step()
        dt = time.perf_counter() - t0
        sim += dt
        step_s.append(dt)
        for o in outs:
            gen += len(o.new_token_ids)
            first.setdefault(o.request_id, sim)
            if o.request_id in last:
                gaps.append(sim - last[o.request_id])
            last[o.request_id] = sim
            if o.finished:
                finish[o.request_id] = sim
                engine.release(o.request_id)
        if sample_memory:
            running = engine.scheduler.running
            stored = sum(s.num_computed for s in running)
            mem.append((len(running), stored, engine.bm.alloc.num_used * bs))
    lat = np.array([finish[r] - arrival[r] for r in finish])
    ttft = np.array([first[r] - arrival[r] for r in first])
    start = min(arrival.values())
    makespan = max(finish.values()) - start
    res: dict[str, Any] = {
        "requests": len(finish),
        "generated_tokens": gen,
        "steps": engine.stats.steps,
        "makespan_s": makespan,
        "throughput_tok_s": gen / makespan,
        "latency_mean_s": float(lat.mean()),
        "latency_p50_s": float(np.percentile(lat, 50)),
        "latency_p99_s": float(np.percentile(lat, 99)),
        "ttft_mean_s": float(ttft.mean()),
        "ttft_p99_s": float(np.percentile(ttft, 99)),
        "itl_mean_ms": float(np.mean(gaps) * 1e3) if gaps else 0.0,
        "itl_p99_ms": float(np.percentile(gaps, 99) * 1e3) if gaps else 0.0,
        "itl_max_ms": float(np.max(gaps) * 1e3) if gaps else 0.0,
        "step_p99_ms": float(np.percentile(step_s, 99) * 1e3),
        "preemptions": engine.scheduler.num_preemptions,
        "max_running": engine.stats.max_running,
        "mean_batch": engine.stats.generated_tokens / max(1, engine.stats.steps),
    }
    if sample_memory:
        res["memory_samples"] = mem
    return res


def bench_batching(n=128, rate=400.0, max_num_seqs=16, seed=0) -> dict:
    model, tok = load_pretrained("target")
    trace = make_trace(n, rate, seed)
    out = {"trace": {"requests": n, "rate_req_s": rate, "max_num_seqs": max_num_seqs}}
    for policy in ("static", "continuous"):
        eng = LLMEngine(
            model,
            tok,
            EngineConfig(
                num_blocks=1024,
                block_size=16,
                max_num_seqs=max_num_seqs,
                policy=policy,
                enable_prefix_caching=False,
            ),
        )
        out[policy] = run_trace(eng, trace)
    s, c = out["static"], out["continuous"]
    print(
        f"\n## Static vs continuous batching  ({n} requests, Poisson {rate}/s, "
        f"max {max_num_seqs} concurrent, greedy)\n"
    )
    print(
        "| policy | throughput (tok/s) | mean latency (s) | p99 latency (s) | "
        "mean TTFT (s) | engine steps | mean tokens/step |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, r in (("static", s), ("continuous", c)):
        print(
            f"| {name} | {r['throughput_tok_s']:.0f} | {r['latency_mean_s']:.2f} | "
            f"{r['latency_p99_s']:.2f} | {r['ttft_mean_s']:.2f} | {r['steps']} | "
            f"{r['mean_batch']:.1f} |"
        )
    ratio = c["throughput_tok_s"] / s["throughput_tok_s"]
    print(
        f"\ncontinuous / static throughput: {ratio:.2f}x; "
        f"engine steps {s['steps'] / c['steps']:.2f}x fewer; "
        f"mean latency static / continuous: {s['latency_mean_s'] / c['latency_mean_s']:.2f}x"
    )
    return out


def bench_memory(
    n=128, rate=400.0, num_blocks=256, block_size=16, max_model_len=256, seed=0
) -> dict:
    """Run the same trace through the paged engine and, at every step, compare
    the KV slots paged allocation holds with what contiguous max-length
    reservation would hold for the same running set."""
    model, tok = load_pretrained("target")
    trace = make_trace(n, rate, seed)
    eng = LLMEngine(
        model,
        tok,
        EngineConfig(
            num_blocks=num_blocks,
            block_size=block_size,
            max_num_seqs=64,
            max_model_len=max_model_len,
            enable_prefix_caching=False,
        ),
    )
    r = run_trace(eng, trace, sample_memory=True)
    m = np.array(r.pop("memory_samples"), dtype=float)
    m = m[m[:, 0] > 0]
    running, stored, paged = m[:, 0], m[:, 1], m[:, 2]
    contig = running * max_model_len
    pool_slots = num_blocks * block_size
    out = {
        "pool_slots": pool_slots,
        "block_size": block_size,
        "max_model_len": max_model_len,
        "paged_waste_pct": float(100 * (1 - stored.sum() / paged.sum())),
        "contiguous_waste_pct": float(100 * (1 - stored.sum() / contig.sum())),
        "paged_max_concurrent": int(running.max()),
        "contiguous_max_concurrent": pool_slots // max_model_len,
        "paged_mean_concurrent": float(running.mean()),
        "kv_bytes_per_token": model.cfg.kv_bytes_per_token,
        "preemptions": r["preemptions"],
    }
    print(
        f"\n## KV cache memory: contiguous reservation vs paged "
        f"(block_size={block_size}, max_model_len={max_model_len})\n"
    )
    print(
        "| scheme | slots holding real tokens | wasted (reserved, never used) | "
        "max concurrent seqs in a pool of "
        f"{pool_slots} slots |"
    )
    print("|---|---|---|---|")
    print(
        f"| contiguous (reserve max_model_len) | {100 - out['contiguous_waste_pct']:.1f}% | "
        f"{out['contiguous_waste_pct']:.1f}% | {out['contiguous_max_concurrent']} |"
    )
    print(
        f"| paged | {100 - out['paged_waste_pct']:.1f}% | {out['paged_waste_pct']:.1f}% | "
        f"{out['paged_max_concurrent']} (measured) |"
    )
    print(
        f"\nKV bytes/token: {out['kv_bytes_per_token']} "
        f"({model.cfg.n_kv_heads} KV heads of {model.cfg.n_heads}: GQA halves this)"
    )
    return out


def bench_prefix(n=32, system_len=192, user_len=(8, 32), max_tokens=16, seed=0) -> dict:
    """Chat-style traffic: every request = the same long system prompt + a short
    user turn. With prefix caching only the first request prefills the
    shared part."""
    model, tok = load_pretrained("target")
    rng = np.random.default_rng(seed)
    text = DATA_PATH.read_text(encoding="utf-8")
    system = text[:system_len]
    prompts = []
    for _ in range(n):
        L = int(rng.integers(*user_len))
        s = int(rng.integers(system_len, len(text) - L))
        prompts.append(system + text[s : s + L])
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    out: dict[str, Any] = {}
    texts: dict[str, list[str]] = {}
    for enabled in (False, True):
        eng = LLMEngine(
            model, tok, EngineConfig(num_blocks=1024, max_num_seqs=8, enable_prefix_caching=enabled)
        )
        t0 = time.perf_counter()
        res = eng.generate(prompts, params)
        dt = time.perf_counter() - t0
        key = "prefix_cache" if enabled else "no_cache"
        texts[key] = [r[0].text for r in res]
        st = eng.stats
        out[key] = {
            "seconds": dt,
            "prefill_tokens_computed": st.prefill_tokens,
            "prompt_tokens_from_cache": st.cached_prompt_tokens,
        }
    out["outputs_identical"] = texts["no_cache"] == texts["prefix_cache"]
    a, b = out["no_cache"], out["prefix_cache"]
    total = a["prefill_tokens_computed"]
    print(f"\n## Prefix caching ({n} requests sharing a {system_len}-token system prompt)\n")
    print("| | prompt tokens computed | served from cache | wall time (s) |")
    print("|---|---|---|---|")
    print(f"| no prefix cache | {total} | 0 | {a['seconds']:.2f} |")
    print(
        f"| prefix cache | {b['prefill_tokens_computed']} | {b['prompt_tokens_from_cache']} "
        f"({100 * b['prompt_tokens_from_cache'] / total:.0f}%) | {b['seconds']:.2f} |"
    )
    print(
        f"\nspeedup {a['seconds'] / b['seconds']:.2f}x; greedy outputs identical: "
        f"{out['outputs_identical']}"
    )
    return out


def bench_spec(ks=(1, 2, 4, 6), max_tokens=160, n_prompts=6, seed=0) -> dict:
    target, tok = load_pretrained("target")
    draft, _ = load_pretrained("draft")
    rng = np.random.default_rng(seed)
    text = DATA_PATH.read_text(encoding="utf-8")
    prompts = []
    for _ in range(n_prompts):
        s = int(rng.integers(0, len(text) - 64))
        prompts.append(tok.encode(text[s : s + 64]))
    out: dict[str, Any] = {
        "target_params": target.num_params(),
        "draft_params": draft.num_params(),
        "runs": [],
    }

    def fwd_cost(m, n, reps=300):
        """Mean wall time of one cached forward over n new tokens at context 64."""
        c = KVCache(m.cfg)
        forward_with_caches(m, [c], [[1] * 64])
        t0 = time.perf_counter()
        for _ in range(reps):
            forward_with_caches(m, [c], [[1] * n])
            c.truncate(64)
        return (time.perf_counter() - t0) / reps * 1e6

    costs = {
        "target_1tok_us": fwd_cost(target, 1),
        "target_5tok_us": fwd_cost(target, 5),
        "draft_1tok_us": fwd_cost(draft, 1),
    }
    out["forward_costs_us"] = costs
    print(
        f"\n## Speculative decoding (target {target.num_params() / 1e6:.2f}M params, "
        f"{target.cfg.n_layers} layers; draft {draft.num_params() / 1e6:.3f}M, "
        f"{draft.cfg.n_layers} layer; {n_prompts} prompts x {max_tokens} tokens)\n"
    )
    print("| sampling | k | acceptance rate | tokens / target forward | wall speedup |")
    print("|---|---|---|---|---|")
    for label, temp in (("greedy", 0.0), ("T=0.8", 0.8)):
        base = 0.0
        for i, p in enumerate(prompts):
            _, dt = autoregressive(
                target, p, SamplingParams(max_tokens=max_tokens, temperature=temp, seed=i)
            )
            base += dt
        for k in ks:
            dec = SpeculativeDecoder(target, draft, k)
            acc = prop = emit = fwd = 0
            secs = 0.0
            for i, p in enumerate(prompts):
                _, st = dec.generate(
                    p, SamplingParams(max_tokens=max_tokens, temperature=temp, seed=i)
                )
                acc += st.accepted
                prop += st.proposed
                emit += st.emitted
                fwd += st.target_forwards
                secs += st.seconds
            row = {
                "sampling": label,
                "k": k,
                "acceptance_rate": acc / prop,
                "tokens_per_target_forward": emit / fwd,
                "speedup": base / secs,
                "baseline_s": base,
                "spec_s": secs,
            }
            out["runs"].append(row)
            print(
                f"| {label} | {k} | {row['acceptance_rate']:.2f} | "
                f"{row['tokens_per_target_forward']:.2f} | {row['speedup']:.2f}x |"
            )
    print(
        f"\nmeasured forward cost: target 1 token {costs['target_1tok_us']:.0f}us, "
        f"target 5 tokens {costs['target_5tok_us']:.0f}us, draft 1 token "
        f"{costs['draft_1tok_us']:.0f}us (draft/target = "
        f"{costs['draft_1tok_us'] / costs['target_1tok_us']:.2f})"
    )
    return out


def bench_scaling(batch_sizes=(1, 2, 4, 8, 16, 32), steps=20, context=128, repeats=3) -> dict:
    """Decode cost per step and per token as the batch grows, with the batched
    decode-attention path on and off. This is the curve continuous batching
    lives on: it only pays off if per-token cost falls as the batch grows."""
    model, tok = load_pretrained("target")
    text = DATA_PATH.read_text(encoding="utf-8")
    out: dict = {"context_tokens": context, "rows": []}
    print(
        f"\n## Decode cost vs batch size (context ~{context} tokens, "
        f"best of {repeats} x {steps} steps)\n"
    )
    print(
        "| batch | per-sequence attention (ms/token) | batched decode attention (ms/token) "
        "| speedup |"
    )
    print("|---|---|---|---|")
    for bs in batch_sizes:
        row: dict = {"batch": bs}
        for batched in (False, True):
            eng = LLMEngine(
                model, tok, EngineConfig(num_blocks=1024, max_num_seqs=64, batch_decode=batched)
            )
            for i in range(bs):
                eng.add_request(
                    text[i * 997 : i * 997 + context],
                    SamplingParams(max_tokens=repeats * steps + bs + 8, temperature=0),
                )
            # The token budget admits only a few prompts per step: keep stepping
            # until every sequence is decoding, so no prefill is timed.
            while eng.scheduler.waiting:
                eng.step()
            eng.step()
            timings = []
            for _ in range(repeats):  # best of N: least disturbed by the OS
                t0 = time.perf_counter()
                for _ in range(steps):
                    eng.step()
                timings.append((time.perf_counter() - t0) / steps)
            assert len(eng.scheduler.running) == bs
            ms = min(timings) * 1e3
            row["batched" if batched else "per_sequence"] = {
                "ms_per_step": ms,
                "ms_per_token": ms / bs,
            }
        row["speedup"] = row["per_sequence"]["ms_per_step"] / row["batched"]["ms_per_step"]
        out["rows"].append(row)
        print(
            f"| {bs} | {row['per_sequence']['ms_per_token']:.3f} | "
            f"{row['batched']['ms_per_token']:.3f} | {row['speedup']:.2f}x |"
        )
    return out


def bench_chunked(n=96, rate=200.0, max_num_seqs=16, budgets=(64, 128), seed=0) -> dict:
    """Long prompts arriving while others decode: the case chunked prefill exists
    for. Without chunking a 200-token prefill lands in one step and every running
    request waits for it; with chunking each step is capped at the token budget."""
    model, tok = load_pretrained("target")
    trace = make_trace(n, rate, seed, prompt_len=(96, 224), out_len=(16, 96))
    configs = [("no chunking (budget 256)", False, 256)] + [
        (f"chunked, budget {b}", True, b) for b in budgets
    ]
    out: dict[str, Any] = {
        "trace": {"requests": n, "rate_req_s": rate, "max_num_seqs": max_num_seqs},
        "runs": [],
    }
    for label, chunked, budget in configs:
        eng = LLMEngine(
            model,
            tok,
            EngineConfig(
                num_blocks=1024,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=budget,
                enable_chunked_prefill=chunked,
                enable_prefix_caching=False,
            ),
        )
        r = run_trace(eng, trace)
        r["label"], r["budget"], r["chunked"] = label, budget, chunked
        out["runs"].append(r)
    print(
        f"\n## Chunked prefill ({n} requests, prompts 96 to 224 tokens, Poisson {rate}/s, "
        f"max {max_num_seqs} concurrent)\n"
    )
    print(
        "| config | p99 step (ms) | ITL mean (ms) | ITL p99 (ms) | ITL max (ms) | "
        "mean TTFT (s) | throughput (tok/s) |"
    )
    print("|---|---|---|---|---|---|---|")
    for r in out["runs"]:
        print(
            f"| {r['label']} | {r['step_p99_ms']:.1f} | {r['itl_mean_ms']:.1f} | "
            f"{r['itl_p99_ms']:.1f} | {r['itl_max_ms']:.1f} | {r['ttft_mean_s']:.2f} | "
            f"{r['throughput_tok_s']:.0f} |"
        )
    return out


def bench_http(requests=64, concurrency=16, budget=64, seed=0) -> dict:
    """The full stack over real HTTP: launch ``pagellm serve`` as a subprocess
    (chunked prefill off, then on) and drive it with concurrent streaming clients."""
    from .loadtest import make_workload, print_report, run_load

    workload = make_workload(requests, seed)
    out: dict[str, Any] = {"runs": []}
    for label, extra in (
        ("no chunking", []),
        (
            f"chunked, budget {budget}",
            ["--chunked-prefill", "--max-num-batched-tokens", str(budget)],
        ),
    ):
        with _server_process(extra) as url:
            r = run_load(url, requests, concurrency, workload=workload)
        r["label"] = label
        out["runs"].append(r)
        print_report(r, f"HTTP load test, {label}")
    return out


@contextlib.contextmanager
def _server_process(extra: list[str]):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pagellm",
            "serve",
            "--port",
            str(port),
            "--num-blocks",
            "1024",
            "--log-level",
            "warning",
            *extra,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 60
        while True:
            try:
                urllib.request.urlopen(url + "/health", timeout=1)
                break
            except OSError:
                if proc.poll() is not None or time.time() > deadline:
                    raise RuntimeError("benchmark server failed to start") from None
                time.sleep(0.1)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=15)


BENCHES: dict[str, Callable[..., dict]] = {
    "batching": bench_batching,
    "memory": bench_memory,
    "prefix": bench_prefix,
    "spec": bench_spec,
    "scaling": bench_scaling,
    "chunked": bench_chunked,
    "http": bench_http,
}

# Small, fast settings for CI smoke runs: same code paths, smaller workloads.
QUICK: dict[str, dict[str, Any]] = {
    "batching": dict(n=24, rate=400.0, max_num_seqs=8),
    "memory": dict(n=24, rate=400.0),
    "prefix": dict(n=8, max_tokens=8),
    "spec": dict(ks=(2, 4), max_tokens=48, n_prompts=2),
    "scaling": dict(batch_sizes=(1, 8), steps=5),
    "chunked": dict(n=16, budgets=(64,)),
    "http": dict(requests=12, concurrency=4),
}


def environment() -> dict:
    """Where the numbers came from: results are only comparable on equal hardware."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def main(which: str = "all", json_out: str | None = None, quick: bool = False) -> dict:
    names = list(BENCHES) if which == "all" else [which]
    results: dict = {"environment": environment(), "quick": quick}
    for n in names:
        kwargs = QUICK[n] if quick else {}
        results[n] = BENCHES[n](**kwargs)
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results
