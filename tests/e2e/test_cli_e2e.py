"""End-to-end: run the installed CLI exactly as a user would, in a subprocess,
with the real bundled weights, and talk to it only over HTTP and stdout."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from conftest import needs_weights
from helpers import free_port

pytestmark = [pytest.mark.e2e, needs_weights]

ROOT = Path(__file__).resolve().parents[2]
CLI = [sys.executable, "-m", "pagellm"]


def run_cli(*args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*CLI, *args], capture_output=True, text=True, timeout=timeout, encoding="utf-8", cwd=ROOT
    )


@pytest.fixture(scope="module")
def live_server():
    port = free_port()
    proc = subprocess.Popen(
        [*CLI, "serve", "--port", str(port), "--num-blocks", "512", "--log-level", "warning"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while True:
        try:
            if json.loads(urllib.request.urlopen(base + "/health", timeout=2).read())["status"]:
                break
        except (urllib.error.URLError, ConnectionError):
            pass
        if proc.poll() is not None or time.time() > deadline:
            proc.kill()
            pytest.fail(f"server did not start:\n{proc.stdout.read()}")
        time.sleep(0.2)
    yield base
    proc.terminate()
    proc.wait(timeout=15)


def post(base: str, body: dict, timeout: float = 60):
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def test_version_flag():
    r = run_cli("--version")
    assert r.returncode == 0 and r.stdout.startswith("pagellm ")


def test_generate_streams_shakespeare():
    r = run_cli("generate", "ROMEO:", "--max-tokens", "120", "--temperature", "0")
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("ROMEO:") and len(r.stdout) >= 120
    assert any(w in r.stdout.lower().split() for w in ("the", "and", "to", "i", "my", "of"))


def test_generate_speculative_reports_stats():
    r = run_cli(
        "generate", "JULIET:", "--max-tokens", "60", "--temperature", "0", "--speculative", "4"
    )
    assert r.returncode == 0, r.stderr
    assert "acceptance" in r.stderr and "tokens/target forward" in r.stderr


def test_generate_parallel_samples():
    r = run_cli("generate", "To be", "--max-tokens", "20", "-n", "3", "--seed", "1")
    assert r.returncode == 0, r.stderr
    assert all(f"--- sample {i} ---" in r.stdout for i in range(3))


def test_models_and_health(live_server):
    models = json.loads(urllib.request.urlopen(live_server + "/v1/models").read())
    assert models["data"][0]["id"] == "pagellm-shakespeare"
    assert models["data"][0]["max_model_len"] == 256


def test_blocking_completion_is_deterministic(live_server):
    body = {"prompt": "JULIET:\n", "max_tokens": 60, "temperature": 0.7, "seed": 3}
    a = json.loads(post(live_server, body).read())
    b = json.loads(post(live_server, body).read())
    assert a["choices"][0]["text"] == b["choices"][0]["text"]
    assert a["usage"] == {"prompt_tokens": 8, "completion_tokens": 60, "total_tokens": 68}


def test_sse_stream_equals_blocking(live_server):
    body = {"prompt": "KING HENRY VI:\n", "max_tokens": 80, "temperature": 0}
    blocking = json.loads(post(live_server, body).read())["choices"][0]["text"]
    pieces, done = [], False
    with post(live_server, {**body, "stream": True}) as resp:
        assert resp.headers["Content-Type"] == "text/event-stream"
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                break
            pieces.append(json.loads(line[6:])["choices"][0]["text"])
    assert done and len(pieces) > 10 and "".join(pieces) == blocking


def test_sixteen_concurrent_clients(live_server):
    def one(i):
        r = json.loads(
            post(
                live_server,
                {"prompt": f"Citizen {i}:\n", "max_tokens": 40, "temperature": 0.8, "seed": i},
            ).read()
        )
        return r["choices"][0]["text"]

    with ThreadPoolExecutor(16) as ex:
        texts = list(ex.map(one, range(16)))
    assert all(len(t) == 40 for t in texts)
    assert one(5) == texts[5]  # batching does not change seeded outputs


def test_metrics_reflect_traffic(live_server):
    text = urllib.request.urlopen(live_server + "/metrics").read().decode()
    vals = {
        k: float(v)
        for k, v in (ln.rsplit(" ", 1) for ln in text.splitlines() if ln and not ln.startswith("#"))
    }
    assert vals["pagellm_generated_tokens_total"] > 500
    assert vals["pagellm_running_sequences"] == 0
    assert vals["pagellm_kv_blocks_used"] == 0  # nothing leaked after all tests


def test_loadtest_cli_against_live_server(live_server, tmp_path):
    out = tmp_path / "load.json"
    r = run_cli(
        "loadtest",
        "--url",
        live_server,
        "--requests",
        "12",
        "--concurrency",
        "4",
        "--json",
        str(out),
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text())
    assert data["succeeded"] == 12 and data["output_tok_per_s"] > 0
    assert "inter-token latency" in r.stdout


def test_serve_with_chunked_prefill_flag():
    port = free_port()
    proc = subprocess.Popen(
        [
            *CLI,
            "serve",
            "--port",
            str(port),
            "--chunked-prefill",
            "--max-num-batched-tokens",
            "32",
            "--log-level",
            "warning",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=ROOT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(base + "/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        prompt = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 3
        r = json.loads(post(base, {"prompt": prompt, "max_tokens": 20, "temperature": 0}).read())
        assert r["usage"]["prompt_tokens"] > 32  # needed several chunks
        assert r["usage"]["completion_tokens"] == 20
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_bad_request_over_the_wire(live_server):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(live_server, {"prompt": "x", "seed": "abc"})
    assert e.value.code == 400
    assert json.loads(e.value.read())["error"]["type"] == "invalid_request_error"


def test_bench_quick_writes_json(tmp_path):
    out = tmp_path / "bench.json"
    r = run_cli("bench", "prefix", "--quick", "--json", str(out), timeout=300)
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text())
    assert data["prefix"]["outputs_identical"] is True
    assert data["environment"]["numpy"]


@pytest.mark.parametrize(
    "script",
    [
        "01_offline_batch.py",
        "02_stream_client.py",
        "03_speculative.py",
        "04_paged_kv_walkthrough.py",
    ],
)
def test_examples_run(script):
    r = subprocess.run(
        [sys.executable, str(ROOT / "examples" / script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip()
