import json
import threading
import urllib.error
import urllib.request

import pytest

from pagellm.engine import EngineConfig, LLMEngine
from pagellm.serving import Server


@pytest.fixture(scope="module")
def server(model, tok):
    eng = LLMEngine(model, tok, EngineConfig(num_blocks=256, block_size=8))
    srv = Server(eng, port=0).start_background()
    yield srv
    srv.shutdown()


def post(srv, body, raw=False):
    req = urllib.request.Request(
        f"http://127.0.0.1:{srv.port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    return resp if raw else json.loads(resp.read())


def sse_events(resp):
    events = []
    for line in resp.read().decode().split("\n\n"):
        if line.startswith("data: "):
            events.append(line[6:])
    return events


def test_health_and_models(server):
    base = f"http://127.0.0.1:{server.port}"
    assert json.loads(urllib.request.urlopen(base + "/health").read())["status"] == "ok"
    models = json.loads(urllib.request.urlopen(base + "/v1/models").read())
    assert models["data"][0]["id"] == "pagellm-shakespeare"


def test_completion(server):
    r = post(server, {"prompt": "ROMEO:", "max_tokens": 12, "temperature": 0})
    assert r["object"] == "text_completion"
    ch = r["choices"][0]
    assert len(ch["text"]) == 12 and ch["finish_reason"] == "length"
    assert r["usage"] == {"prompt_tokens": 6, "completion_tokens": 12, "total_tokens": 18}


def test_streaming_matches_blocking(server):
    body = {"prompt": "To be, or not", "max_tokens": 20, "temperature": 0}
    full = post(server, body)["choices"][0]["text"]
    resp = post(server, {**body, "stream": True}, raw=True)
    assert resp.headers["Content-Type"] == "text/event-stream"
    events = sse_events(resp)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert len(chunks) > 1  # actually incremental
    assert "".join(c["choices"][0]["text"] for c in chunks) == full
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_concurrent_requests_are_batched(server):
    results = {}

    def go(i):
        results[i] = post(
            server, {"prompt": f"Speak {i}", "max_tokens": 30, "temperature": 0.7, "seed": i}
        )["choices"][0]["text"]

    threads = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 8 and all(len(t) == 30 for t in results.values())
    # seeded => identical to a sequential re-run
    assert (
        post(server, {"prompt": "Speak 3", "max_tokens": 30, "temperature": 0.7, "seed": 3})[
            "choices"
        ][0]["text"]
        == results[3]
    )


def test_n_and_stop(server):
    r = post(server, {"prompt": "A", "max_tokens": 10, "n": 2, "seed": 1})
    assert [c["index"] for c in r["choices"]] == [0, 1]
    base = post(server, {"prompt": "B", "max_tokens": 20, "temperature": 0})["choices"][0]["text"]
    stop = base[5:7]
    r = post(server, {"prompt": "B", "max_tokens": 20, "temperature": 0, "stop": stop})
    assert r["choices"][0]["text"] == base[: base.find(stop)]
    assert r["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "body", [{}, {"prompt": 5}, {"prompt": "x", "temperature": -1}, {"prompt": ""}]
)
def test_bad_requests(server, body):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(server, body)
    assert e.value.code == 400


def get(srv, path):
    return urllib.request.urlopen(f"http://127.0.0.1:{srv.port}{path}", timeout=30)


def test_bad_seed_is_rejected_and_engine_survives(server):
    """Regression: a string seed used to raise TypeError inside the engine thread,
    killing it and hanging every later request."""
    with pytest.raises(urllib.error.HTTPError) as e:
        post(server, {"prompt": "x", "seed": "abc"})
    assert e.value.code == 400
    assert server.loop.alive
    assert len(post(server, {"prompt": "x", "max_tokens": 3})["choices"][0]["text"]) == 3


def test_malformed_json_and_unknown_routes(server):
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.port}/v1/completions",
        data=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=30)
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server, "/nope")
    assert e.value.code == 404


def test_prompt_longer_than_context_is_400(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(server, {"prompt": "z" * 5000, "max_tokens": 1})
    assert e.value.code == 400
    assert "max_model_len" in json.loads(e.value.read())["error"]["message"]


def test_metrics_endpoint(server):
    post(server, {"prompt": "metrics", "max_tokens": 4})
    text = get(server, "/metrics").read().decode()
    assert "# TYPE pagellm_engine_steps_total counter" in text
    lines = dict(line.rsplit(" ", 1) for line in text.splitlines() if not line.startswith("#"))
    assert float(lines["pagellm_generated_tokens_total"]) >= 4
    assert float(lines['pagellm_http_requests_total{route="/v1/completions",status="200"}']) >= 1
    assert lines["pagellm_kv_blocks_total"] == "256"


def test_health_reports_queue_depth(server):
    h = json.loads(get(server, "/health").read())
    assert h == {"status": "ok", "running": 0, "waiting": 0}


def test_client_disconnect_mid_stream_frees_blocks(server):
    import socket
    import time

    body = json.dumps(
        {"prompt": "Bye", "max_tokens": 200, "stream": True, "temperature": 0}
    ).encode()
    s = socket.create_connection(("127.0.0.1", server.port), timeout=10)
    s.sendall(
        b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    )
    assert b"200 OK" in s.recv(4096)  # stream started
    s.close()
    engine = server.loop.engine
    deadline = time.time() + 10
    while time.time() < deadline and (engine.has_unfinished() or engine.bm.alloc.num_used):
        time.sleep(0.05)
    assert not engine.has_unfinished() and engine.bm.alloc.num_used == 0


def test_engine_failure_returns_500_then_recovers(model, tok):
    eng = LLMEngine(model, tok, EngineConfig(num_blocks=64, block_size=8))
    real_step, calls = eng.step, {"n": 0}

    def flaky_step():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated kernel fault")
        return real_step()

    eng.step = flaky_step
    srv = Server(eng, port=0).start_background()
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            post(srv, {"prompt": "first", "max_tokens": 5})
        assert e.value.code == 500
        assert json.loads(e.value.read())["error"]["type"] == "server_error"
        assert eng.bm.alloc.num_used == 0  # failed request's blocks reclaimed
        r = post(srv, {"prompt": "second", "max_tokens": 5})
        assert len(r["choices"][0]["text"]) == 5
    finally:
        srv.shutdown()


def test_load_tester_against_chunked_server(model, tok):
    from pagellm.benchmarks.loadtest import make_workload, run_load

    eng = LLMEngine(
        model,
        tok,
        EngineConfig(
            num_blocks=256, block_size=8, max_num_batched_tokens=32, enable_chunked_prefill=True
        ),
    )
    srv = Server(eng, port=0).start_background()
    try:
        work = make_workload(10, seed=1, prompt_len=(40, 120), out_len=(5, 20))
        r = run_load(f"http://127.0.0.1:{srv.port}", concurrency=5, workload=work)
    finally:
        srv.shutdown()
    assert r["succeeded"] == 10 and r["failed"] == 0
    assert r["ttft_ms"]["p50"] > 0 and r["itl_ms"]["p99"] >= r["itl_ms"]["p50"]
    assert r["output_tok_per_s"] > 0
    assert eng.bm.alloc.num_used == 0


def test_load_tester_reports_failures():
    from helpers import free_port
    from pagellm.benchmarks.loadtest import run_load

    r = run_load(f"http://127.0.0.1:{free_port()}", n_requests=3, concurrency=2, timeout=2)
    assert r["succeeded"] == 0 and r["failed"] == 3
