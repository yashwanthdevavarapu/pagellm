"""OpenAI-compatible HTTP server (stdlib only).

    POST /v1/completions   {"prompt": "...", "max_tokens": 64, "stream": true, ...}
    GET  /v1/models
    GET  /health
    GET  /metrics          Prometheus text format

Threading model: ``ThreadingHTTPServer`` gives each connection a thread, but
the engine is single-threaded by design (the scheduler owns global state, and
batching only works if one loop sees every request). So handler threads never
touch the engine directly: they enqueue a submission and then block on a
per-request ``queue.Queue`` of deltas. One engine thread drains submissions,
calls ``engine.step()`` (which advances *all* in-flight requests at once) and
fans the deltas out. See docs/adr/0004-single-engine-thread.md.

Failure isolation: a bad request is rejected before it reaches the engine
thread, and if a step ever raises, the in-flight requests are failed with a 500
and the loop keeps serving. A client that disconnects mid-stream is detected on
the next write and its request is aborted, freeing its KV blocks.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..engine.llm_engine import LLMEngine, StepOutput
from ..engine.sampling import SamplingParams
from .metrics import ServerMetrics, render
from .protocol import (
    ProtocolError,
    choice,
    completion_chunk,
    error_body,
    parse_completion_request,
)

log = logging.getLogger("pagellm.server")

DEFAULT_MODEL_NAME = "pagellm-shakespeare"
MAX_BODY_BYTES = 1 << 20
_DONE = object()


@dataclass(frozen=True)
class Failure:
    """Sent down a request's queue instead of a delta when it cannot complete."""

    status: int
    message: str


class EngineLoop:
    """Owns the engine; the only thread that ever calls into it."""

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._submit: queue.Queue = queue.Queue()
        self._streams: dict[str, queue.Queue] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._submit.put(None)
        self._thread.join(timeout=5)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def submit(self, prompt: str, params: SamplingParams) -> tuple[str, queue.Queue]:
        rid = f"cmpl-{uuid.uuid4().hex[:24]}"
        q: queue.Queue = queue.Queue()
        self._submit.put(("add", rid, prompt, params, q))
        return rid, q

    def abort(self, rid: str) -> None:
        self._submit.put(("abort", rid))

    def _drain(self, block: bool) -> None:
        try:
            item = self._submit.get(block=block, timeout=0.5 if block else None)
        except queue.Empty:
            return
        while item is not None:
            if item[0] == "add":
                _, rid, prompt, params, q = item
                try:
                    self.engine.add_request(prompt, params, request_id=rid)
                    self._streams[rid] = q
                except ValueError as e:  # e.g. prompt longer than max_model_len
                    q.put(Failure(400, str(e)))
                except Exception:  # never let one request kill the loop
                    log.exception("add_request failed for %s", rid)
                    q.put(Failure(500, "internal error while admitting request"))
            else:
                rid = item[1]
                self.engine.abort(rid)
                self.engine.release(rid)
                self._streams.pop(rid, None)
            try:
                item = self._submit.get_nowait()
            except queue.Empty:
                item = None

    def _fail_all(self) -> None:
        for rid, q in list(self._streams.items()):
            self.engine.abort(rid)
            self.engine.release(rid)
            q.put(Failure(500, "internal engine error"))
        self._streams.clear()

    def _run(self) -> None:
        while not self._stop.is_set():
            # Block only when idle; while requests are in flight, never wait:
            # every millisecond spent sleeping is a millisecond of no decoding.
            self._drain(block=not self.engine.has_unfinished())
            if not self.engine.has_unfinished():
                continue
            try:
                outputs = self.engine.step()
            except Exception:
                log.exception("engine step failed; failing in-flight requests")
                self._fail_all()
                continue
            for out in outputs:
                q = self._streams.get(out.request_id)
                if q is not None:
                    q.put(out)
            live = self._unfinished()
            for rid in [r for r in self._streams if r not in live]:
                self._streams.pop(rid).put(_DONE)
                self.engine.release(rid)

    def _unfinished(self) -> set[str]:
        s = self.engine.scheduler
        return {x.request_id for x in (*s.waiting, *s.running)}


def make_handler(loop: EngineLoop, model_name: str, metrics: ServerMetrics):
    engine = loop.engine

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "pagellm"

        def log_message(self, fmt, *args):
            log.debug("%s %s", self.address_string(), fmt % args)

        @property
        def route(self) -> str:
            return self.path.split("?", 1)[0]

        def _send(self, code: int, data: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            metrics.record(self.route, code)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _error(self, code: int, msg: str) -> None:
            kind = "invalid_request_error" if code < 500 else "server_error"
            self._json(code, error_body(msg, kind))

        def do_GET(self):
            if self.route == "/health":
                ok = loop.alive
                self._json(
                    200 if ok else 503,
                    {
                        "status": "ok" if ok else "engine_down",
                        "running": len(engine.scheduler.running),
                        "waiting": len(engine.scheduler.waiting),
                    },
                )
            elif self.route == "/v1/models":
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": model_name,
                                "object": "model",
                                "owned_by": "pagellm",
                                "max_model_len": engine.max_model_len,
                            }
                        ],
                    },
                )
            elif self.route == "/metrics":
                self._send(200, render(engine, metrics).encode(), "text/plain; version=0.0.4")
            else:
                self._error(404, f"no route {self.route}")

        def do_POST(self):
            if self.route != "/v1/completions":
                return self._error(404, f"no route {self.route}")
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > MAX_BODY_BYTES:
                    return self._error(413, f"request body over {MAX_BODY_BYTES} bytes")
                body = json.loads(self.rfile.read(n) or b"{}")
                req = parse_completion_request(body)
            except (ProtocolError, ValueError) as e:  # JSONDecodeError is a ValueError
                return self._error(400, f"bad request: {e}")
            rid, q = loop.submit(req.prompt, req.params)
            created = int(time.time())
            prompt_tokens = len(engine.tok.encode(req.prompt))
            if req.stream:
                self._stream(rid, q, created)
            else:
                self._blocking(rid, q, created, req.params.n, prompt_tokens)

        def _blocking(self, rid, q, created, n, prompt_tokens):
            texts = [""] * n
            reasons: list[str | None] = [None] * n
            completion_tokens = 0
            while (item := q.get()) is not _DONE:
                if isinstance(item, Failure):
                    return self._error(item.status, item.message)
                texts[item.index] += item.new_text
                completion_tokens += len(item.new_token_ids)
                if item.finished:
                    reasons[item.index] = item.finish_reason
            out = completion_chunk(
                rid, created, model_name, [choice(i, texts[i], reasons[i]) for i in range(n)]
            )
            out["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            self._json(200, out)

        def _write_event(self, payload: bytes) -> None:
            self.wfile.write(b"data: " + payload + b"\n\n")
            self.wfile.flush()

        def _stream(self, rid, q, created):
            first = q.get()
            if isinstance(first, Failure):
                return self._error(first.status, first.message)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            # No Content-Length on a stream: close the connection to delimit it.
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            metrics.record(self.route, 200)
            item = first
            try:
                while item is not _DONE:
                    if isinstance(item, Failure):
                        self._write_event(
                            json.dumps(error_body(item.message, "server_error")).encode()
                        )
                        break
                    out: StepOutput = item
                    if out.new_text or out.finished:
                        ev = completion_chunk(
                            rid,
                            created,
                            model_name,
                            [choice(out.index, out.new_text, out.finish_reason)],
                        )
                        self._write_event(json.dumps(ev).encode())
                        metrics.chunk()
                    item = q.get()
                self._write_event(b"[DONE]")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                loop.abort(rid)  # client went away: give its KV blocks back
                metrics.abort()

    return Handler


class _HTTPServer(ThreadingHTTPServer):
    # socketserver's default listen backlog is 5: a burst of 8 simultaneous
    # clients gets connections reset by the OS before a handler ever runs.
    request_queue_size = 128
    daemon_threads = True


class Server:
    def __init__(
        self,
        engine: LLMEngine,
        host: str = "127.0.0.1",
        port: int = 8000,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.loop = EngineLoop(engine)
        self.metrics = ServerMetrics()
        self.httpd = _HTTPServer((host, port), make_handler(self.loop, model_name, self.metrics))

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def start_background(self) -> Server:
        self.loop.start()
        threading.Thread(target=self.httpd.serve_forever, name="http", daemon=True).start()
        return self

    def serve_forever(self) -> None:
        self.loop.start()
        try:
            self.httpd.serve_forever()
        finally:
            self.loop.stop()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.loop.stop()
