"""OpenAI ``/v1/completions`` wire format: request validation and response shapes.

Kept free of sockets and threads so every rule here is unit-testable. Anything a
client can send is validated *before* it reaches the engine thread: a bad value
must become a 400 for that client, never an exception inside the shared loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..engine.sampling import SamplingParams

MAX_N = 16
MAX_STOP_STRINGS = 4


class ProtocolError(ValueError):
    """A request the server must reject with HTTP 400."""


@dataclass(frozen=True)
class CompletionRequest:
    prompt: str
    params: SamplingParams
    stream: bool


def _int(body: dict, key: str, default: int, lo: int, hi: int | None = None) -> int:
    v = body.get(key, default)
    # bool is a subclass of int; {"max_tokens": true} is a client bug, not 1.
    if isinstance(v, bool) or not isinstance(v, int):
        raise ProtocolError(f"{key} must be an integer")
    if v < lo or (hi is not None and v > hi):
        bound = f">= {lo}" if hi is None else f"in [{lo}, {hi}]"
        raise ProtocolError(f"{key} must be {bound}")
    return v


def _float(body: dict, key: str, default: float) -> float:
    v = body.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ProtocolError(f"{key} must be a finite number")
    return float(v)


def _prompt(body: dict) -> str:
    if "prompt" not in body:
        raise ProtocolError("missing required field: prompt")
    prompt = body["prompt"]
    if isinstance(prompt, list):
        if len(prompt) != 1 or not isinstance(prompt[0], str):
            raise ProtocolError("only a single string prompt is supported")
        prompt = prompt[0]
    if not isinstance(prompt, str):
        raise ProtocolError("prompt must be a string")
    if not prompt:
        raise ProtocolError("prompt must not be empty")
    return prompt


def _stop(body: dict) -> tuple[str, ...]:
    stop = body.get("stop")
    if stop is None:
        return ()
    if isinstance(stop, str):
        stop = [stop]
    if (
        not isinstance(stop, list)
        or len(stop) > MAX_STOP_STRINGS
        or not all(isinstance(s, str) and s for s in stop)
    ):
        raise ProtocolError(
            f"stop must be a non-empty string or a list of up to "
            f"{MAX_STOP_STRINGS} non-empty strings"
        )
    return tuple(stop)


def parse_completion_request(body: Any) -> CompletionRequest:
    """Validate a decoded JSON body and turn it into engine inputs."""
    if not isinstance(body, dict):
        raise ProtocolError("request body must be a JSON object")
    prompt = _prompt(body)
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ProtocolError("seed must be a non-negative integer")
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise ProtocolError("stream must be a boolean")
    try:
        params = SamplingParams(
            max_tokens=_int(body, "max_tokens", 16, 1),  # 16 is OpenAI's default
            temperature=_float(body, "temperature", 1.0),
            top_k=_int(body, "top_k", 0, 0),
            top_p=_float(body, "top_p", 1.0),
            repetition_penalty=_float(body, "repetition_penalty", 1.0),
            seed=seed,
            n=_int(body, "n", 1, 1, MAX_N),
            stop=_stop(body),
        )
    except ProtocolError:
        raise
    except ValueError as e:  # range checks owned by SamplingParams
        raise ProtocolError(str(e)) from e
    return CompletionRequest(prompt, params, stream)


def error_body(message: str, kind: str = "invalid_request_error") -> dict:
    return {"error": {"message": message, "type": kind}}


def completion_chunk(rid: str, created: int, model: str, choices: list[dict]) -> dict:
    return {
        "id": rid,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": choices,
    }


def choice(index: int, text: str, finish_reason: str | None) -> dict:
    return {"index": index, "text": text, "logprobs": None, "finish_reason": finish_reason}
