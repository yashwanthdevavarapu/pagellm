import math

import pytest

from pagellm.serving.protocol import (
    MAX_N,
    ProtocolError,
    choice,
    completion_chunk,
    error_body,
    parse_completion_request,
)


def test_defaults_follow_openai():
    r = parse_completion_request({"prompt": "hi"})
    assert r.prompt == "hi" and r.stream is False
    assert r.params.max_tokens == 16 and r.params.temperature == 1.0 and r.params.n == 1
    assert r.params.stop == () and r.params.seed is None


def test_full_request_is_mapped():
    r = parse_completion_request(
        {
            "prompt": ["ROMEO:"],
            "max_tokens": 5,
            "temperature": 0,
            "top_k": 3,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "seed": 7,
            "n": 2,
            "stop": ["\n", "END"],
            "stream": True,
        }
    )
    p = r.params
    assert r.prompt == "ROMEO:" and r.stream
    assert (p.max_tokens, p.temperature, p.top_k, p.top_p) == (5, 0.0, 3, 0.9)
    assert (p.repetition_penalty, p.seed, p.n, p.stop) == (1.2, 7, 2, ("\n", "END"))


def test_single_stop_string_is_wrapped():
    assert parse_completion_request({"prompt": "a", "stop": "x"}).params.stop == ("x",)


@pytest.mark.parametrize(
    "body",
    [
        [],  # not an object
        {},  # no prompt
        {"prompt": 5},
        {"prompt": ""},
        {"prompt": ["a", "b"]},
        {"prompt": [1]},
        {"prompt": "a", "seed": "abc"},  # used to crash the engine thread
        {"prompt": "a", "seed": -1},
        {"prompt": "a", "seed": True},
        {"prompt": "a", "seed": 1.5},
        {"prompt": "a", "max_tokens": 0},
        {"prompt": "a", "max_tokens": True},
        {"prompt": "a", "max_tokens": "10"},
        {"prompt": "a", "n": MAX_N + 1},
        {"prompt": "a", "temperature": -1},
        {"prompt": "a", "temperature": math.nan},
        {"prompt": "a", "temperature": "hot"},
        {"prompt": "a", "top_p": 0},
        {"prompt": "a", "top_k": -2},
        {"prompt": "a", "repetition_penalty": 0},
        {"prompt": "a", "stop": ""},
        {"prompt": "a", "stop": ["a", "b", "c", "d", "e"]},
        {"prompt": "a", "stop": [1]},
        {"prompt": "a", "stream": "yes"},
    ],
)
def test_invalid_requests_raise_protocol_error(body):
    with pytest.raises(ProtocolError):
        parse_completion_request(body)


def test_response_shapes():
    c = completion_chunk("cmpl-1", 123, "m", [choice(0, "hi", "length")])
    assert c == {
        "id": "cmpl-1",
        "object": "text_completion",
        "created": 123,
        "model": "m",
        "choices": [{"index": 0, "text": "hi", "logprobs": None, "finish_reason": "length"}],
    }
    assert error_body("nope")["error"] == {"message": "nope", "type": "invalid_request_error"}
