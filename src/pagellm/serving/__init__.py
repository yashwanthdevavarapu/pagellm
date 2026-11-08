"""OpenAI-compatible HTTP serving: protocol, metrics, and the threaded server."""

from .protocol import CompletionRequest, ProtocolError, parse_completion_request
from .server import DEFAULT_MODEL_NAME, EngineLoop, Server

__all__ = [
    "DEFAULT_MODEL_NAME",
    "CompletionRequest",
    "EngineLoop",
    "ProtocolError",
    "Server",
    "parse_completion_request",
]
