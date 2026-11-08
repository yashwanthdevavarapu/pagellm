"""Shared test helpers (importable because pytest puts tests/ on sys.path)."""

from __future__ import annotations

import socket

import numpy as np

from pagellm.model.transformer import ModelConfig, Transformer


def tiny_model(
    vocab: int,
    seed: int = 0,
    layers: int = 2,
    dim: int = 32,
    heads: int = 4,
    kv_heads: int = 2,
    max_seq_len: int = 256,
) -> Transformer:
    """A random but *opinionated* model for exercising cache paths quickly.

    Default init is tiny (std 0.02), which makes every logit nearly equal; scaling
    the matrices up makes attention peaky and argmax decisions far from ties, so a
    wrong cache read changes the output instead of hiding in round-off.
    """
    cfg = ModelConfig(
        vocab_size=vocab,
        dim=dim,
        n_layers=layers,
        n_heads=heads,
        n_kv_heads=kv_heads,
        ffn_dim=2 * dim + 16,
        max_seq_len=max_seq_len,
    )
    m = Transformer(cfg, seed=seed)
    for w in m.w.values():
        if w.ndim == 2:
            w *= 20.0
    return m


def reference_greedy(model: Transformer, tok, prompt: str, n: int) -> str:
    """Uncached greedy decoding: the oracle every engine configuration must match."""
    ids = tok.encode(prompt)
    for _ in range(n):
        ids.append(int(np.argmax(model.logits_full(ids)[-1])))
    return tok.decode(ids[len(tok.encode(prompt)) :])


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
