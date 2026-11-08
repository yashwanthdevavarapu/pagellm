"""Chunked prefill must change *when* prompt tokens are computed, never *what* comes out."""

from __future__ import annotations

import pytest

from pagellm.engine import EngineConfig, LLMEngine
from pagellm.engine.sampling import SamplingParams

PROMPTS = [
    "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 2,
    "ROMEO:\nBut soft, what light through yonder window breaks?",
    "To be, or not",
    "KING RICHARD III:\nNow is the winter of our discontent\nMade glorious summer",
    "A",
]


def engine(model, tok, chunked: bool, budget: int, **kw) -> LLMEngine:
    cfg = dict(
        num_blocks=128,
        block_size=8,
        max_num_seqs=8,
        max_num_batched_tokens=budget,
        enable_chunked_prefill=chunked,
    )
    cfg.update(kw)
    return LLMEngine(model, tok, EngineConfig(**cfg))


def texts(outs):
    return [o[0].text for o in outs]


@pytest.mark.parametrize("budget", [1, 7, 16, 48])
@pytest.mark.parametrize("temperature", [0.0, 0.9])
def test_outputs_identical_to_unchunked(model, tok, budget, temperature):
    p = SamplingParams(max_tokens=24, temperature=temperature, top_k=20, seed=7)
    want = texts(engine(model, tok, False, 256).generate(PROMPTS, p))
    got = engine(model, tok, True, budget)
    assert texts(got.generate(PROMPTS, p)) == want
    assert got.bm.alloc.num_used == 0


def test_long_prompt_spans_steps_and_samples_only_at_the_end(model, tok):
    eng = engine(model, tok, True, 16)
    prompt = "x" * 50  # 50 tokens at 16 per step: chunks of 16, 16, 16, 2
    eng.add_request(prompt, SamplingParams(max_tokens=3, temperature=0))
    chunks, outputs = [], []
    while eng.has_unfinished():
        seq = (eng.scheduler.running or list(eng.scheduler.waiting))[0]
        before = seq.num_computed
        outputs.append(eng.step())
        chunks.append(seq.num_computed - before if seq.num_computed > before else None)
    assert chunks[:4] == [16, 16, 16, 2]
    assert [len(o) for o in outputs[:4]] == [0, 0, 0, 1]  # first token after the last chunk
    assert eng.stats.prefill_tokens == 50


def test_decodes_are_never_stalled_by_a_long_prefill(model, tok):
    """With a 16-token budget, a 200-token prompt arriving mid-stream must not stop
    the running request from getting one token every step."""
    eng = engine(model, tok, True, 16)
    a = eng.add_request("ROMEO:", SamplingParams(max_tokens=40, temperature=0))
    for _ in range(3):
        eng.step()
    eng.add_request("y" * 200, SamplingParams(max_tokens=2, temperature=0))
    for _ in range(12):  # the long prompt needs 13 chunks
        outs = eng.step()
        assert any(o.request_id == a for o in outs)
        assert sum(s.num_scheduled for s in eng.scheduler.running) <= 16


def test_budget_below_max_model_len_requires_chunking(model, tok):
    with pytest.raises(ValueError):
        engine(model, tok, False, 64)
    assert engine(model, tok, True, 64).config.max_num_batched_tokens == 64
    with pytest.raises(ValueError):
        engine(model, tok, True, 0)


def test_chunking_with_preemption_prefix_cache_and_forks(model, tok):
    """Every mechanism at once under a tiny budget and a tight pool."""
    system = "You are a Shakespearean assistant. Answer in verse.\n"
    prompts = [system + q for q in ("Who art thou?", "What news?", "Speak!")] + PROMPTS[:2]
    params = [
        SamplingParams(max_tokens=40, temperature=0.8, seed=i, n=1 + (i == 1))
        for i in range(len(prompts))
    ]
    roomy = engine(model, tok, False, 256, max_model_len=128)
    want = [[c.text for c in o] for o in roomy.generate(prompts, params)]
    tight = engine(model, tok, True, 12, num_blocks=40, block_size=4, max_model_len=128)
    got = [[c.text for c in o] for o in tight.generate(prompts, params)]
    assert got == want
    assert tight.scheduler.num_preemptions > 0
    assert tight.stats.cached_prompt_tokens > 0
    assert tight.bm.alloc.num_used == 0
