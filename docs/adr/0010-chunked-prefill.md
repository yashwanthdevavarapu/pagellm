# 0010. Chunked prefill with decode-first scheduling

- Status: Accepted
- Date: 2026-11-08

## Context

Without chunking, a prompt is prefilled in a single step. A 200-token prompt that arrives while 15 other requests are streaming turns that step into a large one, and every streaming request waits for it. Users see this as a stall in the middle of a response (a spike in inter-token latency).
It also forces `max_num_batched_tokens >= max_model_len`, so the per-step token budget cannot be used to bound step time.

## Decision

`EngineConfig.enable_chunked_prefill` (CLI: `pagellm serve --chunked-prefill --max-num-batched-tokens N`) splits uncomputed prompt tokens across steps.

- Each sequence carries `num_scheduled`: the tokens it computes this step. Without chunking that is all of its uncomputed tokens, so the old behaviour is unchanged.
- Running sequences that are decoding are scheduled **before** any prefill chunk. A long prompt can therefore never stall tokens that are already streaming.
- A prompt gets `min(remaining tokens, remaining budget)` per step. Admission still follows FCFS (ADR 0008): the head of the queue starts its first chunk before anything behind it.
- Logits are sampled only when a sequence's last chunk completes. Earlier chunks write K and V and publish full blocks to the prefix cache as they go.
- KV blocks for the whole prompt are allocated at admission, so a sequence never runs out of blocks halfway through its own prompt.
- With chunking on, any budget of at least 1 makes progress, and the `max_num_batched_tokens >= max_model_len` rule is lifted.

## Consequences

- Outputs are identical with and without chunking. Tests cover budgets from 1 to 48 tokens, greedy and sampled decoding, and preemption, prefix caching and `n > 1` forks all at once under a 12-token budget.
- A test proves a running request receives a token on every step while a 200-token prompt is prefilled beside it.
- `pagellm bench chunked` and `pagellm bench http` measure the effect on step time, inter-token latency and time to first token.
- Time to first token for a long prompt can grow, because its prefill is spread over several steps. That is the intended trade: smooth streaming for everyone in exchange for a slightly later start for the newcomer.
