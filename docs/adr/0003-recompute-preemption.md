# 0003. Recompute preemption instead of swap

- Status: Accepted
- Date: 2026-11-08

## Context

With paged allocation the pool can run dry while sequences are mid-generation.
Some running sequence must give up its blocks so that older ones can continue.
There are two options: swap the victim's blocks to a CPU tier and copy them back later, or drop them and recompute them later with a prefill.

## Decision

Preempt by recompute. The youngest running sequence frees all its blocks, its `num_computed` resets to zero, and it goes back to the front of the waiting queue.
Its prompt plus the tokens it already generated are prefilled again when it is readmitted.
No new requests are admitted in a step that preempted, because admitting work into a pool that just ran dry only causes the next preemption (thrashing).

## Consequences

- There is no second memory tier and no transfer bookkeeping.
- Every sequence carries its own seeded RNG, so a preempted sequence produces exactly the tokens it would have produced anyway. A test checks this with sampled decoding, not only greedy.
- Without chunked prefill a re-prefill must fit in one step, so `max_num_batched_tokens >= max_model_len` is enforced when the engine is built. ADR 0010 lifts this rule when chunked prefill is on.
- For very long sequences recompute costs more than swap would. A cost model that picks between them is on the roadmap.
