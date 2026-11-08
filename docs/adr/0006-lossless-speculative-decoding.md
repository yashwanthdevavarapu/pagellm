# 0006. Speculative decoding on the processed distribution

- Status: Accepted
- Date: 2026-11-08

## Context

Speculative decoding lets a cheap draft model propose k tokens that the target model verifies in one forward pass.
The rejection-sampling acceptance rule makes the output distribution exactly the target's, but only if the distributions it compares are the ones that would actually be sampled from.

## Decision

Acceptance compares target and draft distributions after the full logits pipeline: repetition penalty, temperature, top-k and top-p.
A draft token `x` is accepted with probability `min(1, p(x) / q(x))`.
On the first rejection a token is drawn from the normalised residual `max(0, p - q)` and the step ends.
If all k are accepted, a bonus token is drawn from the target.
The target cache is rolled back by truncation after each verification.

## Consequences

- Output follows exactly the same distribution as non-speculative sampling with the same parameters, not only under greedy decoding. A statistical test compares the empirical output law with the target law.
- A shortcut such as "accept if the draft token is the target argmax" would bias sampling. A test demonstrates that bias as a guard.
- Speculation runs on a single sequence with contiguous caches, outside the batched engine. Integrating it with paged rollback is on the roadmap.
