# 0005. Chained SHA-256 block hashes for prefix caching

- Status: Accepted
- Date: 2026-11-08

## Context

Prefix caching reuses KV blocks across requests that share a prompt prefix, such as a system prompt.
A block's K and V depend on every token before it (through attention history and the RoPE position), so the same 16 characters at position 0 and at position 200 are different cache contents.
A wrong cache hit does not crash. It silently produces wrong text.

## Decision

The key of a full block is `sha256(parent_key || block_tokens)`, which chains in the hash of every earlier block.
Only full blocks are published.
The last prompt token is always recomputed because its logits are needed to sample. As a result, cache hits are always full, read-only blocks, and prefix sharing never needs copy-on-write.
Freed blocks that carry a hash move to an LRU evictable set instead of the free list.
A sequence's blocks are released tail first, so eviction removes the end of a cached prefix before its head; evicting the head first would orphan the rest.

## Consequences

- A false hit would require a SHA-256 collision. Python's 64-bit `hash()` was rejected for that reason.
- Greedy outputs are byte-identical with and without the cache; the prefix benchmark checks this on every run.
- Hashing costs one SHA-256 per full block per sequence, which is negligible next to a forward pass.
