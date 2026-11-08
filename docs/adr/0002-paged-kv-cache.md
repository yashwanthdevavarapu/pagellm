# 0002. Paged KV cache with reference-counted blocks

- Status: Accepted
- Date: 2026-11-08

## Context

A contiguous KV cache must reserve `max_model_len` slots per sequence, because a contiguous buffer cannot grow in place.
Most requests stop far short of the maximum, so most reserved memory is never written.
Memory, not compute, then caps how many sequences can run at once.

## Decision

K and V live in a pool of fixed-size physical blocks (`PagedKVPool`, 16 tokens per block by default).
Each sequence owns a block table that maps its logical blocks to physical ones.
Blocks are reference counted, so forks (`n > 1`) and identical prompt prefixes share physical memory.
A shared block that must be written is copied first (copy-on-write).

The model never sees this layout. It hands each layer's q, k and v to a `KVBackend`.
The paged backend maps positions to slots (`slot_mapping`) and reads back through the block table.

## Consequences

- Waste is bounded by `block_size - 1` slots per sequence. See the memory benchmark in the README for measured numbers.
- Forking and prefix sharing are ref-count operations instead of copies.
- Attention reads through an indirection. Two read paths exist: `gather` (index, then dense attention; fastest in numpy) and `paged` (block-by-block online softmax; the shape of the GPU kernel). Both are tested against contiguous attention.
- Allocation can fail mid-generation, which requires a preemption policy (ADR 0003).
