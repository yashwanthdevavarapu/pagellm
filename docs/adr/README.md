# Architecture Decision Records

Each record captures one decision: the context, what was decided, and what it costs.
Records are immutable once accepted. A change of mind gets a new record that supersedes the old one.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-numpy-only-runtime.md) | Pure numpy runtime, no deep learning framework | Accepted |
| [0002](0002-paged-kv-cache.md) | Paged KV cache with reference-counted blocks | Accepted |
| [0003](0003-recompute-preemption.md) | Recompute preemption instead of swap | Accepted |
| [0004](0004-single-engine-thread.md) | One engine thread behind a queue | Accepted |
| [0005](0005-prefix-cache-hashing.md) | Chained SHA-256 block hashes for prefix caching | Accepted |
| [0006](0006-lossless-speculative-decoding.md) | Speculative decoding on the processed distribution | Accepted |
| [0007](0007-stdlib-http-server.md) | Standard library HTTP server with strict request validation | Accepted |
| [0008](0008-fcfs-admission.md) | FCFS admission without skip-ahead | Accepted |
| [0009](0009-batched-decode-attention.md) | Batched decode attention and float32 discipline | Accepted |
| [0010](0010-chunked-prefill.md) | Chunked prefill with decode-first scheduling | Accepted |
| [0011](0011-single-threaded-blas.md) | Single-threaded BLAS by default | Accepted |

New records start from [template.md](template.md).
