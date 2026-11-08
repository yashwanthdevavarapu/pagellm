
## Static vs continuous batching  (128 requests, Poisson 400.0/s, max 16 concurrent, greedy)

| policy | throughput (tok/s) | mean latency (s) | p99 latency (s) | mean TTFT (s) | engine steps | mean tokens/step |
|---|---|---|---|---|---|---|
| static | 1322 | 2.53 | 4.63 | 2.40 | 1960 | 3.4 |
| continuous | 1601 | 2.16 | 3.81 | 1.68 | 511 | 13.1 |

continuous / static throughput: 1.21x; engine steps 3.84x fewer; mean latency static / continuous: 1.17x

## KV cache memory: contiguous reservation vs paged (block_size=16, max_model_len=256)

| scheme | slots holding real tokens | wasted (reserved, never used) | max concurrent seqs in a pool of 4096 slots |
|---|---|---|---|
| contiguous (reserve max_model_len) | 38.3% | 61.7% | 16 |
| paged | 92.9% | 7.1% | 61 (measured) |

KV bytes/token: 2048 (2 KV heads of 4: GQA halves this)

## Prefix caching (32 requests sharing a 192-token system prompt)

| | prompt tokens computed | served from cache | wall time (s) |
|---|---|---|---|
| no prefix cache | 6765 | 0 | 1.26 |
| prefix cache | 1005 | 5760 (85%) | 0.51 |

speedup 2.46x; greedy outputs identical: True

## Speculative decoding (target 0.76M params, 4 layers; draft 0.055M, 1 layer; 6 prompts x 160 tokens)

| sampling | k | acceptance rate | tokens / target forward | wall speedup |
|---|---|---|---|---|
| greedy | 1 | 0.69 | 1.68 | 1.24x |
| greedy | 2 | 0.46 | 1.92 | 1.30x |
| greedy | 4 | 0.32 | 2.25 | 1.20x |
| greedy | 6 | 0.25 | 2.44 | 0.79x |
| T=0.8 | 1 | 0.68 | 1.68 | 0.85x |
| T=0.8 | 2 | 0.64 | 2.26 | 1.47x |
| T=0.8 | 4 | 0.45 | 2.75 | 1.31x |
| T=0.8 | 6 | 0.37 | 3.19 | 0.94x |

measured forward cost: target 1 token 1158us, target 5 tokens 1808us, draft 1 token 444us (draft/target = 0.38)

## Decode cost vs batch size (context ~128 tokens, best of 3 x 20 steps)

| batch | per-sequence attention (ms/token) | batched decode attention (ms/token) | speedup |
|---|---|---|---|
| 1 | 1.386 | 1.111 | 1.25x |
| 2 | 1.187 | 1.055 | 1.13x |
| 4 | 0.706 | 0.616 | 1.15x |
| 8 | 0.564 | 0.288 | 1.96x |
| 16 | 0.443 | 0.421 | 1.05x |
| 32 | 0.517 | 0.528 | 0.98x |

## Chunked prefill (96 requests, prompts 96 to 224 tokens, Poisson 200.0/s, max 16 concurrent)

| config | p99 step (ms) | ITL mean (ms) | ITL p99 (ms) | ITL max (ms) | mean TTFT (s) | throughput (tok/s) |
|---|---|---|---|---|---|---|
| no chunking (budget 256) | 46.5 | 15.4 | 47.0 | 58.0 | 1.63 | 958 |
| chunked, budget 64 | 22.7 | 12.6 | 22.8 | 25.3 | 1.73 | 986 |
| chunked, budget 128 | 27.9 | 13.2 | 28.0 | 53.8 | 1.40 | 1107 |

## HTTP load test, no chunking (64 streaming requests, 16 concurrent)

succeeded 64, failed 0, wall 3.12 s, 20.52 req/s, 1329 output tok/s

| metric (ms) | p50 | p95 | p99 |
|---|---|---|---|
| time to first token | 37.9 | 159.4 | 161.4 |
| inter-token latency | 10.1 | 20.4 | 24.2 |
| end-to-end latency | 765.6 | 1201.9 | 1271.0 |

## HTTP load test, chunked, budget 64 (64 streaming requests, 16 concurrent)

succeeded 64, failed 0, wall 2.87 s, 22.29 req/s, 1443 output tok/s

| metric (ms) | p50 | p95 | p99 |
|---|---|---|---|
| time to first token | 47.2 | 197.7 | 226.3 |
| inter-token latency | 9.4 | 14.5 | 16.9 |
| end-to-end latency | 681.2 | 1035.7 | 1133.8 |
