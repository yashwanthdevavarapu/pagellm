# Architecture

This document shows how pagellm is put together and how a request flows through it. Every diagram is Mermaid, so GitHub renders it in place. The design decisions behind each part are recorded in the [ADRs](adr/README.md).

- [1. System overview](#1-system-overview)
- [2. Request lifecycle](#2-request-lifecycle)
- [3. One engine step](#3-one-engine-step)
- [4. Scheduler decisions](#4-scheduler-decisions)
- [5. Sequence states](#5-sequence-states)
- [6. KV block lifecycle](#6-kv-block-lifecycle)
- [7. Paged memory layout](#7-paged-memory-layout)
- [8. Speculative decoding](#8-speculative-decoding)
- [9. Package structure](#9-package-structure)

## 1. System overview

HTTP handler threads never touch the engine. They validate a request, hand it to the single engine thread through a queue, and wait on a per-request queue of output deltas ([ADR 0004](adr/0004-single-engine-thread.md)).

```mermaid
flowchart LR
    subgraph Clients
        C1["curl / OpenAI SDK"]
        C2["pagellm loadtest"]
        C3["Prometheus"]
    end

    subgraph Server["pagellm serve (one process)"]
        direction LR
        H["HTTP handler threads<br/>ThreadingHTTPServer"]
        P["protocol.py<br/>strict validation"]
        M["metrics.py<br/>/metrics, /health"]
        SQ[("submit queue")]
        DQ[("per-request<br/>delta queues")]

        subgraph EngineThread["engine thread (EngineLoop)"]
            direction TB
            S["Scheduler<br/>continuous batching<br/>chunked prefill<br/>preemption"]
            E["LLMEngine.step()"]
            T["Transformer.forward<br/>RMSNorm, RoPE, GQA, SwiGLU"]
            SM["Sampler<br/>per-sequence seeded RNG"]
        end

        subgraph Memory["KV memory"]
            BM["BlockManager<br/>block tables, COW,<br/>prefix cache (SHA-256, LRU)"]
            POOL[("PagedKVPool<br/>K, V blocks")]
        end
    end

    C1 -->|"POST /v1/completions"| H
    C2 -->|"SSE streams"| H
    C3 -->|"GET /metrics"| M
    H --> P --> SQ --> E
    E --> S
    S <--> BM
    E --> T
    T <-->|"write and attend<br/>through block tables"| POOL
    T --> SM --> DQ --> H
```

## 2. Request lifecycle

A streaming completion from the first byte to `[DONE]`. A blocking request follows the same path; the handler simply collects every delta before replying.

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant H as Handler thread
    participant V as protocol.py
    participant L as EngineLoop (engine thread)
    participant S as Scheduler
    participant E as LLMEngine
    participant B as BlockManager

    C->>H: POST /v1/completions {"stream": true}
    H->>V: parse_completion_request(body)
    alt invalid field
        V-->>H: ProtocolError
        H-->>C: 400 invalid_request_error
    else valid
        V-->>H: prompt + SamplingParams
        H->>L: submit(prompt, params) via queue
        L->>E: add_request()
        E->>S: enqueue sequence (WAITING)
        loop every engine step
            L->>E: step()
            E->>S: schedule()
            S->>B: allocate / can_append / prefix lookup
            B-->>S: block table, cached tokens
            S-->>E: batch (decodes first, then prefill chunks)
            E->>E: one forward pass over the flat batch
            E-->>L: StepOutput deltas
            L-->>H: delta on the request's queue
            H-->>C: data: {"choices":[{"text": ...}]}
        end
        E->>B: free blocks (hashed blocks stay cached)
        L-->>H: DONE
        H-->>C: data: [DONE]
    end
    Note over C,H: If the client disconnects, the handler aborts the<br/>request and its KV blocks are freed for others.
```

## 3. One engine step

One step is exactly one forward pass. Prompt chunks and single-token decodes from many sequences are concatenated into one flat batch, so they share every matmul.

```mermaid
flowchart TD
    A(["step()"]) --> B["Scheduler.schedule()"]
    B --> C{"anything<br/>scheduled?"}
    C -- no --> Z(["return []"])
    C -- yes --> D["apply copy-on-write<br/>block copies"]
    D --> F["build flat batch:<br/>tokens, positions, slot mapping<br/>(num_scheduled tokens per sequence)"]
    F --> G["Transformer.forward<br/>PagedBatch writes K, V to slots"]
    G --> H["attention:<br/>batched for all decode rows,<br/>per sequence for prefill chunks"]
    H --> I["logits only for the last row<br/>of each sequence"]
    I --> J["for each sequence:<br/>num_computed += num_scheduled<br/>publish full blocks to prefix cache"]
    J --> K{"prompt fully<br/>computed?"}
    K -- "no (mid-prompt chunk)" --> L["no sample this step"]
    K -- yes --> M{"n > 1 and<br/>not forked yet?"}
    M -- yes --> N["fork children:<br/>share blocks, ref count + 1"]
    M -- no --> O["sample next token<br/>(own seeded RNG)"]
    N --> O
    O --> P{"stop string, max_tokens<br/>or max_model_len?"}
    P -- yes --> Q["finish: free blocks"]
    P -- no --> R["stream delta<br/>(hold back partial stop strings)"]
    Q --> R
    L --> S(["return StepOutputs"])
    R --> S
```

## 4. Scheduler decisions

Running sequences are served first, decodes before prefill chunks, oldest first. New requests are admitted strictly first come, first served, and never in a step that had to preempt ([ADR 0003](adr/0003-recompute-preemption.md), [ADR 0008](adr/0008-fcfs-admission.md), [ADR 0010](adr/0010-chunked-prefill.md)).

```mermaid
flowchart TD
    A(["schedule()"]) --> B["budget = max_num_batched_tokens"]
    B --> C["order running sequences:<br/>decodes first, then prefill chunks,<br/>oldest first within each"]
    C --> D{"next running<br/>sequence?"}
    D -- none left --> J
    D -- yes --> E["n = min(uncomputed, budget)<br/>(all-or-nothing when chunking is off)"]
    E --> F{"n > 0?"}
    F -- no --> D
    F -- yes --> G{"enough free blocks<br/>for its next tokens?"}
    G -- yes --> H["schedule n tokens<br/>budget -= n"]
    H --> D
    G -- no --> I["preempt the youngest running sequence:<br/>free all its blocks, back to queue front"]
    I --> G
    J{"preempted anything<br/>this step?"} -- yes --> X(["return batch<br/>(no admissions: avoids thrashing)"])
    J -- no --> K{"static policy and<br/>batch still running?"}
    K -- yes --> X
    K -- no --> L{"waiting queue non-empty,<br/>seats free, budget left?"}
    L -- no --> X
    L -- yes --> M["head of queue:<br/>look up prefix-cache hits,<br/>n = min(uncached tokens, budget)"]
    M --> N{"n > 0 and blocks<br/>for whole prompt?"}
    N -- "no (FCFS: nobody skips ahead)" --> X
    N -- yes --> O["allocate block table,<br/>reuse cached blocks,<br/>schedule n tokens"]
    O --> L
```

## 5. Sequence states

```mermaid
stateDiagram-v2
    [*] --> WAITING: add_request()
    WAITING --> RUNNING: admitted (blocks allocated,<br/>prefix-cache hits reused)
    RUNNING --> RUNNING: prefill chunk or decode token
    RUNNING --> WAITING: preempted (blocks freed,<br/>recomputed later)
    RUNNING --> FINISHED: stop string, max_tokens,<br/>or max_model_len
    WAITING --> FINISHED: aborted
    RUNNING --> FINISHED: aborted (client disconnect)
    FINISHED --> [*]: blocks released
```

## 6. KV block lifecycle

Blocks are reference counted. A freed block that carries a prefix hash is not wiped: it stays in an LRU set and can be revived by a later request with the same prefix ([ADR 0002](adr/0002-paged-kv-cache.md), [ADR 0005](adr/0005-prefix-cache-hashing.md)).

```mermaid
stateDiagram-v2
    [*] --> Free
    Free --> Owned: allocate() (ref = 1)
    Owned --> Shared: fork or prefix hit (ref + 1)
    Shared --> Owned: one owner releases (ref - 1)
    Shared --> Owned: copy-on-write gives the writer<br/>its own copy
    Owned --> Free: release, no hash (ref = 0)
    Owned --> Cached: release, block was full and hashed
    Cached --> Owned: prefix hit revives it
    Cached --> Free: evicted (LRU) when the pool runs dry
```

## 7. Paged memory layout

Each sequence sees a contiguous logical KV cache. Physically its tokens live in fixed-size blocks anywhere in the pool, found through its block table. Two requests that share a system prompt point at the same physical blocks.

```mermaid
flowchart LR
    subgraph SA["sequence A: block table [7, 2, 9]"]
        direction TB
        A0["logical block 0<br/>tokens 0-15"]
        A1["logical block 1<br/>tokens 16-31"]
        A2["logical block 2<br/>tokens 32-40"]
    end

    subgraph Pool["physical PagedKVPool (16 slots per block)"]
        direction TB
        P7["block 7<br/>ref count 2<br/>shared system prompt"]
        P2["block 2"]
        P9["block 9<br/>9 of 16 slots used"]
        P4["block 4<br/>5 of 16 slots used"]
    end

    subgraph SB["sequence B: block table [7, 4]"]
        direction TB
        B0["logical block 0<br/>tokens 0-15"]
        B1["logical block 1<br/>tokens 16-20"]
    end

    A0 --> P7
    A1 --> P2
    A2 --> P9
    B0 --> P7
    B1 --> P4
```

Waste is bounded by one partly filled block per sequence, instead of the whole unused tail that a contiguous `max_model_len` reservation holds.

## 8. Speculative decoding

A cheap draft proposes k tokens; the target scores all of them in one forward pass. The acceptance rule makes the output distribution exactly the target's ([ADR 0006](adr/0006-lossless-speculative-decoding.md)).

```mermaid
flowchart TD
    A(["context"]) --> B["draft model proposes k tokens<br/>x1..xk with distributions q1..qk"]
    B --> C["target model scores all k + 1 positions<br/>in ONE forward pass: p1..pk+1"]
    C --> D["i = 1"]
    D --> E{"accept xi with probability<br/>min(1, pi(xi) / qi(xi))"}
    E -- accepted --> F{"i = k?"}
    F -- no --> G["i = i + 1"] --> E
    F -- yes --> H["bonus token from pk+1"]
    E -- rejected --> I["emit a token from<br/>normalised max(0, pi - qi)"]
    H --> J["truncate both caches to the accepted prefix"]
    I --> J
    J --> K(["1 to k + 1 tokens per target forward"])
```

## 9. Package structure

Arrows point from a package to the packages it imports. The model knows nothing about caches or scheduling: it only talks to the `KVBackend` interface.

```mermaid
flowchart BT
    model["model<br/>transformer, tokenizer, weights"]
    cache["cache<br/>contiguous, paged"]
    scheduling["scheduling<br/>scheduler"]
    engine["engine<br/>llm_engine, sampling, speculative"]
    serving["serving<br/>protocol, server, metrics"]
    training["training<br/>trainer"]
    benchmarks["benchmarks<br/>runner, loadtest"]
    cli["cli"]

    cache --> model
    scheduling --> cache
    engine --> scheduling
    engine --> cache
    engine --> model
    serving --> engine
    training --> model
    benchmarks --> engine
    benchmarks --> cache
    cli --> serving
    cli --> engine
    cli --> benchmarks
    cli --> training
```

| Package | Responsibility | Key types |
|---|---|---|
| `model` | the network and its weights | `Transformer`, `ModelConfig`, `KVBackend`, `CharTokenizer` |
| `cache` | where K and V live, and how attention reads them | `PagedKVPool`, `BlockManager`, `BlockAllocator`, `PagedBatch`, `KVCache` |
| `scheduling` | which sequences run each step, and how many tokens each | `Scheduler`, `Sequence`, `ScheduledBatch` |
| `engine` | the step loop, sampling, speculative decoding | `LLMEngine`, `SamplingParams`, `SpeculativeDecoder` |
| `serving` | HTTP, SSE, validation, metrics | `Server`, `EngineLoop`, `parse_completion_request` |
| `training` | numpy training with a hand-derived backward pass | `train`, `loss_and_grads`, `AdamW` |
| `benchmarks` | every measured number in the README | `run_trace`, `run_load` |
