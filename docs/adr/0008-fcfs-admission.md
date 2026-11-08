# 0008. FCFS admission without skip-ahead

- Status: Accepted
- Date: 2026-11-08

## Context

When the request at the head of the waiting queue does not fit the remaining token budget or the free blocks, a scheduler can either stop admitting for this step or skip ahead to smaller requests that do fit.

## Decision

Admit strictly first come, first served. If the head does not fit, nothing behind it is admitted in that step.
Running sequences are scheduled oldest first, and the youngest is the preemption victim.

## Consequences

- Long prompts cannot be starved by a stream of short ones, and queueing delay is predictable.
- Throughput is slightly lower than with skip-ahead under mixed prompt lengths.
- Priority classes and SLO-aware admission would replace this rule; both are on the roadmap.
