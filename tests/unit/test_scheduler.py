"""Scheduler policy rules, tested without a model: only block accounting matters."""

import numpy as np
import pytest

from pagellm.cache.paged import BlockManager
from pagellm.scheduling.scheduler import Scheduler, SchedulerConfig, SeqStatus, Sequence


def make(num_blocks=16, block_size=4, **cfg):
    defaults = dict(max_num_seqs=8, max_num_batched_tokens=64)
    defaults.update(cfg)
    return Scheduler(SchedulerConfig(**defaults), BlockManager(num_blocks, block_size, False))


def seq(i, n_tokens, arrival=None):
    return Sequence(
        i,
        f"r{i}",
        0,
        list(range(n_tokens)),
        n_tokens,
        np.random.default_rng(i),
        float(i if arrival is None else arrival),
    )


def run_step(s: Scheduler):
    """Pretend the forward pass ran: every scheduled sequence computed its tokens and
    sampled one more."""
    batch = s.schedule()
    for q in batch.seqs:
        q.num_computed = len(q.tokens)
        q.tokens.append(0)
    return batch


def test_rejects_unknown_policy():
    with pytest.raises(ValueError):
        SchedulerConfig(policy="lottery")


def test_admits_fcfs_under_token_budget():
    s = make(max_num_batched_tokens=20)
    for i, n in enumerate((8, 8, 8)):
        s.add(seq(i, n))
    b = s.schedule()
    assert [q.seq_id for q in b.seqs] == [0, 1] and b.num_tokens == 16
    assert all(q.status is SeqStatus.RUNNING for q in b.seqs)


def test_head_of_line_is_not_skipped():
    """A big request at the head blocks smaller ones behind it (no starvation)."""
    s = make(max_num_batched_tokens=20)
    s.add(seq(0, 4))
    s.add(seq(1, 30))  # does not fit this step's remaining budget
    s.add(seq(2, 2))  # would fit, but must not jump the queue
    b = s.schedule()
    assert [q.seq_id for q in b.seqs] == [0]
    assert [q.seq_id for q in s.waiting] == [1, 2]


def test_max_num_seqs_caps_the_batch():
    s = make(max_num_seqs=2)
    for i in range(4):
        s.add(seq(i, 3))
    assert len(s.schedule().seqs) == 2


def test_static_policy_waits_for_the_batch_to_drain():
    s = make(policy="static")
    s.add(seq(0, 3))
    run_step(s)
    s.add(seq(1, 3))
    b = run_step(s)
    assert [q.seq_id for q in b.seqs] == [0]  # newcomer waits
    s.finish(s.running[0], "length")
    assert [q.seq_id for q in s.schedule().seqs] == [1]


def test_preempts_youngest_and_skips_admission_that_step():
    # 4 blocks of 4 slots: two 8-token prompts use 2 blocks each and both
    # need a 3rd block on the next token -> the younger one is preempted.
    s = make(num_blocks=4, block_size=4)
    old, young = seq(0, 8), seq(1, 8)
    s.add(old)
    s.add(young)
    run_step(s)  # both prefilled (9 tokens now)
    s.add(seq(2, 2))  # a newcomer that would fit
    b = s.schedule()
    assert b.preempted == [young]
    assert [q.seq_id for q in b.seqs] == [0]
    assert young.status is SeqStatus.WAITING and young.num_computed == 0
    assert s.waiting[0] is young  # back at the front
    assert all(q.seq_id != 2 for q in b.seqs)  # no admission after preemption
    assert s.num_preemptions == 1 and young.num_preemptions == 1


def test_finish_frees_blocks():
    s = make()
    a = seq(0, 9)
    s.add(a)
    s.schedule()
    assert s.bm.alloc.num_used == 3
    s.finish(a, "stop")
    assert s.bm.alloc.num_used == 0 and not s.has_unfinished()
    assert a.finish_reason == "stop"
