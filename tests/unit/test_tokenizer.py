import numpy as np
import pytest

from pagellm.model.tokenizer import CharTokenizer


def test_roundtrip(tok):
    text = "ROMEO:\nBut soft, what light through yonder window breaks?"
    assert tok.decode(tok.encode(text)) == text
    np.testing.assert_array_equal(tok.encode_np(text), tok.encode(text))


def test_vocab_is_sorted_and_unique():
    t = CharTokenizer.from_text("banana")
    assert t.chars == "abn" and t.vocab_size == 3


def test_unknown_characters_degrade_to_space():
    t = CharTokenizer("ab ")
    assert t.decode(t.encode("a\N{GRINNING FACE}b")) == "a b"


def test_duplicate_vocabulary_rejected():
    with pytest.raises(ValueError):
        CharTokenizer("aa")


def test_blas_threads_pinned_on_import_but_user_choice_wins():
    import os
    import subprocess
    import sys

    code = "import pagellm, os; print(os.environ['OPENBLAS_NUM_THREADS'])"
    env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_NUM_THREADS"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.stdout.strip() == "1"
    env["OPENBLAS_NUM_THREADS"] = "3"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.stdout.strip() == "3"


def test_workload_is_deterministic():
    from pagellm.benchmarks.loadtest import make_workload

    assert make_workload(5, seed=2) == make_workload(5, seed=2)
    assert all(b["stream"] and b["max_tokens"] >= 32 for b in make_workload(5))
