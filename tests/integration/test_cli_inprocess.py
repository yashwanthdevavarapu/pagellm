"""The CLI and the benchmark and training entry points, run in-process so they
count toward coverage. The e2e suite runs the same commands as subprocesses."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import needs_weights
from pagellm import cli
from pagellm.benchmarks import main as bench_main
from pagellm.model.transformer import Transformer
from pagellm.training import train


@needs_weights
def test_generate_command(capsys):
    cli.main(["generate", "ROMEO:", "--max-tokens", "30", "--temperature", "0"])
    out = capsys.readouterr().out
    assert out.startswith("ROMEO:") and len(out.strip()) >= 30


@needs_weights
def test_generate_parallel_and_speculative(capsys):
    cli.main(["generate", "A", "--max-tokens", "10", "-n", "2", "--seed", "4"])
    assert "--- sample 1 ---" in capsys.readouterr().out
    cli.main(["generate", "B", "--max-tokens", "10", "--temperature", "0", "--speculative", "3"])
    assert "acceptance" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.startswith("pagellm ")


@needs_weights
@pytest.mark.slow
def test_every_benchmark_quick_mode(tmp_path, capsys):
    out = tmp_path / "b.json"
    res = bench_main("all", str(out), quick=True)
    saved = json.loads(out.read_text())
    assert set(saved) >= {
        "environment",
        "batching",
        "memory",
        "prefix",
        "spec",
        "scaling",
        "chunked",
        "http",
    }
    assert res["prefix"]["outputs_identical"] is True
    assert res["memory"]["paged_waste_pct"] < res["memory"]["contiguous_waste_pct"]
    assert res["batching"]["continuous"]["steps"] < res["batching"]["static"]["steps"]
    assert all(r["tokens_per_target_forward"] >= 1 for r in res["spec"]["runs"])
    assert "## Prefix caching" in capsys.readouterr().out


def test_train_writes_a_loadable_model(tmp_path):
    path = tmp_path / "m.npz"
    stats = train("draft", steps=3, batch=2, seq=32, lr=3e-3, seed=0, out=path, log_every=1)
    assert stats["steps"] == 3 and np.isfinite(stats["val_loss"])
    model, meta = Transformer.load(path)
    assert model.cfg.max_seq_len == 32 and meta["train"]["preset"] == "draft"


def test_train_cli(tmp_path, capsys):
    path = tmp_path / "t.npz"
    cli.main(
        [
            "train",
            "--preset",
            "draft",
            "--steps",
            "2",
            "--batch",
            "2",
            "--seq",
            "16",
            "--out",
            str(path),
        ]
    )
    assert path.exists() and "saved" in capsys.readouterr().out
