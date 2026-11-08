# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,docs]"
pre-commit install                 # optional: ruff on every commit
```

## The bar for a change

1. `make check` passes: ruff lint, ruff format, mypy, and the full test suite with coverage of at least 90%.
2. New behaviour has a test. A bug fix has a regression test that fails before the fix.
3. Anything that touches the engine, cache, scheduler or model includes a before-and-after benchmark run (`pagellm bench <name>`).
4. A change to a design decision adds or supersedes an ADR in `docs/adr/` (start from `template.md`).

## Where tests go

| Directory | Use it for |
|---|---|
| `tests/unit` | one module, no threads, no network, fast |
| `tests/integration` | several modules together, the in-process server, CLI entry points |
| `tests/e2e` | the installed CLI in a subprocess, talking HTTP only |

Correctness tests compare against an independent reference (uncached decoding, textbook attention, finite differences) rather than against recorded outputs.

## Refreshing results and images

```bash
make bench                              # benchmarks -> docs/results, charts -> docs/assets
python scripts/make_screenshots.py      # terminal screenshots -> docs/assets
```

Run benchmarks on an idle machine on mains power, and update the README tables from `docs/results/benchmarks.md`.
