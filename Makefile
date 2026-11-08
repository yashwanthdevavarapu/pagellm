PY ?= python

.PHONY: install lint format typecheck test test-unit test-e2e cov check bench bench-quick figures docker clean

install:            ## editable install with dev and docs extras
	$(PY) -m pip install -e ".[dev,docs]"

lint:               ## ruff lint + format check
	ruff check src tests examples scripts
	ruff format --check src tests examples scripts

format:             ## apply formatting and safe lint fixes
	ruff format src tests examples scripts
	ruff check --fix src tests examples scripts

typecheck:
	mypy

test:               ## every test tier
	pytest

test-unit:
	pytest tests/unit

test-e2e:
	pytest tests/e2e -v

cov:
	pytest --cov --cov-report=term --cov-report=html

check: lint typecheck cov   ## what CI runs

bench:              ## full benchmarks, results and charts under docs/
	pagellm bench all --json docs/results/benchmarks.json | tee docs/results/benchmarks.md
	$(PY) scripts/make_figures.py

bench-quick:
	pagellm bench all --quick

figures:
	$(PY) scripts/make_figures.py

docker:
	docker build -t pagellm:latest .

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
