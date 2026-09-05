.PHONY: install reproduce example smoke test lint security check

install:
	uv sync --locked --extra dev

reproduce:
	MPLCONFIGDIR=.matplotlib uv run --locked community-detection reproduce

example:
	MPLCONFIGDIR=.matplotlib uv run --locked community-detection analyze --edges examples/weighted_edges.csv --output-root artifacts/example --allow-raw-identifiers

smoke:
	MPLCONFIGDIR=.matplotlib uv run --locked community-detection smoke

test:
	MPLCONFIGDIR=.matplotlib uv run --locked pytest

lint:
	uv run --locked ruff check .
	uv run --locked ruff format --check .

security:
	uv run --locked python scripts/check_sensitive.py

check: lint test security
