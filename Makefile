.PHONY: install reproduce example smoke test lint security check

install:
	python -m pip install -e ".[dev]"

reproduce:
	MPLCONFIGDIR=.matplotlib python -m community_detection.cli reproduce

example:
	MPLCONFIGDIR=.matplotlib python -m community_detection.cli analyze --edges examples/weighted_edges.csv --output-root artifacts/example

smoke:
	MPLCONFIGDIR=.matplotlib python -m community_detection.cli smoke

test:
	MPLCONFIGDIR=.matplotlib python -m pytest

lint:
	python -m ruff check .

security:
	python scripts/check_sensitive.py

check: lint test security
