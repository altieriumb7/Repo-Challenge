.PHONY: install test smoke run-all ablations lint

install:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest -q

smoke:
	python -m mirror run-scenario --train-dir tests/fixtures/tiny_scenario/train --eval-dir tests/fixtures/tiny_scenario/eval --name tiny --output-dir outputs/smoke

run-all:
	python -m mirror run-all --root-dir data --output-root outputs

ablations:
	python scripts/run_ablations.py --train-dir tests/fixtures/tiny_scenario/train --eval-dir tests/fixtures/tiny_scenario/eval --scenario tiny

lint:
	ruff check .
