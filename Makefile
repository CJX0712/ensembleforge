.PHONY: test lint format demo ci check-determinism

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

test:
	pytest -q -W ignore::UserWarning

demo:
	python examples/run_demo.py

check-determinism:
	python examples/run_demo.py --check-determinism

ci: lint test demo
