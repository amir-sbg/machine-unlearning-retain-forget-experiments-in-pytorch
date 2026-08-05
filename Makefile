.PHONY: test run

test:
	python -m pytest -q

run:
	PYTHONPATH=src python -m unlearning_lab.experiment
