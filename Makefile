.PHONY: test run

test:
	python -m pytest -q

run:
	python -m unlearning_lab.experiment
