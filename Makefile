.PHONY: setup train test lint mlflow-ui clean

setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

train:
	dvc repro

test:
	pytest tests/ -v

lint:
	ruff check src/

mlflow-ui:
	mlflow ui --port 5000

clean:
	rm -rf data/interim/* data/processed/* models/* reports/*
	dvc gc -w
