# Makefile for Federated Mental Health Prediction

.PHONY: help install test lint format clean run-experiment docker-build docker-run

PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
FLAKE8 := flake8
ISORT := isort

# Help target
help:
	@echo "Federated Mental Health Prediction - Available commands:"
	@echo ""
	@echo "  make install        Install dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make test           Run tests"
	@echo "  make test-cov       Run tests with coverage"
	@echo "  make lint           Run linting"
	@echo "  make format         Format code"
	@echo "  make clean          Clean build artifacts"
	@echo "  make run            Run experiment with default config"
	@echo "  make run-quick      Run quick experiment (fewer rounds)"
	@echo "  make generate-data  Generate synthetic data only"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run in Docker container"
	@echo ""

# Installation
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 isort mypy

# Testing
test:
	$(PYTEST) tests/ -v

test-cov:
	$(PYTEST) tests/ -v --cov=. --cov-report=html --cov-report=term-missing

# Linting and formatting
lint:
	$(FLAKE8) . --max-line-length=100 --exclude=venv,__pycache__,.git
	$(ISORT) . --check-only --diff

format:
	$(BLACK) . --line-length=100 --exclude="venv|__pycache__"
	$(ISORT) .

typecheck:
	mypy . --ignore-missing-imports

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf build dist htmlcov .coverage 2>/dev/null || true

clean-experiments:
	rm -rf experiments/*/

# Running experiments
run:
	$(PYTHON) experiments/run_experiment.py --config config.yaml

run-quick:
	$(PYTHON) experiments/run_experiment.py \
		--n-clients 5 \
		--n-rounds 10 \
		--experiment-name quick_test

run-no-dp:
	$(PYTHON) experiments/run_experiment.py \
		--n-clients 10 \
		--n-rounds 50 \
		--no-dp \
		--experiment-name no_dp_baseline

run-high-privacy:
	$(PYTHON) experiments/run_experiment.py \
		--n-clients 10 \
		--n-rounds 100 \
		--epsilon 0.5 \
		--experiment-name high_privacy

generate-data:
	$(PYTHON) -c "from data.synthetic_generator import SyntheticMentalHealthData, SyntheticConfig; \
		config = SyntheticConfig(n_users=10000, n_days=30); \
		generator = SyntheticMentalHealthData(config); \
		df = generator.generate(); \
		generator.save('./data/synthetic'); \
		print(f'Generated {len(df)} samples')"

# Docker
docker-build:
	docker build -t federated-mental-health:latest .

docker-run:
	docker run -it --rm \
		-v $(PWD)/experiments:/app/experiments \
		-v $(PWD)/data:/app/data \
		federated-mental-health:latest

docker-run-gpu:
	docker run -it --rm --gpus all \
		-v $(PWD)/experiments:/app/experiments \
		-v $(PWD)/data:/app/data \
		federated-mental-health:latest

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Validation
validate:
	$(PYTHON) -m py_compile data/*.py
	$(PYTHON) -m py_compile models/*.py
	$(PYTHON) -m py_compile federated/*.py
	$(PYTHON) -m py_compile privacy/*.py
	$(PYTHON) -m py_compile evaluation/*.py
	@echo "All modules compiled successfully"

# Benchmarks
benchmark-aggregation:
	$(PYTHON) -c "from federated.aggregation import compare_aggregation_methods; \
		import torch; import numpy as np; \
		updates = [(i, {'w': torch.randn(1000)}, 100) for i in range(10)]; \
		results = compare_aggregation_methods(updates); \
		print('Aggregation methods tested:', list(results.keys()))"

benchmark-privacy:
	$(PYTHON) -c "from privacy.dp_accounting import RDPAccountant; \
		acc = RDPAccountant(); \
		for i in range(100): acc.add_mechanism(1.0, 0.01, 100); \
		eps, delta, order = acc.get_privacy_spent(); \
		print(f'After 10000 steps: ε={eps:.4f}, δ={delta:.2e}')"
