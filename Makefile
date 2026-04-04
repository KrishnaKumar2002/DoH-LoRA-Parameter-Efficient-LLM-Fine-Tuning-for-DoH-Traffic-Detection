.PHONY: help install test lint format clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install black isort flake8 mypy pytest

test: ## Run tests
	python -m pytest tests/ -v

lint: ## Run linting
	flake8 src/ --max-line-length=127
	mypy src/ --ignore-missing-imports

format: ## Format code
	black src/
	isort src/

clean: ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	rm -rf .pytest_cache
	rm -rf results/
	rm -rf logs/

run: ## Run the pipeline
	python -m src.doh_lora.main

setup: ## Setup development environment
	python -m venv venv
	@echo "Run 'source venv/bin/activate' (or venv\\Scripts\\activate on Windows) to activate the virtual environment"