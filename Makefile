.PHONY: help install install-dev test lint format clean run build

help: ## Show this help message
	@echo "Crop Yield Prediction App - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"

test: ## Run tests
	pytest tests/ -v --cov=. --cov-report=html

test-fast: ## Run tests without coverage
	pytest tests/ -v

lint: ## Run linting checks
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format: ## Format code with black
	black .

format-check: ## Check code formatting
	black --check --diff .

type-check: ## Run type checking with mypy
	mypy . --ignore-missing-imports

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +

run: ## Run the Streamlit application
	streamlit run app.py

build: ## Build the package
	python setup.py sdist bdist_wheel

check-all: format-check lint type-check test ## Run all checks

dev-setup: install-dev ## Set up development environment
	pre-commit install

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

docker-build: ## Build Docker image
	docker build -t crop-yield-prediction .

docker-run: ## Run Docker container
	docker run -p 8501:8501 crop-yield-prediction

venv: ## Create virtual environment
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  Windows: venv\\Scripts\\activate"
	@echo "  Unix/MacOS: source venv/bin/activate"

requirements-update: ## Update requirements.txt with current environment
	pip freeze > requirements.txt

requirements-dev: ## Generate requirements-dev.txt
	pip freeze | grep -E "(pytest|black|flake8|mypy|pre-commit)" > requirements-dev.txt
