# Single entry point for the project. Run `make` for an interactive menu or `make help`.
UV := uv run
SOURCES := src tests

.DEFAULT_GOAL := menu
.PHONY: menu help install run test lint format format-check typecheck docstrings check preprocess train

menu: ## Open the interactive menu
	@$(UV) python -m music_genre_classification.menu

help: ## List all targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install: ## Install dependencies and git hooks
	uv sync --locked
	$(UV) pre-commit install

run: ## Start the Streamlit app
	$(UV) music-genre-classification

test: ## Run the tests with coverage
	$(UV) pytest

lint: ## Lint with ruff
	$(UV) ruff check $(SOURCES)

format: ## Format the code with ruff
	$(UV) ruff format $(SOURCES)

format-check: ## Check formatting without changing files
	$(UV) ruff format --check $(SOURCES)

typecheck: ## Type-check with mypy (strict)
	$(UV) mypy $(SOURCES)

docstrings: ## Check docstring presence and consistency
	$(UV) pydoclint src
	$(UV) interrogate src

check: format-check lint typecheck docstrings test ## Run everything CI runs

preprocess: ## Extract MFCC features from the GTZAN dataset
	$(UV) python -m music_genre_classification.data_preprocessing

train: ## Train the baseline model
	$(UV) python -m music_genre_classification.training
