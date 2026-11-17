# Single entry point for the project. Run `make` for an interactive menu or `make help`.
UV := uv run
SOURCES := src tests
WEIGHTS_DIR ?= genre_classifier/models

.DEFAULT_GOAL := menu
.PHONY: menu help install run test lint format format-check typecheck docstrings check preprocess train-svm train-cnn train-transformer import-weights

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

train-svm: ## Train the SVM classifier
	$(UV) python -m music_genre_classification.training --model svm

train-cnn: ## Train the CNN classifier
	$(UV) python -m music_genre_classification.training --model cnn

train-transformer: ## Train the Transformer classifier
	$(UV) python -m music_genre_classification.training --model transformer

import-weights: ## Copy the trained CNN and Transformer weights of the original project (WEIGHTS_DIR)
	mkdir -p models
	cp $(WEIGHTS_DIR)/cnn.pth models/cnn.pt
	cp $(WEIGHTS_DIR)/transformer_model.pth models/transformer.pt
