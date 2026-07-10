########################################
# Host tests
########################################

.PHONY: require-venv
require-venv:
	@test -f "$(VENV_ACTIVATE)" || { \
		echo "Missing virtual environment at $(VENV_DIR). Create it before running tests."; \
		exit 1; \
	}

.PHONY: test
test: require-venv
	@. "$(VENV_ACTIVATE)" && python -m pytest

.PHONY: check
check: require-venv
	@uv lock --check
	@uv run ruff check
	@uv run ruff format --check
	@uv run mypy src tests
	@uv run --extra dev vulture --config pyproject.toml
	@uv run pytest
