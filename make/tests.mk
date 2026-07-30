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
	@uv run --extra dev ruff check
	@uv run --extra dev ruff format --check
	@uv run --extra dev mypy src tests isaac/runtime/extensions isaac/runtime/standalone \
		make/scripts flight_analysis px4/scripts
	@uv run --extra dev vulture --config pyproject.toml
	@uv run --extra dev pytest
