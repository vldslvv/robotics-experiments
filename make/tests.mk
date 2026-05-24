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
check: test
