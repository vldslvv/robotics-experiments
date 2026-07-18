########################################
# Isaac core performance regression benchmark
########################################

BENCHMARK_ROOT ?= $(CURDIR)/.build/benchmarks
ISAAC_CORE_BENCHMARK_BASELINE = $(CURDIR)/benchmarks/isaac_core/baseline.json
ISAAC_CORE_BENCHMARK_REPORT = make/scripts/isaac_core_benchmark_report.py
ISAAC_CORE_BENCHMARK_RUNNER = make/scripts/run_isaac_core_benchmark.sh

.PHONY: benchmark-isaac-core
benchmark-isaac-core:
	@BENCHMARK_BASELINE='$(ISAAC_CORE_BENCHMARK_BASELINE)' \
	    BENCHMARK_ROOT='$(BENCHMARK_ROOT)' \
	    ISAAC_IMAGE='$(ISAAC_IMAGE)' \
	    MAKE_COMMAND='make' \
	    REPORT_SCRIPT='$(ISAAC_CORE_BENCHMARK_REPORT)' \
	    bash '$(ISAAC_CORE_BENCHMARK_RUNNER)'

.PHONY: benchmark-isaac-core-accept
benchmark-isaac-core-accept:
	@test -n '$(RESULT)' || { \
	    echo 'Usage: make benchmark-isaac-core-accept RESULT=.build/benchmarks/isaac-core.<run-id>/report.json' >&2; \
	    exit 2; \
	}
	@uv run python '$(ISAAC_CORE_BENCHMARK_REPORT)' accept \
	    --result '$(RESULT)' \
	    --baseline '$(ISAAC_CORE_BENCHMARK_BASELINE)'
