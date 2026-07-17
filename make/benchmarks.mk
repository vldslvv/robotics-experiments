########################################
# Flight performance regression benchmark
########################################

BENCHMARK_ROOT ?= $(CURDIR)/.build/benchmarks
FLIGHT_BENCHMARK_BASELINE = $(CURDIR)/benchmarks/flight/baseline.json
FLIGHT_BENCHMARK_REPORT = make/scripts/flight_benchmark_report.py
FLIGHT_BENCHMARK_RUNNER = make/scripts/run_flight_benchmark.sh

.PHONY: benchmark-flight
benchmark-flight:
	@BENCHMARK_BASELINE='$(FLIGHT_BENCHMARK_BASELINE)' \
	    BENCHMARK_ROOT='$(BENCHMARK_ROOT)' \
	    ISAAC_IMAGE='$(ISAAC_IMAGE)' \
	    MAKE_COMMAND='make' \
	    PX4_REF='$(PX4_REF)' \
	    REPORT_SCRIPT='$(FLIGHT_BENCHMARK_REPORT)' \
	    bash '$(FLIGHT_BENCHMARK_RUNNER)'

.PHONY: benchmark-flight-accept
benchmark-flight-accept:
	@test -n '$(RESULT)' || { \
	    echo 'Usage: make benchmark-flight-accept RESULT=.build/benchmarks/flight.<run-id>/report.json' >&2; \
	    exit 2; \
	}
	@uv run python '$(FLIGHT_BENCHMARK_REPORT)' accept \
	    --result '$(RESULT)' \
	    --baseline '$(FLIGHT_BENCHMARK_BASELINE)'
