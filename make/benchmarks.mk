########################################
# Isaac core performance regression benchmark
########################################

BENCHMARK_ROOT ?= $(CURDIR)/.build/benchmarks
COMMANDING_BENCHMARK_BASELINE = $(CURDIR)/benchmarks/commanding/baseline.json
COMMANDING_BENCHMARK = make/scripts/commanding_benchmark.py
ISAAC_CORE_BENCHMARK_BASELINE = $(CURDIR)/benchmarks/isaac_core/baseline.json
ISAAC_CORE_BENCHMARK_REPORT = make/scripts/isaac_core_benchmark_report.py
ISAAC_CORE_BENCHMARK_RUNNER = make/scripts/run_isaac_core_benchmark.sh
FLIGHT_BENCHMARK_REPORT = make/scripts/flight_benchmark_report.py
FLIGHT_BENCHMARK_RUNNER = make/scripts/run_flight_benchmark.sh
FLIGHT_BENCHMARK_COMPARE_ROOT ?= $(abspath $(CURDIR)/../.build/benchmarks)
FLIGHT_BENCHMARK_BUILD_FRESH ?= 0
FLIGHT_BENCHMARK_MEASUREMENT_LEVEL ?= full

.PHONY: benchmark-commanding
benchmark-commanding:
	@PYTHONPATH='$(CURDIR)/src' uv run python '$(COMMANDING_BENCHMARK)' run \
	    --output-root '$(BENCHMARK_ROOT)' \
	    --baseline '$(COMMANDING_BENCHMARK_BASELINE)'

.PHONY: benchmark-commanding-accept
benchmark-commanding-accept:
	@test -n '$(RESULT)' || { \
	    echo 'Usage: make benchmark-commanding-accept RESULT=.build/benchmarks/commanding.<run-id>/report.json' >&2; \
	    exit 2; \
	}
	@PYTHONPATH='$(CURDIR)/src' uv run python '$(COMMANDING_BENCHMARK)' accept \
	    --result '$(RESULT)' \
	    --baseline '$(COMMANDING_BENCHMARK_BASELINE)'

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

.PHONY: benchmark-flight
benchmark-flight: HYDRA_CONFIG = configs/flight_benchmark.yaml
benchmark-flight: xhost
	@BENCHMARK_ROOT='$(BENCHMARK_ROOT)' \
	    FLIGHT_BENCHMARK_BUILD_FRESH='$(FLIGHT_BENCHMARK_BUILD_FRESH)' \
	    FLIGHT_BENCHMARK_MEASUREMENT_LEVEL='$(FLIGHT_BENCHMARK_MEASUREMENT_LEVEL)' \
	    HOST_PROJECT_DIR='$(CURDIR)' \
	    HYDRA_CONFIG_HOST_PATH='$(HYDRA_CONFIG_HOST_PATH)' \
	    ISAAC_DOCKER_ENV='$(ISAAC_DOCKER_ENV)' \
	    ISAAC_DOCKER_RUNTIME='$(ISAAC_DOCKER_RUNTIME)' \
	    ISAAC_DOCKER_VOLUMES='$(ISAAC_DOCKER_VOLUMES)' \
	    ISAAC_IMAGE='$(ISAAC_IMAGE)' \
	    ISAAC_INTERACTIVE_ENV='$(ISAAC_INTERACTIVE_ENV)' \
	    ISAAC_INTERACTIVE_VOLUMES='$(ISAAC_INTERACTIVE_VOLUMES)' \
	    MAKE_COMMAND='$(MAKE)' \
	    PX4_IMAGE='$(PX4_IMAGE)' \
	    PX4_REF='$(PX4_REF)' \
	    PX4_ROOTFS='$(PX4_ROOTFS)' \
	    REPORT_SCRIPT='$(FLIGHT_BENCHMARK_REPORT)' \
	    bash '$(FLIGHT_BENCHMARK_RUNNER)'

.PHONY: benchmark-flight-compare
benchmark-flight-compare:
	@test -n '$(REFERENCE)' -a -n '$(CANDIDATE)' || { \
	    echo 'Usage: make benchmark-flight-compare REFERENCE=<flight report/dir> CANDIDATE=<flight report/dir>' >&2; \
	    exit 2; \
	}
	@uv run python '$(FLIGHT_BENCHMARK_REPORT)' compare \
	    --reference '$(REFERENCE)' \
	    --candidate '$(CANDIDATE)' \
	    --output-root '$(FLIGHT_BENCHMARK_COMPARE_ROOT)'
