########################################
# Flight log analysis (host, pyulog/numpy via uv --no-project)
########################################

# pyulog/numpy are intentionally not project dependencies; run the analysis
# tools through a throwaway uv environment so they never touch .venv.
FLIGHT_ANALYSIS_DIR  = $(CURDIR)/flight_analysis/tools
FLIGHT_UV            = env UV_CACHE_DIR=/tmp/uv-cache uv run --no-project \
                           --with pyulog --with numpy python

# Optional flag passthrough for compare-logs. Override on the command line, e.g.
#   make compare-logs A=a.ulg B=b.ulg ALIGN=arm HOVER_LAST=3 LABELS=sim,real
ALIGN      ?= takeoff
HOVER_LAST ?= 2.0
WINDOW     ?=
LABELS     ?=
CSV        ?=

COMPARE_LOGS_FLAGS = --align $(ALIGN) --hover-last $(HOVER_LAST)
ifneq ($(strip $(WINDOW)),)
COMPARE_LOGS_FLAGS += --window $(WINDOW)
endif
ifneq ($(strip $(LABELS)),)
COMPARE_LOGS_FLAGS += --labels $(LABELS)
endif
ifneq ($(strip $(CSV)),)
COMPARE_LOGS_FLAGS += --csv $(CSV)
endif

# Compare two PX4 ULogs (symmetric, origin-agnostic). Requires A and B.
.PHONY: compare-logs
compare-logs:
	@test -n "$(A)" -a -n "$(B)" || { \
		echo "Usage: make compare-logs A=<log_a.ulg> B=<log_b.ulg> \\"; \
		echo "         [ALIGN=takeoff|arm|none] [HOVER_LAST=2.0] [WINDOW=START,END] \\"; \
		echo "         [LABELS=A,B] [CSV=out.csv]"; \
		exit 2; \
	}
	$(FLIGHT_UV) $(FLIGHT_ANALYSIS_DIR)/compare_logs.py "$(A)" "$(B)" $(COMPARE_LOGS_FLAGS)

# Print a focused single-log flight timeline. Requires ULOG.
.PHONY: ulog-timeline
ulog-timeline:
	@test -n "$(ULOG)" || { \
		echo "Usage: make ulog-timeline ULOG=<log.ulg> [SAMPLES=10,11,12]"; \
		exit 2; \
	}
	$(FLIGHT_UV) $(FLIGHT_ANALYSIS_DIR)/ulog_timeline.py "$(ULOG)" \
		$(if $(strip $(SAMPLES)),--samples $(SAMPLES),)
