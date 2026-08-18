.DEFAULT_GOAL := help

include make/config.mk
include make/docker.mk
include make/images.mk
include make/containers.mk
include make/tests.mk
include make/integration-tests.mk
include make/alignment.mk
include make/rl.mk
include make/flight-parameters.mk
include make/benchmarks.mk
include make/flight-analysis.mk

.PHONY: help
help:
	@printf '%s\n' \
		'Robotics experiment targets:' \
		'' \
		'  make build                 Build all Docker images' \
		'  make build-isaac           Build the Isaac Sim image' \
		'  make build-ros             Build the ROS 2 image' \
		'  make build-px4             Build the PX4 SITL image' \
		'  make print-airframes       Print the generated PX4 airframe manifest (review)' \
		'' \
		'  make run-isaac             Start or attach the Isaac Sim container' \
		'  make preflight-policy-artifacts  Validate policy artifacts before launch' \
		'  make run-ros               Start or attach the ROS 2 container' \
		'  make run-px4               Start or attach the PX4 SITL container' \
		'  make run                   Start Isaac, ROS 2, and PX4 in terminals' \
		'  make run-all-autorun       Like run-all, auto-launching the Isaac runtime' \
		'  make generate-drone-usd    Regenerate all config-driven drone USD assets' \
		'' \
		'  make stop                  Stop persistent containers' \
		'  make remove-containers     Remove persistent containers' \
		'  make remove-images         Remove runnable image tags, keeping build cache anchors' \
		'  make remove-cache-images   Remove Docker build cache anchor images' \
		'  make destroy               Stop containers and remove containers/runnable image tags' \
		'' \
		'  make test                  Run host Python tests' \
		'  make check                 Run the complete non-mutating host quality gate' \
		'  make test-integration      Run container-backed integration tests' \
		'  make test-integration-all  Run all integration tests in parallel with a summary' \
		'  make test-integration-isaac Run the Isaac standalone integration test' \
		'  make test-integration-px4-takeoff Run the Isaac/PX4 takeoff stability test' \
		'  make test-integration-px4-navigation Run the PX4 waypoint navigation test' \
		'  make test-integration-px4-takeoff-instability Run the PX4 takeoff negative test' \
		'' \
		'Controller alignment targets:' \
		'' \
		'  make align-controllers     Capture PX4 probes, sweep Lab, fit, and publish' \
		'  make alignment-capture-reference Capture task-neutral PX4 velocity probes' \
		'  make alignment-sweep-target Replay all declarative Isaac Lab candidates' \
		'  make alignment-fit         Score candidates and update latest/best links' \
		'  make test-integration-alignment-target Validate the selected target envelope' \
		'  make measure-flight-envelope Capture and publish task-neutral hover parameters' \
		'' \
		'RL, Isaac Lab, and press workflow targets:' \
		'' \
		'  make train-rl              Train, select, and publish latest/best policy labels' \
		'  make eval-press            Independently evaluate the selected press artifact' \
		'  make test-integration-press-hover-goto Run the in-Lab hover/goto gate' \
		'  make test-integration-press-contact-chain Run the real switch contact/latch gate' \
		'  make test-integration-press-policy Run the trained hover-policy PX4 gate' \
		'  make test-integration-press-policy-transfer Run warehouse press transfer acceptance' \
		'  make test-integration-press-approach-routine Verify waypoint-to-actor invocation' \
		'  make measure-press-navigation-handoff Measure production waypoint entry noise' \
		'' \
		'Benchmark targets:' \
		'' \
		'  make benchmark-commanding    Run the isolated host commanding performance gate' \
		'  make benchmark-commanding-accept RESULT=... Accept a commanding baseline' \
		'  make benchmark-isaac-core  Run and gate the Isaac core performance benchmark' \
		'  make benchmark-isaac-core-accept RESULT=... Accept an Isaac core baseline' \
		'  make benchmark-flight      Run the diagnostic full-GUI two-drone benchmark' \
		'  make benchmark-flight-compare REFERENCE=... CANDIDATE=... Compare flight reports'
