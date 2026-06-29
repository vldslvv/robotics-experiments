.DEFAULT_GOAL := help

include make/config.mk
include make/docker.mk
include make/images.mk
include make/containers.mk
include make/tests.mk
include make/integration-tests.mk
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
		'' \
		'  make run-isaac             Start or attach the Isaac Sim container' \
		'  make run-ros               Start or attach the ROS 2 container' \
		'  make run-px4               Start or attach the PX4 SITL container' \
		'  make run                   Start Isaac, ROS 2, and PX4 in terminals' \
		'  make run-all-autorun       Like run-all, auto-launching the Isaac runtime' \
		'' \
		'  make stop                  Stop persistent containers' \
		'  make remove-containers     Remove persistent containers' \
		'  make remove-images         Remove built images' \
		'  make destroy               Stop containers and remove containers/images' \
		'' \
		'  make test                  Run host Python tests' \
		'  make test-integration      Run container-backed integration tests' \
		'  make test-integration-all  Run all integration tests in parallel with a summary' \
		'  make test-integration-isaac Run the Isaac standalone integration test' \
		'  make test-integration-px4-takeoff Run the Isaac/PX4 takeoff stability test' \
		'  make test-integration-px4-navigation Run the PX4 waypoint navigation test' \
		'  make test-integration-px4-takeoff-instability Run the PX4 takeoff negative test'
