########################################
# RL, Isaac Lab, and press workflows
########################################

TRAIN_RL_SCRIPT = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/train_rl.py
TRAIN_RL_LOG = /isaac-sim/.nvidia-omniverse/logs/train_rl.log
TRAIN_ARGS ?=
SELECT_RL_SCRIPT = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/select_rl_checkpoint.py
SELECT_RL_LOG = /isaac-sim/.nvidia-omniverse/logs/select_rl_checkpoint.log
SELECT_ARGS ?=
EVAL_PRESS_SCRIPT = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/eval_press.py
EVAL_PRESS_LOG = /isaac-sim/.nvidia-omniverse/logs/eval_press.log
EVAL_ARGS ?=
RL_TENSORBOARD_HOST ?= 127.0.0.1
RL_TENSORBOARD_PORT ?= 6006
RL_TENSORBOARD_CONTAINER_NAME ?= vv-rl-tensorboard
RL_TENSORBOARD_LOGDIR ?= .build/training
TEMPORAL_TRACE ?= .build/benchmarks/press-temporal/px4-traces.json
TEMPORAL_LAB_TRACE ?= .build/benchmarks/press-temporal/lab-traces.json
TEMPORAL_DELAY_GRID ?= 0.0,0.2,0.44
TEMPORAL_HORIZONTAL_KP_VEL_SCALE_GRID ?= 1.0
TEMPORAL_HORIZONTAL_KD_VEL_SCALE_GRID ?= 1.0

# Training owns optimization, stored metadata, and checkpoints only. A direct
# run is intentionally unpublished until select-rl-checkpoint selects it.
.PHONY: train-rl
train-rl: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
train-rl: ISAAC_TEST_EXTRA_ARGS = $(TRAIN_ARGS)
train-rl: generate-drone-usd
	@TEST_TARGET='train-rl' \
	    ISAAC_TEST_LOG='$(TRAIN_RL_LOG)' \
	    ISAAC_TEST_SCRIPT='$(TRAIN_RL_SCRIPT)' \
	    $(ISAAC_INTEGRATION_COMMON_ENV) \
	    bash "$(ISAAC_INTEGRATION_RUNNER)"

# Selection evaluates the checkpoints from RL_RUN_ID, exports the winner, and
# is the sole automatic owner of the policy namespace's latest symlink.
.PHONY: select-rl-checkpoint
select-rl-checkpoint: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
select-rl-checkpoint: ISAAC_TEST_EXTRA_ARGS = $(SELECT_ARGS)
select-rl-checkpoint: generate-drone-usd
	@TEST_TARGET='select-rl-checkpoint' \
	    ISAAC_TEST_LOG='$(SELECT_RL_LOG)' \
	    ISAAC_TEST_SCRIPT='$(SELECT_RL_SCRIPT)' \
	    $(ISAAC_INTEGRATION_COMMON_ENV) \
	    bash "$(ISAAC_INTEGRATION_RUNNER)"

# Evaluation independently verifies the selected artifact. When RL_RUN_ID is
# supplied it evaluates that exact published run instead of resolving latest.
.PHONY: eval-press
eval-press: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
eval-press: HYDRA_CONFIG = configs/train_press_approach_quad.yaml
eval-press: ISAAC_TEST_EXTRA_ARGS = $(EVAL_ARGS)
eval-press: generate-drone-usd
	@TEST_TARGET='eval-press' \
	    ISAAC_TEST_LOG='$(EVAL_PRESS_LOG)' \
	    ISAAC_TEST_SCRIPT='$(EVAL_PRESS_SCRIPT)' \
	    $(ISAAC_INTEGRATION_COMMON_ENV) \
	    bash "$(ISAAC_INTEGRATION_RUNNER)"

# The training wrapper runs this foreground server in the existing Isaac Lab
# image and stops its named container on exit.
.PHONY: tensorboard-rl
tensorboard-rl:
	@docker run --rm \
	    --network=host \
	    --name '$(RL_TENSORBOARD_CONTAINER_NAME)' \
	    -v '$(CURDIR):$(ISAAC_PROJECT_DIR):ro' \
	    --entrypoint /isaac-sim/python.sh \
	    '$(ISAAC_LAB_IMAGE)' \
	    -m tensorboard.main \
	    --logdir '$(ISAAC_PROJECT_DIR)/$(RL_TENSORBOARD_LOGDIR)' \
	    --host '$(RL_TENSORBOARD_HOST)' \
	    --port '$(RL_TENSORBOARD_PORT)'

.PHONY: test-integration-press-hover-goto
test-integration-press-hover-goto: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
test-integration-press-hover-goto: generate-drone-usd
	@TEST_TARGET='test-integration-press-hover-goto' \
	    ISAAC_TEST_LOG='$(ISAAC_PRESS_HOVER_GOTO_TEST_LOG)' \
	    ISAAC_TEST_SCRIPT='$(ISAAC_PRESS_HOVER_GOTO_TEST)' \
	    $(ISAAC_INTEGRATION_COMMON_ENV) \
	    bash "$(ISAAC_INTEGRATION_RUNNER)"

.PHONY: test-integration-press-contact-chain
test-integration-press-contact-chain: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
test-integration-press-contact-chain: HYDRA_CONFIG = configs/train_press_approach_quad.yaml
test-integration-press-contact-chain: ISAAC_TEST_EXTRA_ARGS = --contact-chain --num-envs 1
test-integration-press-contact-chain: generate-drone-usd
	@TEST_TARGET='test-integration-press-contact-chain' \
	    ISAAC_TEST_LOG='$(ISAAC_PRESS_CONTACT_CHAIN_TEST_LOG)' \
	    ISAAC_TEST_SCRIPT='$(ISAAC_PRESS_HOVER_GOTO_TEST)' \
	    $(ISAAC_INTEGRATION_COMMON_ENV) \
	    bash "$(ISAAC_INTEGRATION_RUNNER)"

.PHONY: test-integration-press-policy
test-integration-press-policy: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
test-integration-press-policy: HYDRA_CONFIG = configs/policy_hover_quad.yaml
test-integration-press-policy: generate-drone-usd
	@TEST_TARGET='test-integration-press-policy' \
	    HOST_ISAAC_LOG_BASENAME='test_press_policy_hover.log' \
	    HOST_PX4_LOG_BASENAME='test_press_policy_hover_px4.log' \
	    ISAAC_TEST_LOG='$(ISAAC_PRESS_POLICY_TEST_LOG)' \
	    ISAAC_TEST_SCRIPT='$(ISAAC_PX4_TAKEOFF_TEST)' \
	    PX4_TEST_CONTAINER_NAME='$(PX4_TEST_CONTAINER_NAME).test-integration-press-policy' \
	    STRICT='1' \
	    $(PX4_INTEGRATION_COMMON_ENV) \
	    bash "$(PX4_INTEGRATION_RUNNER)"

.PHONY: test-integration-press-approach-routine
test-integration-press-approach-routine: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
test-integration-press-approach-routine: HYDRA_CONFIG = configs/policy_press_approach_quad.yaml
test-integration-press-approach-routine: SCENE = @config
test-integration-press-approach-routine: generate-drone-usd
	@TEST_TARGET='test-integration-press-approach-routine' \
	    HOST_ISAAC_LOG_BASENAME='test_px4_press_approach_routine.log' \
	    HOST_PX4_LOG_BASENAME='test_px4_press_approach_routine_px4.log' \
	    ISAAC_TEST_LOG='$(ISAAC_PRESS_APPROACH_ROUTINE_TEST_LOG)' \
	    ISAAC_TEST_SCRIPT='$(ISAAC_PRESS_APPROACH_ROUTINE_TEST)' \
	    PX4_TEST_CONTAINER_NAME='$(PX4_TEST_CONTAINER_NAME).test-integration-press-approach-routine' \
	    STRICT='1' \
	    $(PX4_INTEGRATION_COMMON_ENV) \
	    bash "$(PX4_INTEGRATION_RUNNER)"

# Wall-free production-actor run at three entry ranges. Structured
# ROUTINE_STAGE / PRESS_POLICY records in the isolated Isaac log are the raw
# input to controller_conformance.py's temporal report.
.PHONY: test-integration-press-temporal-conformance
test-integration-press-temporal-conformance: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
test-integration-press-temporal-conformance: HYDRA_CONFIG = configs/temporal_conformance_press_approach_quad.yaml
test-integration-press-temporal-conformance: SCENE = @config
test-integration-press-temporal-conformance: generate-drone-usd
	@TEST_TARGET='test-integration-press-temporal-conformance' \
	    HOST_ISAAC_LOG_BASENAME='test_px4_press_temporal_conformance.log' \
	    HOST_PX4_LOG_BASENAME='test_px4_press_temporal_conformance_px4.log' \
	    ISAAC_TEST_LOG='/isaac-sim/.nvidia-omniverse/logs/test_px4_press_temporal_conformance.log' \
	    ISAAC_TEST_SCRIPT='$(ISAAC_PRESS_APPROACH_ROUTINE_TEST)' \
	    PX4_TEST_CONTAINER_NAME='$(PX4_TEST_CONTAINER_NAME).test-integration-press-temporal-conformance' \
	    STRICT='1' \
	    $(PX4_INTEGRATION_COMMON_ENV) \
	    bash "$(PX4_INTEGRATION_RUNNER)"

.PHONY: test-integration-press-temporal-replay
test-integration-press-temporal-replay: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
test-integration-press-temporal-replay: HYDRA_CONFIG = configs/temporal_conformance_press_approach_quad.yaml
test-integration-press-temporal-replay: ISAAC_TEST_EXTRA_ARGS = --temporal-replay $(ISAAC_PROJECT_DIR)/$(TEMPORAL_TRACE) --temporal-output $(ISAAC_PROJECT_DIR)/$(TEMPORAL_LAB_TRACE) --temporal-delay-grid $(TEMPORAL_DELAY_GRID) --temporal-horizontal-kp-vel-scale-grid $(TEMPORAL_HORIZONTAL_KP_VEL_SCALE_GRID) --temporal-horizontal-kd-vel-scale-grid $(TEMPORAL_HORIZONTAL_KD_VEL_SCALE_GRID)
test-integration-press-temporal-replay: generate-drone-usd
	@TEST_TARGET='test-integration-press-temporal-replay' \
	    ISAAC_TEST_LOG='$(ISAAC_PRESS_HOVER_GOTO_TEST_LOG)' \
	    ISAAC_TEST_SCRIPT='$(ISAAC_PRESS_HOVER_GOTO_TEST)' \
	    $(ISAAC_INTEGRATION_COMMON_ENV) \
	    bash "$(ISAAC_INTEGRATION_RUNNER)"
