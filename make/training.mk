########################################
# RL training (Isaac Lab image)
########################################

# Trains rl.training.task_id via isaac/runtime/standalone/train_rl.py
# and leaves the policy artifact under .build/training/<policy-name>/<run-id>/
# (updating the named .build/training/<policy-name>/latest symlink). Reuses the
# single-container integration
# runner on the Isaac Lab superset image; pass TRAIN_ARGS for overrides, e.g.
#   make train-rl TRAIN_ARGS='--max-iterations 5'   (smoke run)
TRAIN_RL_SCRIPT = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/train_rl.py
TRAIN_RL_LOG = /isaac-sim/.nvidia-omniverse/logs/train_rl.log
TRAIN_ARGS ?=
EVAL_PRESS_SCRIPT = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/eval_press.py
EVAL_PRESS_LOG = /isaac-sim/.nvidia-omniverse/logs/eval_press.log
EVAL_ARGS ?=

.PHONY: train-rl
train-rl: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
train-rl: ISAAC_TEST_EXTRA_ARGS = $(TRAIN_ARGS)
train-rl: generate-drone-usd
	@TEST_TARGET='train-rl' \
	    ISAAC_TEST_LOG='$(TRAIN_RL_LOG)' \
	    ISAAC_TEST_SCRIPT='$(TRAIN_RL_SCRIPT)' \
	    $(ISAAC_INTEGRATION_COMMON_ENV) \
	    bash "$(ISAAC_INTEGRATION_RUNNER)"

.PHONY: eval-press
eval-press: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
eval-press: HYDRA_CONFIG = configs/train_press_quad.yaml
eval-press: ISAAC_TEST_EXTRA_ARGS = $(EVAL_ARGS)
eval-press: generate-drone-usd
	@TEST_TARGET='eval-press' \
	    ISAAC_TEST_LOG='$(EVAL_PRESS_LOG)' \
	    ISAAC_TEST_SCRIPT='$(EVAL_PRESS_SCRIPT)' \
	    $(ISAAC_INTEGRATION_COMMON_ENV) \
	    bash "$(ISAAC_INTEGRATION_RUNNER)"
