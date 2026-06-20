"""Standalone Isaac/PX4 negative takeoff stability scenario.

This test runs the same arm/takeoff command flow as the positive stability
test, then injects a hard one-rotor thrust fault plus a sustained body torque
after PX4 accepts takeoff. It passes only if the scenario reaches takeoff and
detects the resulting flight instability through the stability envelope.

Run through the host Make target so PX4 lifecycle and logs are owned outside
Isaac:
    make test-integration-px4-takeoff-instability
"""

from __future__ import annotations

import numpy as np
from _px4_takeoff_scenario import (
    ExpectedInstabilityDetected,
    Px4TakeoffScenarioOptions,
    run_px4_takeoff_scenario,
)
from _runtime import SceneTestContext, run_scene_test

from robotics_experiments.px4_bridge import ActuatorCommands

DEFAULT_SCENE = "@isaac/assets/TestScenes/MinimalIntegration.usda"
TEST_NAME = "px4_takeoff_instability_detection"
PX4_CONTAINER_LOG_NAME = "test_px4_takeoff_instability_px4.log"
FAULTED_ROTOR_INDEX = 0
FAULTED_ROTOR_COMMAND_SCALE = 0.0
FAULT_BODY_TORQUE_BIAS_N_M = np.array([4.0, 0.0, 0.0], dtype=np.float64)


class _RotorFailureFault:
    def __init__(
        self,
        base_model,
        *,
        rotor_index: int,
        command_scale: float,
        body_torque_bias_n_m: np.ndarray,
    ):
        self._base_model = base_model
        self._rotor_index = int(rotor_index)
        self._command_scale = float(command_scale)
        self._body_torque_bias_n_m = np.asarray(body_torque_bias_n_m, dtype=np.float64)
        if self._body_torque_bias_n_m.shape != (3,):
            raise ValueError(
                f"Body torque fault expects shape (3,), got {self._body_torque_bias_n_m.shape}"
            )

    @property
    def num_rotors(self) -> int:
        return self._base_model.num_rotors

    def _faulted_commands(self, commands: ActuatorCommands) -> ActuatorCommands:
        faulted = np.asarray(commands.controls, dtype=np.float64).copy()
        if faulted.shape != (self.num_rotors,):
            raise ValueError(f"Rotor fault expects shape ({self.num_rotors},), got {faulted.shape}")
        faulted[self._rotor_index] *= self._command_scale
        return ActuatorCommands(faulted)

    def commands_to_forces(self, commands):
        return self._base_model.commands_to_forces(self._faulted_commands(commands))

    def commands_to_dynamics(self, commands, *, linear_body_velocity_flu=None):
        dynamics = self._base_model.commands_to_dynamics(
            self._faulted_commands(commands),
            linear_body_velocity_flu=linear_body_velocity_flu,
        )
        return type(dynamics)(
            forces_n=dynamics.forces_n,
            body_torque_n_m=(
                np.asarray(dynamics.body_torque_n_m, dtype=np.float64) + self._body_torque_bias_n_m
            ),
            body_drag_n=dynamics.body_drag_n,
        )


def _inject_one_rotor_fault(context: SceneTestContext, host) -> None:
    bridge = getattr(host, "_bridge", None)
    if bridge is None:
        raise AssertionError("PX4 bridge was not constructed before fault injection")
    thrust_model = getattr(bridge, "_thrust_model", None)
    if thrust_model is None:
        raise AssertionError("PX4 bridge has no thrust model to fault")
    if FAULTED_ROTOR_INDEX >= thrust_model.num_rotors:
        raise AssertionError(
            f"Faulted rotor index {FAULTED_ROTOR_INDEX} is outside "
            f"thrust model size {thrust_model.num_rotors}"
        )

    bridge._thrust_model = _RotorFailureFault(
        thrust_model,
        rotor_index=FAULTED_ROTOR_INDEX,
        command_scale=FAULTED_ROTOR_COMMAND_SCALE,
        body_torque_bias_n_m=FAULT_BODY_TORQUE_BIAS_N_M,
    )
    context.report(
        f"{TEST_NAME}: injected rotor {FAULTED_ROTOR_INDEX} command scale fault "
        f"after accepted takeoff command: scale={FAULTED_ROTOR_COMMAND_SCALE:.2f} "
        f"body_torque_bias_n_m={FAULT_BODY_TORQUE_BIAS_N_M.tolist()}"
    )


def _run_test(context: SceneTestContext) -> None:
    try:
        run_px4_takeoff_scenario(
            context,
            Px4TakeoffScenarioOptions(
                test_name=TEST_NAME,
                px4_container_log_name=PX4_CONTAINER_LOG_NAME,
                post_takeoff_hook=_inject_one_rotor_fault,
            ),
        )
    except ExpectedInstabilityDetected as exc:
        # Only a stability-envelope breach counts as detection. A
        # Px4FailsafeDetected is intentionally NOT caught here: a PX4 failsafe is
        # a rerun-worthy infrastructure flake, not proof the envelope caught the
        # injected fault, so it must propagate as a failure instead of a pass.
        context.report(f"{TEST_NAME}: detected expected instability: {exc}")
        return
    raise AssertionError("Post-takeoff rotor fault did not trigger the takeoff stability envelope")


def main() -> None:
    run_scene_test(
        test_name=TEST_NAME,
        default_scene=DEFAULT_SCENE,
        test_body=_run_test,
        description=__doc__ or "",
    )


if __name__ == "__main__":
    main()
