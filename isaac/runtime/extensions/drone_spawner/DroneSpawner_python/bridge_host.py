"""PX4 bridge wiring owned by the drone-spawner host.

Per the contract, the scene-owning extension is the one that constructs the
``Px4Bridge`` and registers a physics-step callback. The bridge itself owns
no scene state. Keeping this wiring out of the bridge extension also keeps
the bridge extension single-responsibility.

The host is opt-in: when the experiment config has ``px4.enabled = false``
(the default), no bridge is built and no callbacks are registered. This lets
the bridge layers be tested in stages (port alone, articulation_io alone,
full bridge) by flipping the flag.
"""

from __future__ import annotations

from typing import Any

import carb
import omni.physx

from robotics_experiments.config import Px4BridgeConfig

from .global_variables import EXTENSION_TITLE

_LOG_PREFIX = f"{EXTENSION_TITLE}.BridgeHost"


class BridgeHost:
    """Lifecycle owner for a single ``Px4Bridge`` instance.

    Constructed once per Play session by the drone-spawner extension. Builds
    the bridge from the supplied config, registers a physics-step callback
    that ticks it, and tears everything down on stop.
    """

    def __init__(self, *, articulation_path: str, config: Px4BridgeConfig):
        self._cfg = config
        self._articulation_path = articulation_path
        self._bridge: Any | None = None
        self._physx_sub = None
        self._sim_time_us: int = 0
        self._physics_tick_count: int = 0
        self._start_attempted: bool = False

    @property
    def is_running(self) -> bool:
        return self._bridge is not None

    def start(self) -> None:
        """Construct the bridge and subscribe to physics steps.

        Idempotent: a second call while running is a no-op.
        """
        if self._bridge is not None:
            return
        if self._physx_sub is not None:
            return
        if not self._cfg.enabled:
            return

        self._sim_time_us = 0
        self._physics_tick_count = 0
        self._start_attempted = False
        physx = omni.physx.get_physx_interface()
        self._physx_sub = physx.subscribe_physics_step_events(self._on_physics_step)
        settle_time_s = max(0.0, float(getattr(self._cfg, "startup_settle_time_s", 0.0)))
        if settle_time_s > 0.0:
            carb.log_info(
                f"[{_LOG_PREFIX}] delaying PX4 bridge start for "
                f"startup_settle_time_s={settle_time_s:.3f}"
            )
            return

        self._start_bridge()

    def _start_bridge(self) -> None:
        """Construct the bridge after any configured startup settle delay."""
        if self._bridge is not None or self._start_attempted:
            return
        self._start_attempted = True

        # Late imports so a disabled bridge does not pay the import cost
        # (and does not require the bridge extension to be loaded).
        import omni.usd

        from robotics_experiments.px4_bridge import (
            PegasusQuadraticThrustModel,
            Px4Bridge,
            Px4MavlinkPort,
            discover_px4_rotor_mapping,
        )

        articulation_path = self._cfg.articulation_path or self._articulation_path
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            carb.log_error(
                f"[{_LOG_PREFIX}] cannot discover PX4 rotors without an open USD stage. "
                "Bridge not started."
            )
            return
        try:
            rotor_mapping = discover_px4_rotor_mapping(stage, articulation_path)
        except Exception as exc:
            carb.log_error(f"[{_LOG_PREFIX}] PX4 rotor discovery failed: {exc}")
            return

        rotor_paths = rotor_mapping.rotor_link_paths
        num_rotors = rotor_mapping.num_rotors

        port = Px4MavlinkPort(
            num_rotors=num_rotors,
            connection_type=self._cfg.connection_type,
            connection_ip=self._cfg.connection_ip,
            connection_baseport=self._cfg.connection_baseport,
            vehicle_id=self._cfg.vehicle_id,
            px4_autolaunch=self._cfg.px4_autolaunch,
            px4_dir=self._cfg.px4_dir,
            px4_vehicle_model=self._cfg.px4_vehicle_model,
            input_offset=self._cfg.input_offset,
            input_scaling=self._cfg.input_scaling,
            zero_position_armed=self._cfg.zero_position_armed,
            lockstep_timeout_s=self._cfg.lockstep_timeout_s,
            first_heartbeat_timeout_s=self._cfg.first_heartbeat_timeout_s,
            max_bootstrap_actuator_lag_s=self._cfg.max_bootstrap_actuator_lag_s,
            heartbeat_rate_hz=self._cfg.heartbeat_rate_hz,
            gps_enabled=self._cfg.gps_enabled,
            gps_send_rate_hz=self._cfg.gps_send_rate_hz,
            sensor_noise_enabled=self._cfg.sensor_noise_enabled,
            home_lat_deg=self._cfg.home_lat_deg,
            home_lon_deg=self._cfg.home_lon_deg,
            home_alt_m=self._cfg.home_alt_m,
            debug_log_path=self._cfg.debug_log_path,
            debug_log_level=self._cfg.debug_log_level,
            sensor_debug_log_interval=self._cfg.sensor_debug_log_interval,
        )
        thrust_model = PegasusQuadraticThrustModel(
            num_rotors=num_rotors,
            rotor_constant=self._cfg.rotor_constant,
            rolling_moment_coefficient=self._cfg.rolling_moment_coefficient,
            rot_dir=rotor_mapping.rot_dir,
            min_rotor_velocity=self._cfg.min_rotor_velocity,
            max_rotor_velocity=self._cfg.max_rotor_velocity,
            linear_drag=self._cfg.linear_drag,
        )
        endpoint = (
            f"{self._cfg.connection_type}:{self._cfg.connection_ip}:"
            f"{self._cfg.connection_baseport + self._cfg.vehicle_id}"
        )
        carb.log_info(
            f"[{_LOG_PREFIX}] starting PX4 bridge endpoint={endpoint} "
            f"articulation={articulation_path} rotor_paths={rotor_paths} "
            f"rot_dir={rotor_mapping.rot_dir} "
            f"lockstep_timeout_s={self._cfg.lockstep_timeout_s} "
            f"first_heartbeat_timeout_s={self._cfg.first_heartbeat_timeout_s} "
            f"max_bootstrap_actuator_lag_s={self._cfg.max_bootstrap_actuator_lag_s} "
            f"gps_enabled={self._cfg.gps_enabled} gps_rate_hz={self._cfg.gps_send_rate_hz} "
            f"sensor_noise_enabled={self._cfg.sensor_noise_enabled} "
            f"debug_log_path={self._cfg.debug_log_path}"
        )
        self._bridge = Px4Bridge(
            articulation_path=articulation_path,
            rotor_link_paths=rotor_paths,
            thrust_model=thrust_model,
            port=port,
            max_tilt_angle_deg=self._cfg.max_tilt_angle_deg,
        )

        try:
            self._bridge.start()
        except Exception as exc:
            carb.log_error(f"[{_LOG_PREFIX}] bridge.start() failed: {exc}")
            self._bridge = None
            return

        carb.log_info(f"[{_LOG_PREFIX}] PX4 bridge started")

    def stop(self) -> None:
        """Unsubscribe and tear down the bridge."""
        self._physx_sub = None
        if self._bridge is not None:
            try:
                self._bridge.stop()
            except Exception as exc:
                carb.log_warn(f"[{_LOG_PREFIX}] bridge.stop() raised: {exc}")
            self._bridge = None

    def _on_physics_step(self, dt: float) -> None:
        self._sim_time_us += int(round(dt * 1_000_000))
        self._physics_tick_count += 1
        settle_time_s = max(0.0, float(getattr(self._cfg, "startup_settle_time_s", 0.0)))
        settle_time_us = int(round(settle_time_s * 1_000_000))
        if self._bridge is None:
            if self._sim_time_us >= settle_time_us:
                carb.log_info(
                    f"[{_LOG_PREFIX}] startup settle complete at "
                    f"tick={self._physics_tick_count} sim_time_us={self._sim_time_us}; "
                    f"starting PX4 bridge"
                )
                self._start_bridge()
                return
            elif self._physics_tick_count == 1 or self._physics_tick_count % 250 == 0:
                carb.log_info(
                    f"[{_LOG_PREFIX}] waiting for startup settle "
                    f"tick={self._physics_tick_count} dt={dt:.6f} "
                    f"sim_time_us={self._sim_time_us} target_us={settle_time_us}"
                )
            return

        bridge_sim_time_us = self._sim_time_us - settle_time_us
        if self._physics_tick_count == 1 or self._physics_tick_count % 250 == 0:
            carb.log_info(
                f"[{_LOG_PREFIX}] physics tick={self._physics_tick_count} "
                f"dt={dt:.6f} sim_time_us={bridge_sim_time_us} "
                f"connected={self._bridge.is_connected}"
            )
        try:
            self._bridge.on_physics_step(dt, bridge_sim_time_us)
        except Exception as exc:
            # One bad tick must not crash the simulator. Log and continue;
            # persistent failure will be visible via is_connected.
            carb.log_warn(
                f"[{_LOG_PREFIX}] on_physics_step error at tick={self._physics_tick_count} "
                f"sim_time_us={bridge_sim_time_us}: {exc}"
            )
