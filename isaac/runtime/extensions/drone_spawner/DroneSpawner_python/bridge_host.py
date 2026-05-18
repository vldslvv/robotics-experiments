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

from typing import Any, List

import carb
import omni.physx

from robotics_experiments.config import Px4BridgeConfig


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

    @property
    def is_running(self) -> bool:
        return self._bridge is not None

    def start(self) -> None:
        """Construct the bridge and subscribe to physics steps.

        Idempotent: a second call while running is a no-op.
        """
        if self._bridge is not None:
            return
        if not self._cfg.enabled:
            return

        # Late imports so a disabled bridge does not pay the import cost
        # (and does not require the bridge extension to be loaded).
        from robotics_experiments.px4_bridge import (
            LinearThrustModel,
            Px4Bridge,
            Px4MavlinkPort,
        )

        rotor_paths = self._resolve_rotor_paths(self._cfg.rotor_link_paths)
        if len(rotor_paths) != self._cfg.num_rotors:
            carb.log_error(
                f"[Px4BridgeHost] rotor_link_paths has {len(rotor_paths)} entries; "
                f"expected num_rotors={self._cfg.num_rotors}. Bridge not started."
            )
            return

        port = Px4MavlinkPort(
            num_rotors=self._cfg.num_rotors,
            connection_type=self._cfg.connection_type,
            connection_ip=self._cfg.connection_ip,
            connection_baseport=self._cfg.connection_baseport,
            vehicle_id=self._cfg.vehicle_id,
            px4_autolaunch=self._cfg.px4_autolaunch,
            px4_dir=self._cfg.px4_dir,
            px4_vehicle_model=self._cfg.px4_vehicle_model,
            lockstep_timeout_s=self._cfg.lockstep_timeout_s,
            gps_send_rate_hz=self._cfg.gps_send_rate_hz,
            home_lat_deg=self._cfg.home_lat_deg,
            home_lon_deg=self._cfg.home_lon_deg,
            home_alt_m=self._cfg.home_alt_m,
        )
        thrust_model = LinearThrustModel(
            num_rotors=self._cfg.num_rotors,
            max_force_n=self._cfg.max_rotor_force_n,
        )
        articulation_path = self._cfg.articulation_path or self._articulation_path
        self._bridge = Px4Bridge(
            articulation_path=articulation_path,
            rotor_link_paths=rotor_paths,
            thrust_model=thrust_model,
            port=port,
        )

        try:
            self._bridge.start()
        except Exception as exc:
            carb.log_error(f"[Px4BridgeHost] bridge.start() failed: {exc}")
            self._bridge = None
            return

        self._sim_time_us = 0
        physx = omni.physx.get_physx_interface()
        self._physx_sub = physx.subscribe_physics_step_events(self._on_physics_step)
        carb.log_info("[Px4BridgeHost] PX4 bridge started")

    def stop(self) -> None:
        """Unsubscribe and tear down the bridge."""
        self._physx_sub = None
        if self._bridge is not None:
            try:
                self._bridge.stop()
            except Exception as exc:
                carb.log_warn(f"[Px4BridgeHost] bridge.stop() raised: {exc}")
            self._bridge = None

    def _on_physics_step(self, dt: float) -> None:
        if self._bridge is None:
            return
        self._sim_time_us += int(round(dt * 1_000_000))
        try:
            self._bridge.on_physics_step(dt, self._sim_time_us)
        except Exception as exc:
            # One bad tick must not crash the simulator. Log and continue;
            # persistent failure will be visible via is_connected.
            carb.log_warn(f"[Px4BridgeHost] on_physics_step error: {exc}")

    def _resolve_rotor_paths(self, configured: List[str]) -> List[str]:
        """Prefix relative rotor link paths with the articulation root."""
        root = self._cfg.articulation_path or self._articulation_path
        out: List[str] = []
        for p in configured:
            if not p:
                continue
            if p.startswith("/"):
                out.append(p)
            else:
                out.append(f"{root.rstrip('/')}/{p}")
        return out
