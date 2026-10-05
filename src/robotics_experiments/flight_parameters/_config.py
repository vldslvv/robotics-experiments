"""Physical flight-suite definitions, independent of alignment and training."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, replace
from pathlib import Path

from robotics_experiments.config import load_config
from robotics_experiments.flight_parameters._models import (
    FlightCommand,
    FlightParametersConfig,
    FlightStage,
    HoverConfig,
)

__all__: list[str] = []


def flight_seeds(*, config: FlightParametersConfig) -> tuple[int, ...]:
    """Assign one random stream per independent complete flight."""
    return tuple(config.start_seed + index for index in range(config.flight_count))


def load_flight_parameters_config(
    *, path: str | Path, flight_count: int | None = None
) -> FlightParametersConfig:
    """Resolve physical stages from the same experiment used by the simulator."""
    experiment = load_config(path)
    authored = experiment.flight_parameters
    if authored is None:
        raise ValueError("experiment requires flight_parameters settings for capture")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", authored.flight_parameters_id) is None:
        raise ValueError("flight_parameters_id must be a single identifier")
    if authored.schema_version != 1 or authored.source_backend != "px4-sitl":
        raise ValueError("flight capture requires schema 1 and PX4 SITL")
    if authored.state_source != "simulator-ground-truth":
        raise ValueError("flight capture requires simulator ground truth")
    stages = []
    commands: tuple[FlightCommand, ...]
    for item in authored.stages:
        if item["kind"] == "hover":
            if item["commands"] or item["control_mode"] != "auto-loiter":
                raise ValueError("native hover uses AUTO_LOITER and no streamed commands")
            hover = HoverConfig(
                **{
                    **item["hover"],
                    "target_position_enu_m": tuple(item["hover"]["target_position_enu_m"]),
                }
            )
            commands = ()
        elif item["kind"] == "controller-response":
            hover = None
            required = {"duration_s"} | (
                {"velocity_body_flu_mps", "yaw_rate_radps"}
                if item["control_mode"] == "velocity"
                else {"collective_thrust_normalized", "body_rate_flu_radps"}
            )
            if item.get("hover") is not None or any(
                set(command) != required for command in item["commands"]
            ):
                raise ValueError("response command fields must match their control mode")
            commands = tuple(
                FlightCommand(
                    duration_s=command["duration_s"],
                    velocity_body_flu_mps=tuple(
                        command.get("velocity_body_flu_mps", (0.0, 0.0, 0.0))
                    ),
                    yaw_rate_radps=command.get("yaw_rate_radps", 0.0),
                    collective_thrust_normalized=command.get("collective_thrust_normalized", 0.0),
                    body_rate_flu_radps=tuple(command.get("body_rate_flu_radps", (0.0, 0.0, 0.0))),
                )
                for command in item["commands"]
            )
            if item["control_mode"] not in {"velocity", "ctbr"} or not commands:
                raise ValueError("response stages need a control mode and commands")
            if any(command.duration_s <= 0 for command in commands):
                raise ValueError("command durations must be positive")
            for command in commands:
                vectors = (command.velocity_body_flu_mps, command.body_rate_flu_radps)
                values = (
                    *command.velocity_body_flu_mps,
                    *command.body_rate_flu_radps,
                    command.duration_s,
                    command.yaw_rate_radps,
                    command.collective_thrust_normalized,
                )
                if any(len(vector) != 3 for vector in vectors) or not all(
                    math.isfinite(v) for v in values
                ):
                    raise ValueError(
                        "flight commands require finite values and three-component vectors"
                    )
        else:
            raise ValueError(f"unsupported flight stage kind: {item['kind']}")
        stages.append(FlightStage(**{**item, "commands": commands, "hover": hover}))
    if (
        len(stages) < 2
        or stages[0].kind != "hover"
        or any(stage.hover is not None for stage in stages[1:])
    ):
        raise ValueError("a flight suite starts with one hover followed by response stages")
    if len({stage.name for stage in stages}) != len(stages):
        raise ValueError("flight stage names must be unique")
    count = authored.flight_count if flight_count is None else flight_count
    if type(count) is not int or count <= 0:
        raise ValueError("flight_count must be a positive integer")
    # Use the same portable path in host and container flight definitions.
    repo_root = Path(__file__).resolve().parents[3]
    config_path = (repo_root / path).resolve()
    experiment_path = (
        config_path.relative_to(repo_root) if config_path.is_relative_to(repo_root) else config_path
    )
    config = FlightParametersConfig(
        schema_version=authored.schema_version,
        flight_parameters_id=authored.flight_parameters_id,
        source_backend=authored.source_backend,
        state_source=authored.state_source,
        experiment_config=experiment_path.as_posix(),
        vehicle=authored.vehicle,
        start_seed=authored.start_seed,
        flight_count=count,
        takeoff_min_climb_m=authored.takeoff_min_climb_m,
        stages=tuple(stages),
        preparation=authored.preparation,
        execution=authored.execution,
        stability=authored.stability,
        config_fingerprint="",
        command_rate_hz=experiment.rates.command_rate_hz,
        sample_rate_hz=experiment.rates.sample_rate_hz,
    )
    if config.execution.worker_count <= 0:
        raise ValueError("worker_count must be positive")
    execution = config.execution
    if any(type(value) is not int or value < 0 for value in asdict(execution).values()):
        raise ValueError("flight execution settings must be nonnegative integers")
    if (
        execution.gzip_level > 9
        or min(execution.raw_flight_budget_bytes, execution.estimated_archive_bytes_per_flight) == 0
    ):
        raise ValueError("invalid compression level or flight storage budget")
    if any(
        value <= 0 or not math.isfinite(value)
        for key, value in asdict(config.preparation).items()
        if key != "pusher_deployed"
    ):
        raise ValueError("flight preparation values must be finite and positive")
    hover = config.hover
    if (
        min(
            hover.mode_transition_timeout_s,
            hover.mode_report_max_age_s,
            hover.settle_s,
            hover.measure_s,
            config.sample_rate_hz,
        )
        <= 0
    ):
        raise ValueError("hover schedule and sample rate must be positive")
    sample_count = hover.measure_s * config.sample_rate_hz
    if (
        not math.isclose(sample_count, round(sample_count), rel_tol=0.0, abs_tol=1e-9)
        or sample_count < 2
    ):
        raise ValueError(
            "hover measurement must contain an integer number of samples, at least two"
        )
    if (
        config.preparation.timeout_s
        < config.preparation.minimum_neutral_s + config.preparation.stable_hold_s
    ):
        raise ValueError("response preparation cannot complete before its readiness interval")
    definition = asdict(config)
    definition.pop("stability")
    encoded = json.dumps(
        definition, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return replace(config, config_fingerprint=hashlib.sha256(encoded).hexdigest())
