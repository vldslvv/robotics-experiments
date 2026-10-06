"""Read-only analysis of completed flight evidence, independent of capture and consumers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np

from robotics_experiments.config import FlightStabilityConfig
from robotics_experiments.flight_parameters._dataset import (
    FlightDataset,
    _hover_trace,
    flight_stage_views,
    load_flight_dataset,
)
from robotics_experiments.flight_parameters._derive import _trace_signal_values, summarize_signal
from robotics_experiments.flight_parameters._evidence import (
    open_flight_evidence,
)
from robotics_experiments.flight_parameters._models import (
    FlightTrace,
    SignalStatistics,
)

__all__: list[str] = []


@dataclass(frozen=True)
class _HoverSummary:
    identity: str
    flight_count: int
    sample_count: int
    signals: tuple[SignalStatistics, ...]
    worst_rms_flight: str
    worst_flight_horizontal_rmse_m: float
    peak_flight: str
    excursion_sample_fraction: float
    excursion_flight_count: int
    excursion_count: int
    excursion_rate_hz: float
    longest_excursion_s: float
    maximum_endpoint_displacement_m: float
    maximum_position_trend_mps: float


@dataclass(frozen=True)
class _Difference:
    identity: str
    field: str
    equal: bool
    reference_count: int
    current_count: int
    first_different_index: int | None
    maximum_absolute_difference: float | None
    rmse_difference: float | None
    first_reference_sim_time_us: int | None = None
    first_current_sim_time_us: int | None = None


@dataclass(frozen=True)
class _PopulationComparison:
    group: str
    signal: str
    reference_value: float
    current_value: float
    reference_flights: int
    current_flights: int
    relative_change: float | None
    confidence_interval: tuple[float, float] | None
    status: str


@dataclass(frozen=True)
class _ResponseSummary:
    profile: str
    signal: str
    flight_count: int
    sample_count: int
    statistics: SignalStatistics
    worst_flight_seed: int
    worst_flight_rms: float
    peak_flight_seed: int


@dataclass(frozen=True)
class _RunSummary:
    run_dir: str
    run_id: str
    artifact_sha256: str
    manifest_sha256: str | None
    evidence_sha256: tuple[tuple[str, str], ...]
    hover: tuple[_HoverSummary, ...]
    response: tuple[_ResponseSummary, ...]


@dataclass(frozen=True)
class _Comparison:
    reference_run_id: str
    seed_relationship: str
    repeatability_status: str
    differences: tuple[_Difference, ...]
    population: tuple[_PopulationComparison, ...]
    population_status: str
    response_population: tuple[_PopulationComparison, ...]
    response_status: str


@dataclass(frozen=True)
class StabilityReport:
    """Immutable descriptive report; it never authorizes consumer use or a baseline."""

    artifact_kind: str
    schema_version: int
    settings: FlightStabilityConfig
    settings_sha256: str
    runs: tuple[_RunSummary, ...]
    comparisons: tuple[_Comparison, ...]
    status: str


def _validate_settings(settings: FlightStabilityConfig, /) -> None:
    payload = asdict(settings)
    for key, value in payload.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value):
            raise ValueError(f"stability {key} must be finite numeric")
        if key in {"bootstrap_repetitions", "bootstrap_seed", "minimum_flights"}:
            if not isinstance(value, int) or value < (0 if key == "bootstrap_seed" else 2):
                raise ValueError(f"stability {key} must be an integer in range")
        elif value <= 0:
            raise ValueError(f"stability {key} must be positive")
    if not 0 < payload["confidence"] < 1 or not 0 < payload["relative_change_limit"] < 1:
        raise ValueError("stability confidence and relative change limit must be between 0 and 1")


def _digest(path: Path, /) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identity(trace: FlightTrace, /) -> str:
    return f"flight-{trace.trial_index:03d}/seed-{trace.seed}"


def _horizontal(trace: FlightTrace, /) -> np.ndarray:
    return np.linalg.norm(
        (np.asarray(trace.position_enu_m) - trace.target_position_enu_m)[:, :2], axis=1
    )


def _hover_summary(
    *, identity: str, traces: tuple[FlightTrace, ...], settings: FlightStabilityConfig
) -> _HoverSummary:
    by_signal: dict[tuple[str, str], list[np.ndarray]] = {}
    excursion_samples = excursion_flights = excursion_count = 0
    longest = displacement = trend = total_duration = 0.0
    rms: list[float] = []
    peaks: list[float] = []
    for trace in traces:
        time = np.asarray(trace.time_s)
        if len(time) < 2 or np.any(np.diff(time) <= 0):
            raise ValueError("stability requires increasing timestamps and at least two samples")
        period = float(np.median(np.diff(time)))
        duration = time[-1] + period
        total_duration += float(duration - time[0])
        if settings.drift_window_s * 2 > duration:
            raise ValueError("stability drift windows must not overlap")
        signals = _trace_signal_values(trace=trace) + (
            (
                "horizontal_speed",
                "m/s",
                np.linalg.norm(np.asarray(trace.linear_velocity_enu_mps)[:, :2], axis=1),
            ),
            ("speed", "m/s", np.linalg.norm(trace.linear_velocity_enu_mps, axis=1)),
            ("angular_speed", "rad/s", np.linalg.norm(trace.angular_velocity_enu_radps, axis=1)),
        )
        for name, unit, values in signals:
            by_signal.setdefault((name, unit), []).append(values)
        horizontal = _horizontal(trace)
        rms.append(float(np.sqrt(np.mean(horizontal**2))))
        peaks.append(float(np.max(horizontal)))
        excursion = horizontal > settings.excursion_horizontal_m
        excursion_samples += int(np.count_nonzero(excursion))
        excursion_flights += int(np.any(excursion))
        boundaries = np.diff(np.r_[False, excursion, False].astype(int))
        starts, ends = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
        excursion_count += len(starts)
        if len(starts):
            edges = np.r_[time, duration]
            longest = max(longest, float(np.max(edges[ends] - edges[starts])))
        positions = np.asarray(trace.position_enu_m)
        endpoint_delta = positions[time >= duration - settings.drift_window_s].mean(
            axis=0
        ) - positions[time < settings.drift_window_s].mean(axis=0)
        displacement = max(displacement, float(np.linalg.norm(endpoint_delta)))
        centered_time = time - time.mean()
        slope = (
            centered_time @ (positions - positions.mean(axis=0)) / (centered_time @ centered_time)
        )
        trend = max(trend, float(np.linalg.norm(slope)))
    sample_count = sum(len(trace.time_s) for trace in traces)
    return _HoverSummary(
        identity=identity,
        flight_count=len(traces),
        sample_count=sample_count,
        signals=tuple(
            summarize_signal(name=name, unit=unit, values=np.concatenate(arrays))
            for (name, unit), arrays in by_signal.items()
        ),
        worst_rms_flight=_identity(traces[int(np.argmax(rms))]),
        worst_flight_horizontal_rmse_m=max(rms),
        peak_flight=_identity(traces[int(np.argmax(peaks))]),
        excursion_sample_fraction=excursion_samples / sample_count,
        excursion_flight_count=excursion_flights,
        excursion_count=excursion_count,
        excursion_rate_hz=excursion_count / total_duration,
        longest_excursion_s=longest,
        maximum_endpoint_displacement_m=displacement,
        maximum_position_trend_mps=trend,
    )


def _difference(*, identity: str, field: str, reference: Any, current: Any) -> _Difference:
    a, b = np.asarray(reference), np.asarray(current)
    equal = np.array_equal(a, b)
    first = maximum = rmse = None
    if a.shape == b.shape and a.dtype.kind in "iuf" and b.dtype.kind in "iuf":
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError(f"non-finite stability evidence: {identity}/{field}")
        delta = b.astype(float) - a.astype(float)
        maximum = float(np.max(np.abs(delta))) if delta.size else 0.0
        rmse = float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0
        if not equal:
            first = int(np.argwhere(a != b)[0, 0]) if a.ndim else 0
    return _Difference(
        identity=identity,
        field=field,
        equal=equal,
        reference_count=len(a) if a.ndim else 1,
        current_count=len(b) if b.ndim else 1,
        first_different_index=first,
        maximum_absolute_difference=maximum,
        rmse_difference=rmse,
    )


_BRIDGE_SCALARS = re.compile(r"\b(sim_time_us|last_actuator_us|armed)=([^ ]+)")
_BRIDGE_ARRAYS = re.compile(
    r"\b(commands|position_enu|quat_wxyz|velocity_enu|angular_velocity_body_flu)=(\[[^\]]+\])"
)


def _bridge_rows(path: Path, /):
    previous_time = -1
    with open_flight_evidence(path) as stream:
        for encoded in stream:
            line = encoded.decode("utf-8")
            if "PX4 bridge dynamics:" not in line:
                continue
            row = {
                key: (None if value == "None" else int(value))
                if key != "armed"
                else int(value == "True")
                for key, value in _BRIDGE_SCALARS.findall(line)
            }
            row.update({key: json.loads(value) for key, value in _BRIDGE_ARRAYS.findall(line)})
            if len(row) != 8:
                raise ValueError(f"incomplete bridge dynamics evidence in {path}")
            timestamp = row["sim_time_us"]
            if timestamp is None:
                raise ValueError(f"missing bridge simulation timestamp in {path}")
            if timestamp <= previous_time:
                raise ValueError(f"non-increasing bridge simulation timestamps in {path}")
            previous_time = timestamp
            yield row


def _bridge_differences(
    *, reference: Path, current: Path, identity: str
) -> tuple[_Difference, ...]:
    # Stream full-rate logs; wall time, log prefixes, paths, and host IDs are excluded.
    counts = [0, 0]
    statistics: dict[str, dict[str, Any]] = {}
    for index, (a, b) in enumerate(zip_longest(_bridge_rows(reference), _bridge_rows(current))):
        counts[0] += a is not None
        counts[1] += b is not None
        for field in a or b or {}:
            item = statistics.setdefault(
                field,
                dict(
                    equal=True,
                    first=None,
                    maximum=0.0,
                    square_sum=0.0,
                    size=0,
                    a_time=None,
                    b_time=None,
                ),
            )
            av = np.asarray(a[field]) if a is not None else None
            bv = np.asarray(b[field]) if b is not None else None
            if av is None or bv is None or not np.array_equal(av, bv):
                item["equal"] = False
                if item["first"] is None:
                    item.update(
                        first=index,
                        a_time=a["sim_time_us"] if a else None,
                        b_time=b["sim_time_us"] if b else None,
                    )
            if (
                av is not None
                and bv is not None
                and av.shape == bv.shape
                and av.dtype.kind in "iuf"
                and bv.dtype.kind in "iuf"
            ):
                delta = bv.astype(float) - av.astype(float)
                if not np.all(np.isfinite(delta)):
                    raise ValueError("non-finite bridge stability evidence")
                item["maximum"] = max(item["maximum"], float(np.max(np.abs(delta))))
                item["square_sum"] += float(np.sum(delta**2))
                item["size"] += delta.size
    if min(counts) == 0:
        raise ValueError("stability requires recorded bridge dynamics in both response flights")
    return tuple(
        _Difference(
            identity=identity,
            field=f"bridge/{field}",
            equal=item["equal"],
            reference_count=counts[0],
            current_count=counts[1],
            first_different_index=item["first"],
            maximum_absolute_difference=item["maximum"] if item["size"] else None,
            rmse_difference=float(np.sqrt(item["square_sum"] / item["size"]))
            if item["size"]
            else None,
            first_reference_sim_time_us=item["a_time"],
            first_current_sim_time_us=item["b_time"],
        )
        for field, item in statistics.items()
    )


def _population(
    *,
    group: str,
    signal: str,
    reference: tuple[np.ndarray, ...],
    current: tuple[np.ndarray, ...],
    settings: FlightStabilityConfig,
) -> tuple[_PopulationComparison, ...]:
    a, b = reference, current
    if not a or not b:
        return ()
    if any(
        values.ndim != 1 or not values.size or not np.all(np.isfinite(values))
        for values in (*a, *b)
    ):
        raise ValueError("stability requires nonempty finite scalar flight signals")
    rng = np.random.default_rng(settings.bootstrap_seed)
    enough = min(len(a), len(b)) >= settings.minimum_flights

    def metrics(flights: tuple[np.ndarray, ...], /) -> np.ndarray:
        pooled = np.concatenate(flights)
        return np.array(
            [
                np.sqrt(np.mean(pooled**2)),
                np.percentile(pooled, 99),
                np.max(pooled),
                max(np.sqrt(np.mean(f**2)) for f in flights),
            ]
        )

    base, observed = metrics(a), metrics(b)
    distributions: list[np.ndarray] = []
    if enough and np.all(base > 0):
        for _ in range(settings.bootstrap_repetitions):
            sampled_a = metrics(tuple(a[i] for i in rng.integers(len(a), size=len(a))))
            sampled_b = metrics(tuple(b[i] for i in rng.integers(len(b), size=len(b))))
            if np.any(sampled_a[:2] == 0):
                enough = False
                break
            distributions.append(sampled_b[:2] / sampled_a[:2] - 1)
    result = []
    for index, name in enumerate(("rmse", "p99", "peak", "worst_flight_rmse")):
        change = float(observed[index] / base[index] - 1) if base[index] > 0 else None
        interval = None
        status = "insufficient-evidence"
        if index < 2 and enough and distributions:
            bounds = np.quantile(
                np.asarray(distributions)[:, index],
                [(1 - settings.confidence) / 2, (1 + settings.confidence) / 2],
            )
            interval = (float(bounds[0]), float(bounds[1]))
            status = (
                "pass" if max(abs(bounds)) <= settings.relative_change_limit else "inconclusive"
            )
            if (
                bounds[0] > settings.relative_change_limit
                or bounds[1] < -settings.relative_change_limit
            ):
                status = "fail"
        elif index >= 2 and change is not None:
            status = "fail" if change > settings.relative_change_limit else "pass"
        result.append(
            _PopulationComparison(
                group=group,
                signal=f"{signal}/{name}",
                reference_value=float(base[index]),
                current_value=float(observed[index]),
                reference_flights=len(a),
                current_flights=len(b),
                relative_change=change,
                confidence_interval=interval,
                status=status,
            )
        )
    return tuple(result)


def _response_signals(trace: dict[str, Any], /) -> dict[str, np.ndarray]:
    position = np.asarray(trace["position_enu_m"])
    velocity = np.asarray(trace["velocity_body_flu_mps"])
    rates = np.asarray(trace["angular_velocity_body_flu_radps"])
    signals = {"displacement_m": np.linalg.norm(position - trace["initial_position_enu_m"], axis=1)}
    if trace["control_mode"] == "velocity":
        signals["velocity_error_mps"] = np.linalg.norm(
            velocity - trace["command_velocity_body_flu_mps"], axis=1
        )
        signals["body_yaw_rate_error_radps"] = np.abs(rates[:, 2] - trace["command_yaw_rate_radps"])
    elif trace["control_mode"] == "ctbr":
        signals["body_rate_error_radps"] = np.linalg.norm(
            rates - trace["command_body_rate_flu_radps"], axis=1
        )
        signals["speed_mps"] = np.linalg.norm(velocity, axis=1)
    else:
        raise ValueError("stability response control mode is unsupported")
    return signals


def _response_population(
    *,
    reference: tuple[dict[str, Any], ...],
    current: tuple[dict[str, Any], ...],
    settings: FlightStabilityConfig,
) -> tuple[_PopulationComparison, ...]:
    # Each profile is repeated in independent flights. Profiles and their time
    # samples within one flight never count as independent repetitions.
    signals_by_run = []
    identity_fields = ("name", "kind", "control_mode", "axis")
    identities = tuple(tuple(t[field] for field in identity_fields) for t in reference[0]["traces"])
    for flights in (reference, current):
        signals: dict[tuple[str, str], list[np.ndarray]] = {}
        for flight in flights:
            profile_identities = tuple(
                tuple(t[field] for field in identity_fields) for t in flight["traces"]
            )
            if profile_identities != identities:
                raise ValueError("stability response profile identities differ")
            for trace in flight["traces"]:
                for name, values in _response_signals(trace).items():
                    if len(values) != len(trace["time_s"]):
                        raise ValueError("stability response signal sample counts differ")
                    signals.setdefault((trace["name"], name), []).append(values)
        signals_by_run.append(signals)
    a, b = signals_by_run
    return tuple(
        item
        for (profile, signal), values in a.items()
        for item in _population(
            group=f"response/{profile}",
            signal=signal,
            reference=tuple(values),
            current=tuple(b[(profile, signal)]),
            settings=settings,
        )
    )


def _status(statuses: set[str], /) -> str:
    for status in ("incompatible", "fail", "insufficient-evidence", "inconclusive"):
        if status in statuses:
            return status
    return "pass" if statuses == {"pass"} else "insufficient-evidence"


def _response_summary(
    *, profile: str, signal: str, flights: tuple[dict[str, Any], ...]
) -> _ResponseSummary:
    arrays = tuple(
        _response_signals(next(t for t in flight["traces"] if t["name"] == profile))[signal]
        for flight in flights
    )
    rms = tuple(float(np.sqrt(np.mean(a**2))) for a in arrays)
    peaks = tuple(float(np.max(np.abs(a))) for a in arrays)
    unit = "m/s" if signal.endswith("mps") else "rad/s" if signal.endswith("radps") else "m"
    return _ResponseSummary(
        profile,
        signal,
        len(flights),
        sum(len(a) for a in arrays),
        summarize_signal(name=signal, unit=unit, values=np.concatenate(arrays)),
        flights[int(np.argmax(rms))]["seed"],
        max(rms),
        flights[int(np.argmax(peaks))]["seed"],
    )


def _load_run(
    path: Path, /
) -> tuple[FlightDataset, tuple[FlightTrace, ...], tuple[dict[str, Any], ...]]:
    dataset = load_flight_dataset(path)
    hover = []
    response = []
    for flight in dataset.flights:
        views = flight_stage_views(flight)
        hover.append(_hover_trace(dataset=dataset, flight=flight, view=views[0]))
        response.append(
            {
                "seed": flight.seed,
                "traces": views[1:],
                "controller_state_method": flight.metadata["controller_state_method"],
            }
        )
        bridge = flight.directory / "bridge.log.gz"
        if sum(1 for _ in _bridge_rows(bridge)) == 0:
            raise ValueError(f"stability requires recorded bridge dynamics in {bridge}")
    return dataset, tuple(hover), tuple(response)


def analyze_flight_parameters_stability(
    *, run_dir: Path, comparison_run_dirs: tuple[Path, ...], settings: FlightStabilityConfig
) -> StabilityReport:
    """Analyze finished unsplit flights; exact sample differences are diagnostic."""
    _validate_settings(settings)
    paths = tuple(path.resolve(strict=True) for path in (run_dir, *comparison_run_dirs))
    if len(set(paths)) != len(paths):
        raise ValueError("stability comparisons require distinct producer runs")
    loaded = tuple(_load_run(path) for path in paths)
    summaries = []
    for path, (dataset, hover, response) in zip(paths, loaded, strict=True):
        evidence_paths = tuple(
            p.relative_to(path).as_posix()
            for flight in dataset.flights
            for p in (
                flight.directory / "flight.json",
                flight.directory / "motion.jsonl",
                flight.directory / "bridge.log.gz",
            )
        )
        summaries.append(
            _RunSummary(
                run_dir=str(path),
                run_id=dataset.run_id,
                artifact_sha256=dataset.artifact_digest,
                manifest_sha256=_digest(path / "manifest.json")
                if (path / "manifest.json").exists()
                else None,
                evidence_sha256=tuple((p, _digest(path / p)) for p in evidence_paths),
                hover=tuple(
                    _hover_summary(identity=identity, traces=traces, settings=settings)
                    for identity, traces in (
                        ("campaign", hover),
                        *((_identity(t), (t,)) for t in hover),
                    )
                    if traces
                ),
                response=tuple(
                    _response_summary(profile=name, signal=signal, flights=response)
                    for name, signal in (
                        (trace["name"], signal)
                        for flight in response[:1]
                        for trace in flight["traces"]
                        for signal in _response_signals(trace)
                    )
                ),
            )
        )
    current_dataset, current_hover, current_response = loaded[0]
    comparisons = []
    for reference_dataset, reference_hover, reference_response in loaded[1:]:
        current_seeds = tuple(f.seed for f in current_dataset.flights)
        reference_seeds = tuple(f.seed for f in reference_dataset.flights)
        relationship = (
            "identical"
            if current_seeds == reference_seeds
            else "disjoint"
            if not set(current_seeds) & set(reference_seeds)
            else "overlapping"
        )
        compatible = {
            key: value
            for key, value in current_dataset.definition.items()
            if key not in {"start_seed", "flight_count"}
        } == {
            key: value
            for key, value in reference_dataset.definition.items()
            if key not in {"start_seed", "flight_count"}
        }
        for current_flight, reference_flight in zip(
            current_dataset.flights[:1], reference_dataset.flights[:1], strict=False
        ):
            compatible = (
                compatible
                and all(
                    current_flight.metadata[field] == reference_flight.metadata[field]
                    for field in ("sample_rate_hz", "controller_state_method")
                )
                and tuple(
                    (t["name"], t["control_mode"], t["commands"], t["hover"])
                    for t in current_flight.metadata["stages"]
                )
                == tuple(
                    (t["name"], t["control_mode"], t["commands"], t["hover"])
                    for t in reference_flight.metadata["stages"]
                )
            )
        differences = []
        if relationship == "identical" and compatible:
            for a, b in zip(reference_hover, current_hover, strict=True):
                for field in (
                    "measurement_start_us",
                    "measurement_end_us",
                    "target_position_enu_m",
                    "target_attitude_flu_enu_wxyz",
                    "time_s",
                    "position_enu_m",
                    "attitude_flu_enu_wxyz",
                    "linear_velocity_enu_mps",
                    "angular_velocity_enu_radps",
                ):
                    differences.append(
                        _difference(
                            identity=_identity(b),
                            field=field,
                            reference=getattr(a, field),
                            current=getattr(b, field),
                        )
                    )
            for a, b in zip(reference_response, current_response, strict=True):
                for ap, bp in zip(a["traces"], b["traces"], strict=True):
                    for field in sorted(ap.keys() | bp.keys()):
                        av, bv = ap[field], bp[field]
                        if isinstance(av, dict) or isinstance(bv, dict) or field == "commands":
                            av, bv = json.dumps(av, sort_keys=True), json.dumps(bv, sort_keys=True)
                        differences.append(
                            _difference(
                                identity=f"seed-{b['seed']}/{bp['name']}",
                                field=field,
                                reference=av,
                                current=bv,
                            )
                        )
            for a, b in zip(reference_dataset.flights, current_dataset.flights, strict=True):
                differences.extend(
                    _bridge_differences(
                        reference=a.directory / "bridge.log.gz",
                        current=b.directory / "bridge.log.gz",
                        identity=f"seed-{b.seed}/full-flight",
                    )
                )
        population = _population(
            group="campaign",
            signal="horizontal_m",
            reference=tuple(_horizontal(t) for t in reference_hover),
            current=tuple(_horizontal(t) for t in current_hover),
            settings=settings,
        )
        response_population = (
            _response_population(
                reference=reference_response, current=current_response, settings=settings
            )
            if compatible and reference_response and current_response
            else ()
        )
        comparisons.append(
            _Comparison(
                reference_run_id=reference_dataset.run_id,
                seed_relationship=relationship,
                repeatability_status=("pass" if all(d.equal for d in differences) else "fail")
                if relationship == "identical" and compatible
                else "not-applicable",
                differences=tuple(differences),
                population=population,
                population_status=_status({p.status for p in population})
                if compatible
                else "incompatible",
                response_population=response_population,
                response_status=_status({p.status for p in response_population})
                if compatible
                else "incompatible",
            )
        )
    status = _status({s for c in comparisons for s in (c.population_status, c.response_status)})
    if status == "pass" and not {"identical", "disjoint"} <= {
        c.seed_relationship for c in comparisons
    }:
        status = "insufficient-evidence"
    return StabilityReport(
        artifact_kind="robotics-experiments-flight-parameters-stability",
        schema_version=1,
        settings=settings,
        settings_sha256=hashlib.sha256(
            json.dumps(
                asdict(settings), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
        runs=tuple(summaries),
        comparisons=tuple(comparisons),
        status=status,
    )
