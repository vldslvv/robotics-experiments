"""Read-only analysis of completed flight evidence, independent of capture and consumers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import OmegaConf

from robotics_experiments.flight_parameters._artifact import (
    load_flight_parameters,
    load_flight_parameters_bundle,
    load_flight_trace,
)
from robotics_experiments.flight_parameters._derive import _trace_signal_values, summarize_signal
from robotics_experiments.flight_parameters._evidence import (
    flight_evidence_sha256,
    open_flight_evidence,
    resolve_flight_evidence,
)
from robotics_experiments.flight_parameters._models import (
    FlightParameters,
    FlightTrace,
    SignalStatistics,
)

__all__: list[str] = []


@dataclass(frozen=True)
class _Settings:
    excursion_horizontal_m: float
    drift_window_s: float
    bootstrap_repetitions: int
    bootstrap_seed: int
    confidence: float
    relative_change_limit: float
    minimum_flights: int


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
class _RunSummary:
    run_dir: str
    run_id: str
    artifact_sha256: str
    artifact_fingerprint: str
    manifest_sha256: str | None
    evidence_sha256: tuple[tuple[str, str], ...]
    hover: tuple[_HoverSummary, ...]


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
    settings: _Settings
    settings_sha256: str
    runs: tuple[_RunSummary, ...]
    comparisons: tuple[_Comparison, ...]
    status: str


def _settings(path: Path, /) -> _Settings:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict) or set(payload) != set(_Settings.__annotations__):
        raise ValueError("stability settings fields mismatch")
    payload = cast(dict[str, Any], payload)
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
    return _Settings(**payload)


def _digest(path: Path, /) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identity(trace: FlightTrace, /) -> str:
    return f"{trace.set_name}/seed-{trace.seed}/trial-{trace.trial_index:03d}"


def _horizontal(trace: FlightTrace, /) -> np.ndarray:
    return np.linalg.norm(
        (np.asarray(trace.position_enu_m) - trace.target_position_enu_m)[:, :2], axis=1
    )


def _hover_summary(
    *, identity: str, traces: tuple[FlightTrace, ...], settings: _Settings
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
    settings: _Settings,
) -> tuple[_PopulationComparison, ...]:
    a, b = reference, current
    if (
        not a
        or not b
        or any(
            values.ndim != 1 or not values.size or not np.all(np.isfinite(values))
            for values in (*a, *b)
        )
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
    settings: _Settings,
) -> tuple[_PopulationComparison, ...]:
    # Each profile is repeated in independent flights. Profiles and their time
    # samples within one flight never count as independent repetitions.
    signals_by_run = []
    identity_fields = ("profile", "family", "control_mode", "axis", "split")
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
                    signals.setdefault((trace["profile"], name), []).append(values)
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


def _load_run(
    path: Path, /
) -> tuple[FlightParameters, tuple[FlightTrace, ...], tuple[dict[str, Any], ...]]:
    parameters = load_flight_parameters(path)
    if (path / "manifest.json").exists():
        load_flight_parameters_bundle(path=path)
    hover = tuple(
        load_flight_trace(path / "evidence" / f"trial-{i:03d}" / "trace.json")
        for i in range(1, parameters.hover_trial_count + 1)
    )
    for trace, expected in zip(hover, parameters.hover.trace_statistics, strict=True):
        if (trace.campaign_run_id, trace.trial_index, trace.set_name, trace.seed) != (
            parameters.run_id,
            expected.trial_index,
            expected.set_name,
            expected.seed,
        ) or not trace.completed:
            raise ValueError("stability hover evidence identity mismatch or incomplete capture")
    response = []
    for record in parameters.response.traces:
        for relative, digest in (
            (record.reference_path, record.reference_sha256),
            (record.ulog_path, record.ulog_sha256),
            (record.bridge_log_path, record.bridge_log_sha256),
        ):
            if flight_evidence_sha256(path / relative) != digest:
                raise ValueError(f"stability response evidence hash mismatch: {relative}")
        payload = json.loads((path / record.reference_path).read_text())
        if (payload["set_name"], payload["seed"], len(payload["traces"])) != (
            record.set_name,
            record.seed,
            record.profile_count,
        ) or any(not t["completed"] for t in payload["traces"]):
            raise ValueError("stability response evidence identity mismatch or incomplete capture")
        response.append(payload)
    bridge_paths = tuple(
        resolve_flight_evidence(path / "evidence" / f"trial-{trace.trial_index:03d}" / "bridge.log")
        for trace in hover
    ) + tuple(path / record.bridge_log_path for record in parameters.response.traces)
    for bridge_path in bridge_paths:
        if sum(1 for _ in _bridge_rows(bridge_path)) == 0:
            raise ValueError(f"stability requires recorded bridge dynamics in {bridge_path}")
    return parameters, hover, tuple(response)


def analyze_flight_parameters_stability(
    *, run_dir: Path, comparison_run_dirs: tuple[Path, ...], settings_path: Path
) -> StabilityReport:
    """Analyze finished evidence without modifying it or invoking any capture work.

    Both repeated and disjoint seeds receive statistical hover comparisons.
    Exact differences are diagnostic. Response profiles are compared across whole
    flights separately from hover. Reports do not designate a baseline.
    """
    settings = _settings(settings_path)
    paths = tuple(path.resolve(strict=True) for path in (run_dir, *comparison_run_dirs))
    if len(set(paths)) != len(paths):
        raise ValueError("stability comparisons require distinct producer runs")
    loaded = tuple(_load_run(path) for path in paths)
    summaries = []
    for path, (parameters, hover, _) in zip(paths, loaded, strict=True):
        groups = (
            (("campaign", hover),)
            + tuple(
                (name, tuple(t for t in hover if t.set_name == name))
                for name in ("fit", "validation", "acceptance")
            )
            + tuple((_identity(t), (t,)) for t in hover)
        )
        evidence_paths = tuple(f"evidence/trial-{t.trial_index:03d}/trace.json" for t in hover)
        evidence_paths += tuple(
            resolve_flight_evidence(path / "evidence" / f"trial-{t.trial_index:03d}" / "bridge.log")
            .relative_to(path)
            .as_posix()
            for t in hover
        )
        evidence_paths += tuple(r.reference_path for r in parameters.response.traces)
        evidence_paths += tuple(r.bridge_log_path for r in parameters.response.traces)
        summaries.append(
            _RunSummary(
                run_dir=str(path),
                run_id=parameters.run_id,
                artifact_sha256=_digest(path / "flight_parameters.json"),
                artifact_fingerprint=parameters.artifact_fingerprint,
                manifest_sha256=_digest(path / "manifest.json")
                if (path / "manifest.json").exists()
                else None,
                evidence_sha256=tuple((p, _digest(path / p)) for p in evidence_paths),
                hover=tuple(
                    _hover_summary(identity=name, traces=traces, settings=settings)
                    for name, traces in groups
                ),
            )
        )
    current_parameters, current_hover, current_response = loaded[0]
    comparisons = []
    for reference_path, (reference_parameters, reference_hover, reference_response) in zip(
        paths[1:], loaded[1:], strict=True
    ):
        current_seeds = tuple((t.set_name, t.seed) for t in current_hover) + tuple(
            (r["set_name"], r["seed"]) for r in current_response
        )
        reference_seeds = tuple((t.set_name, t.seed) for t in reference_hover) + tuple(
            (r["set_name"], r["seed"]) for r in reference_response
        )
        relationship = (
            "identical"
            if current_seeds == reference_seeds
            else "disjoint"
            if not {s for _, s in current_seeds} & {s for _, s in reference_seeds}
            else "overlapping"
        )
        differences = []
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
        for a_response, b_response in zip(reference_response, current_response, strict=True):
            a_profiles = {t["profile"]: t for t in a_response["traces"]}
            b_profiles = {t["profile"]: t for t in b_response["traces"]}
            if a_profiles.keys() != b_profiles.keys():
                raise ValueError("stability response profile identities differ")
            for name, b in b_profiles.items():
                a = a_profiles[name]
                for field in sorted(a.keys() | b.keys()):
                    if field not in a or field not in b:
                        raise ValueError(f"stability response field missing: {name}/{field}")
                    # Source-age dictionaries retain all values; no timestamp alignment is fitted.
                    av, bv = a[field], b[field]
                    if isinstance(av, dict) or isinstance(bv, dict):
                        av, bv = json.dumps(av, sort_keys=True), json.dumps(bv, sort_keys=True)
                    differences.append(
                        _difference(
                            identity=f"{b_response['set_name']}/seed-{b_response['seed']}/{name}",
                            field=field,
                            reference=av,
                            current=bv,
                        )
                    )
        for a_record, b_record in zip(
            reference_parameters.response.traces, current_parameters.response.traces, strict=True
        ):
            differences.extend(
                _bridge_differences(
                    reference=reference_path / a_record.bridge_log_path,
                    current=paths[0] / b_record.bridge_log_path,
                    identity=f"{b_record.set_name}/seed-{b_record.seed}/full-flight",
                )
            )
        population = tuple(
            item
            for group in ("campaign", "fit", "validation", "acceptance")
            for item in _population(
                group=group,
                signal="horizontal_m",
                reference=tuple(
                    _horizontal(t)
                    for t in reference_hover
                    if group == "campaign" or t.set_name == group
                ),
                current=tuple(
                    _horizontal(t)
                    for t in current_hover
                    if group == "campaign" or t.set_name == group
                ),
                settings=settings,
            )
        )
        response_population = _response_population(
            reference=reference_response, current=current_response, settings=settings
        )
        compatible = all(
            getattr(reference_parameters, field) == getattr(current_parameters, field)
            for field in (
                "experiment_fingerprint",
                "vehicle_fingerprint",
                "source_image",
                "sample_rate_hz",
                "measure_s",
                "settle_s",
            )
        )
        method_equal = all(
            a.get("controller_state_method") == b.get("controller_state_method")
            and a.get("definition_fingerprint") == b.get("definition_fingerprint")
            for a, b in zip(reference_response, current_response, strict=True)
        )
        repeatability = "not-applicable"
        if relationship == "identical":
            repeatability = (
                "pass"
                if compatible and method_equal and all(d.equal for d in differences)
                else "fail"
            )
        population_status = (
            _status({p.status for p in population}) if compatible else "incompatible"
        )
        comparisons.append(
            _Comparison(
                reference_run_id=reference_parameters.run_id,
                seed_relationship=relationship,
                repeatability_status=repeatability,
                differences=tuple(differences),
                population=population,
                population_status=population_status,
                response_population=response_population,
                response_status=_status({p.status for p in response_population})
                if compatible and method_equal
                else "incompatible",
            )
        )
    status = _status({s for c in comparisons for s in (c.population_status, c.response_status)})
    relationships = {c.seed_relationship for c in comparisons}
    if status == "pass" and not {"identical", "disjoint"} <= relationships:
        status = "insufficient-evidence"
    return StabilityReport(
        artifact_kind="robotics-experiments-flight-parameters-stability",
        schema_version=1,
        settings=settings,
        settings_sha256=_digest(settings_path),
        runs=tuple(summaries),
        comparisons=tuple(comparisons),
        status=status,
    )
