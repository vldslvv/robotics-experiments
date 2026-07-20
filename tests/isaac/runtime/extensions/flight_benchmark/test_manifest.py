from __future__ import annotations

import tomllib

from tests.support.paths import REPO_ROOT


def test_flight_benchmark_is_an_opt_in_dependent_extension() -> None:
    benchmark_manifest = (
        REPO_ROOT / "isaac/runtime/extensions/flight_benchmark/config/extension.toml"
    )
    production_manifest = (
        REPO_ROOT / "isaac/runtime/extensions/experiment_runtime/config/extension.toml"
    )

    benchmark = tomllib.loads(benchmark_manifest.read_text(encoding="utf-8"))
    production = tomllib.loads(production_manifest.read_text(encoding="utf-8"))

    assert set(benchmark["dependencies"]) >= {
        "experiment_runtime",
        "isaacsim.benchmark.services",
        "lightswitch_spawner",
    }
    assert benchmark["python"]["module"] == [{"name": "FlightBenchmark_python"}]
    assert "flight_benchmark" not in production["dependencies"]
