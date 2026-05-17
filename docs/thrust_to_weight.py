"""
Thrust-to-weight calculator for multirotor drones.

Supports quad, hex, and X8 coaxial topologies. Encapsulates the formula
discussed in the project chat — see docstring on `compute_tw` for the math.

Usage as a library
------------------
    from thrust_to_weight import compute_tw, TWInputs

    result = compute_tw(TWInputs(
        m_base=1616,
        N=8,
        m_motor=66,
        m_prop=5,
        T_motor=1450,
        is_coaxial=True,
        k_coax=0.80,
        k_overlap=1.0,
    ))
    print(result.summary())

Usage as a CLI
--------------
    python thrust_to_weight.py \\
        --m-base 1616 --N 8 --m-motor 66 --m-prop 5 \\
        --T-motor 1450 --coaxial --k-coax 0.80

Run the built-in self-test:
    python thrust_to_weight.py --self-test

Run the project's reference scenarios:
    python thrust_to_weight.py --scenarios
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

@dataclass
class TWInputs:
    """Inputs to the thrust-to-weight calculation.

    Mass units: grams. Thrust units: grams-force (gf). T:W is dimensionless.
    """

    m_base: float          # drone mass excluding motors and props (g)
    N: int                 # total motor count (4 quad, 6 hex, 8 X8)
    m_motor: float         # mass per motor incl. leads (g)
    m_prop: float          # mass per SINGLE propeller, not per pair (g)
    T_motor: float         # bench thrust per motor at chosen V/prop/throttle (gf)
    is_coaxial: bool = False    # True for X8 / Y6 etc.
    k_coax: float = 0.80        # lower-prop thrust factor (only if is_coaxial)
    k_upper: float = 1.0        # upper-prop loss in coax stack (use 0.97 to be conservative)
    k_overlap: float = 1.0      # adjacent-prop interference (1.0 unless props nearly touch)
    label: Optional[str] = None # optional human-readable name for the build

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError("N must be positive")
        if self.is_coaxial and self.N % 2 != 0:
            raise ValueError("Coaxial topologies require an even motor count")
        if not (0.0 < self.k_coax <= 1.0):
            raise ValueError("k_coax must be in (0, 1]")
        if not (0.0 < self.k_upper <= 1.0):
            raise ValueError("k_upper must be in (0, 1]")
        if not (0.0 < self.k_overlap <= 1.0):
            raise ValueError("k_overlap must be in (0, 1]")
        for name, val in (("m_base", self.m_base), ("m_motor", self.m_motor),
                          ("m_prop", self.m_prop), ("T_motor", self.T_motor)):
            if val < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class TWResult:
    auw_g: float           # all-up weight (g)
    thrust_total_g: float  # total static thrust (gf)
    tw_ratio: float        # thrust-to-weight, dimensionless
    hover_throttle: float  # fractional, e.g. 0.46 means ~46%
    inputs: TWInputs = field(repr=False)

    def summary(self) -> str:
        label = f"  [{self.inputs.label}]" if self.inputs.label else ""
        topology = (
            f"X{self.inputs.N} coaxial (k_coax={self.inputs.k_coax:.2f})"
            if self.inputs.is_coaxial
            else f"{self.inputs.N}-motor non-coax"
        )
        rating = _rate_tw(self.tw_ratio)
        return (
            f"Build{label}\n"
            f"  Topology         : {topology}\n"
            f"  AUW              : {self.auw_g:.0f} g\n"
            f"  Total thrust     : {self.thrust_total_g:.0f} gf\n"
            f"  T:W ratio        : {self.tw_ratio:.2f}:1   ({rating})\n"
            f"  Hover throttle   : {self.hover_throttle * 100:.0f} %\n"
        )


def _rate_tw(tw: float) -> str:
    """Qualitative rating tied to the project's contact-task thresholds."""
    if tw >= 4.0:
        return "EXCELLENT — large margin for contact transients and payload growth"
    if tw >= 3.0:
        return "STRONG — clears the 'ideal' threshold for contact tasks"
    if tw >= 2.5:
        return "ADEQUATE — meets the 2.5:1 contact-task floor"
    if tw >= 2.0:
        return "MARGINAL — flyable but no headroom for disturbance rejection"
    return "INSUFFICIENT — drone will struggle to hover with payload"


def compute_tw(inp: TWInputs) -> TWResult:
    """Compute thrust-to-weight ratio for a multirotor.

    Formula
    -------
    AUW = m_base + N * (m_motor + m_prop)

    Non-coaxial:
        T_total = N * T_motor * k_overlap

    Coaxial (X8 / Y6 / etc.):
        T_total = (N/2) * T_motor * k_overlap * (k_upper + k_coax)

        — half the motors are upper (each producing T_motor * k_overlap * k_upper),
          half are lower (each producing T_motor * k_overlap * k_coax).
        — k_upper defaults to 1.0; set ~0.97 to account for the small upstream
          disturbance the lower prop induces on the upper prop.

    T:W = T_total / AUW

    hover_throttle = sqrt(AUW / T_total)
        — valid when T_motor was measured at 100% throttle, since thrust scales
          ~throttle². Tells you the operating point during stable hover.
    """
    auw = inp.m_base + inp.N * (inp.m_motor + inp.m_prop)

    if inp.is_coaxial:
        per_pair = inp.T_motor * inp.k_overlap * (inp.k_upper + inp.k_coax)
        thrust_total = (inp.N / 2) * per_pair
    else:
        thrust_total = inp.N * inp.T_motor * inp.k_overlap

    tw = thrust_total / auw
    # Guard against >100% hover (drone can't hover; sqrt of >1 is meaningless here)
    if thrust_total <= auw:
        hover_throttle = 1.0  # can't hover at any throttle, pin to 100%
    else:
        hover_throttle = math.sqrt(auw / thrust_total)

    return TWResult(
        auw_g=auw,
        thrust_total_g=thrust_total,
        tw_ratio=tw,
        hover_throttle=hover_throttle,
        inputs=inp,
    )


# ---------------------------------------------------------------------------
# Reference scenarios from the project (X500 V2 X8 build)
# ---------------------------------------------------------------------------

# Mid-config base AUW excluding motors and props is 1,616 g
# Max-config base (with 3D LiDAR + Orin NX upgrade) is 1,927 g
# Both come from the BOM after subtracting 8*100g motors and 4*20g prop pairs.
PROJECT_SCENARIOS = [
    # The current BOM spec
    TWInputs(
        label="BOM as-spec'd: MN3110 KV700 + 12x4.5 polymer (mid)",
        m_base=1616, N=8, m_motor=100, m_prop=5,
        T_motor=1150, is_coaxial=True, k_coax=0.80,
    ),
    TWInputs(
        label="BOM as-spec'd: MN3110 KV700 + 12x4.5 polymer (max)",
        m_base=1927, N=8, m_motor=100, m_prop=5,
        T_motor=1150, is_coaxial=True, k_coax=0.80,
    ),
    # Recommended swap
    TWInputs(
        label="Recommended: MN4006 KV380 + P134.4 13x4.4 CF (mid)",
        m_base=1616, N=8, m_motor=66, m_prop=7,
        T_motor=1450, is_coaxial=True, k_coax=0.80,
    ),
    TWInputs(
        label="Recommended: MN4006 KV380 + P134.4 13x4.4 CF (max)",
        m_base=1927, N=8, m_motor=66, m_prop=7,
        T_motor=1450, is_coaxial=True, k_coax=0.80,
    ),
    # Top-end variant
    TWInputs(
        label="High-perf: MN4006 EVO 380 + P134.4 13x4.4 CF (max)",
        m_base=1927, N=8, m_motor=68, m_prop=7,
        T_motor=1565, is_coaxial=True, k_coax=0.80,
    ),
    # Stage 1 quad reference (no coax)
    TWInputs(
        label="Stage 1 quad: MN3110 KV700 + 12x4.5 polymer",
        m_base=1616 - 88,  # subtract the 4 lower prop guards (Stage 1 has no lowers)
        N=4, m_motor=100, m_prop=5,
        T_motor=1150, is_coaxial=False,
    ),
]


def run_scenarios() -> None:
    print("=" * 78)
    print("Project reference scenarios — Holybro X500 V2 X8 build")
    print("=" * 78)
    for inp in PROJECT_SCENARIOS:
        print(compute_tw(inp).summary())


# ---------------------------------------------------------------------------
# Self-test (regression-style; matches numbers from the chat analysis)
# ---------------------------------------------------------------------------

def run_self_test() -> int:
    """Verify the formula against the worked examples from the chat."""
    failed = 0

    def check(name: str, actual: float, expected: float, tol: float = 0.02) -> None:
        nonlocal failed
        rel = abs(actual - expected) / max(abs(expected), 1e-9)
        ok = rel <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: got {actual:.3f}, expected {expected:.3f}")
        if not ok:
            failed += 1

    print("Self-test: BOM spec MN3110 KV700 mid-config")
    r = compute_tw(TWInputs(
        m_base=1616, N=8, m_motor=100, m_prop=5,
        T_motor=1150, is_coaxial=True, k_coax=0.80,
    ))
    check("AUW", r.auw_g, 2456)
    check("thrust", r.thrust_total_g, 8280)
    check("T:W", r.tw_ratio, 3.371)
    check("hover throttle", r.hover_throttle, 0.545)

    print("\nSelf-test: Recommended MN4006 + 13x4.4 mid-config")
    r = compute_tw(TWInputs(
        m_base=1616, N=8, m_motor=66, m_prop=7,
        T_motor=1450, is_coaxial=True, k_coax=0.80,
    ))
    # AUW = 1616 + 8*73 = 2200
    # T = 4 * 1450 * 1.80 = 10440
    # T:W = 10440/2200 = 4.745
    check("AUW", r.auw_g, 2200)
    check("thrust", r.thrust_total_g, 10440)
    check("T:W", r.tw_ratio, 4.745)

    print("\nSelf-test: Stage 1 quad (non-coaxial)")
    r = compute_tw(TWInputs(
        m_base=1528, N=4, m_motor=100, m_prop=5,
        T_motor=1150, is_coaxial=False,
    ))
    # AUW = 1528 + 4*105 = 1948
    # T = 4 * 1150 = 4600
    # T:W = 4600/1948 = 2.361
    check("AUW", r.auw_g, 1948)
    check("thrust", r.thrust_total_g, 4600)
    check("T:W", r.tw_ratio, 2.361)

    print("\nSelf-test: input validation")
    try:
        compute_tw(TWInputs(m_base=1000, N=7, m_motor=66, m_prop=5,
                            T_motor=1000, is_coaxial=True))
    except ValueError as e:
        print(f"  [PASS] caught odd-N coaxial: {e}")
    else:
        print("  [FAIL] should have rejected odd N for coaxial")
        failed += 1

    print()
    print("ALL TESTS PASSED" if failed == 0 else f"{failed} TEST(S) FAILED")
    return failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute thrust-to-weight ratio for a multirotor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--m-base", type=float,
                   help="Drone mass excluding motors and props (g)")
    p.add_argument("--N", type=int,
                   help="Total motor count")
    p.add_argument("--m-motor", type=float,
                   help="Mass per motor incl. leads (g)")
    p.add_argument("--m-prop", type=float,
                   help="Mass per single propeller, not per pair (g)")
    p.add_argument("--T-motor", type=float,
                   help="Bench thrust per motor at the chosen operating point (gf)")
    p.add_argument("--coaxial", action="store_true",
                   help="Mark this as a coaxial build (X8/Y6 etc.)")
    p.add_argument("--k-coax", type=float, default=0.80,
                   help="Lower-prop thrust factor (only used if --coaxial)")
    p.add_argument("--k-upper", type=float, default=1.0,
                   help="Upper-prop thrust factor in coax stack (0.97 to be conservative)")
    p.add_argument("--k-overlap", type=float, default=1.0,
                   help="Adjacent-prop interference factor (1.0 unless props nearly touch)")
    p.add_argument("--label", type=str, default=None,
                   help="Optional label for the output")

    p.add_argument("--scenarios", action="store_true",
                   help="Print the project's reference scenarios")
    p.add_argument("--self-test", action="store_true",
                   help="Run regression tests and exit")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.self_test:
        return 1 if run_self_test() else 0
    if args.scenarios:
        run_scenarios()
        return 0

    required = ["m_base", "N", "m_motor", "m_prop", "T_motor"]
    missing = [r for r in required if getattr(args, r.replace("-", "_")) is None]
    if missing:
        print(f"Missing required arguments: {', '.join('--' + m.replace('_', '-') for m in missing)}",
              file=sys.stderr)
        print("Run with --scenarios for project reference builds, or --self-test to validate.",
              file=sys.stderr)
        return 2

    inp = TWInputs(
        m_base=args.m_base, N=args.N, m_motor=args.m_motor, m_prop=args.m_prop,
        T_motor=args.T_motor, is_coaxial=args.coaxial,
        k_coax=args.k_coax, k_upper=args.k_upper, k_overlap=args.k_overlap,
        label=args.label,
    )
    print(compute_tw(inp).summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
