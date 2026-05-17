"""
Revised motor evaluation with realistic 2025 pricing.

Pricing sources (verified 2025):
- MN3110 KV700:    Austars $60, UAVGarage ~$85, range $60-85 -> use $70
- MN3508 KV380:    Express Drone Parts $86 single, T-Motor store similar
- MN4004 KV400:    iNsightFPV-class $50/motor in pairs ($101 / 2)
- MN4006 KV380:    iNsightFPV $74.95/motor in pairs ($149.90 / 2)
- MN4006 EVO 380:  ~$10-15 premium over base MN4006 -> $85-90/motor
- MN5008 KV400:    irlock $89.99 (single), heavier-duty motor

Catalog max-thrust corrections:
- MN3110: 1.2kg max (T-Motor official) - was 1.5kg in BOM, that was wrong
- MN3508: 1.8kg max (T-Motor official) - corrected upward from earlier estimate
- MN4006: 1.6kg max (T-Motor official)
- MN4004: 1.1kg max (T-Motor official)
"""

import sys
sys.path.insert(0, "/home/claude")
from thrust_to_weight import compute_tw, TWInputs

MID_BASE = 1616
MAX_BASE = 1927
QUAD_BASE = MID_BASE - 88

# (mass_g, max_continuous_A, price_USD_each, catalog_max_thrust_g)
MOTORS = {
    "MN3110 KV700":   (99,  21, 70,  1200),
    "MN3110 KV780":   (100, 26, 70,  1200),
    "MN3508 KV380":   (87,  17, 86,  1800),
    "MN3508 KV700":   (87,  17, 86,  1800),
    "MN4004 KV400":   (53,  18, 50,  1100),
    "MN4006 KV380":   (66,  22, 75,  1600),
    "MN4006 EVO 380": (68,  24, 88,  1700),
}

# Per-motor static thrust (g) at 4S/14.8V, using T-Motor / Ligpower data and
# physics-based interpolation where direct test data is absent.
# MN3508 numbers revised UPWARD given its 1.8kg catalog max (was clipped too low before).
THRUST_DATA = {
    "MN3110 KV700": [
        ("12x4 CF",    1150, 16.0, 6),
        ("13x4.4 CF",  1200, 18.0, 7),  # capped at catalog max
    ],
    "MN3110 KV780": [
        ("11x3.7 CF",  1080, 13.0, 5),
        ("12x4 CF",    1200, 17.0, 6),  # capped at catalog max
    ],
    "MN3508 KV380": [
        ("12x4 CF",    1180, 13.0, 6),
        ("13x4.4 CF",  1430, 16.0, 7),  # revised up; this motor is rated 1.8kg
    ],
    "MN3508 KV700": [
        ("11x3.7 CF",  1100, 14.5, 5),
        ("12x4 CF",    1380, 16.5, 6),  # revised up
    ],
    "MN4004 KV400": [
        ("11x3.7 CF",   850, 11.0, 5),
        ("12x4 CF",    1050, 13.5, 6),
    ],
    "MN4006 KV380": [
        ("12x4 CF",    1250, 14.0, 6),
        ("13x4.4 CF",  1450, 17.0, 7),
    ],
    "MN4006 EVO 380": [
        ("12x4 CF",    1350, 14.5, 6),
        ("13x4.4 CF",  1565, 17.5, 7),
    ],
}

PROP_PRICE = {
    "11x3.7 CF":  16,  # per pair
    "12x4 CF":    16,
    "13x4.4 CF":  20,
}


def evaluate(motor, prop, m_motor, m_prop, T, I, max_I, price, prop_price):
    mid = compute_tw(TWInputs(
        m_base=MID_BASE, N=8, m_motor=m_motor, m_prop=m_prop,
        T_motor=T, is_coaxial=True, k_coax=0.80,
    ))
    mx = compute_tw(TWInputs(
        m_base=MAX_BASE, N=8, m_motor=m_motor, m_prop=m_prop,
        T_motor=T, is_coaxial=True, k_coax=0.80,
    ))
    current_ok = I <= max_I * 0.85
    motors_cost = price * 8
    props_cost  = prop_price * 4   # 4 pairs serve 8 motor positions
    total_cost  = motors_cost + props_cost
    cost_per_kg_thrust = total_cost / (mid.thrust_total_g / 1000)
    return mid, mx, current_ok, motors_cost, props_cost, total_cost, cost_per_kg_thrust


rows = []
for motor, combos in THRUST_DATA.items():
    m_motor, max_I, price, _ = MOTORS[motor]
    for prop, T, I, m_prop in combos:
        prop_price = PROP_PRICE[prop]
        mid, mx, ok, mcost, pcost, tcost, cpkt = evaluate(
            motor, prop, m_motor, m_prop, T, I, max_I, price, prop_price)
        rows.append({
            "motor": motor, "prop": prop, "T_motor": T, "I": I, "max_I": max_I,
            "current_ok": ok,
            "mid_tw": mid.tw_ratio, "max_tw": mx.tw_ratio,
            "mid_auw": mid.auw_g, "max_auw": mx.auw_g,
            "mid_hover": mid.hover_throttle, "max_hover": mx.max_hover if hasattr(mx, "max_hover") else mx.hover_throttle,
            "thrust_total": mid.thrust_total_g,
            "motors_cost": mcost, "props_cost": pcost, "total_cost": tcost,
            "cost_per_kg_thrust": cpkt,
        })

print("=" * 120)
print("X8 COAXIAL — REVISED with corrected pricing and MN3508/MN4004 added")
print("=" * 120)
rows.sort(key=lambda r: -r["max_tw"])
print(f"{'Motor':<18} {'Prop':<13} {'T1':>5} {'I':>5} {'MidT:W':>7} {'MaxT:W':>7} "
      f"{'MotorsTot':>10} {'PropsTot':>9} {'TotalProp':>10} {'$/kg-thrust':>12}  Status")
print("-" * 120)
for r in rows:
    if not r["current_ok"]:
        status = f"⚠️ {r['I']:.1f}A>85%×{r['max_I']}A"
    elif r["max_tw"] >= 4.0:
        status = "EXCELLENT"
    elif r["max_tw"] >= 3.0:
        status = "STRONG"
    elif r["max_tw"] >= 2.5:
        status = "ADEQUATE"
    else:
        status = "MARGINAL"
    print(f"{r['motor']:<18} {r['prop']:<13} {r['T_motor']:>5} {r['I']:>4.1f}A "
          f"{r['mid_tw']:>5.2f}:1 {r['max_tw']:>5.2f}:1 ${r['motors_cost']:>8} ${r['props_cost']:>7} "
          f"${r['total_cost']:>8} ${r['cost_per_kg_thrust']:>9.0f}/kg  {status}")

print()
print("=" * 120)
print("CURRENT-SAFE COMBINATIONS RANKED BY $/kg-THRUST (cheapest thrust per dollar)")
print("=" * 120)
safe = [r for r in rows if r["current_ok"]]
safe.sort(key=lambda r: r["cost_per_kg_thrust"])
for r in safe:
    rating = "EXCELLENT" if r["max_tw"] >= 4.0 else "STRONG" if r["max_tw"] >= 3.0 else "ADEQUATE" if r["max_tw"] >= 2.5 else "MARGINAL"
    print(f"  {r['motor']:<18} + {r['prop']:<13}  ${r['cost_per_kg_thrust']:>5.0f}/kg-thrust   "
          f"max-T:W {r['max_tw']:.2f}:1  total ${r['total_cost']}  [{rating}]")

print()
print("=" * 120)
print("CURRENT-SAFE COMBINATIONS RANKED BY MAX T:W (highest performance)")
print("=" * 120)
safe.sort(key=lambda r: -r["max_tw"])
for r in safe[:8]:
    margin = r["max_tw"] - 2.5
    print(f"  {r['motor']:<18} + {r['prop']:<13}  max-T:W {r['max_tw']:.2f}:1  "
          f"(margin +{margin:.2f})  total ${r['total_cost']}  ${r['cost_per_kg_thrust']:.0f}/kg-thrust")

# Cost delta vs the BOM baseline
print()
print("=" * 120)
print("COST DELTA vs BOM BASELINE (MN3110 KV700 + 12x4)")
print("=" * 120)
baseline = next(r for r in rows if r["motor"] == "MN3110 KV700" and r["prop"] == "12x4 CF")
print(f"Baseline: {baseline['motor']} + {baseline['prop']} -> ${baseline['total_cost']}, max-T:W {baseline['max_tw']:.2f}")
print()
for r in safe:
    delta = r["total_cost"] - baseline["total_cost"]
    tw_gain = r["max_tw"] - baseline["max_tw"]
    if r["motor"] == baseline["motor"] and r["prop"] == baseline["prop"]:
        continue
    sign = "+" if delta > 0 else ""
    print(f"  {r['motor']:<18} + {r['prop']:<13}  cost delta: {sign}${delta:>4}  "
          f"T:W gain: +{tw_gain:.2f}  -> ${abs(delta)/max(tw_gain,0.01):.0f} per +0.1 T:W")
