"""Cycle summaries, a quality gate and early-life features for cycle-life models.

The features follow Severson et al. (2019), as in the battery-cycle-life
research code (src/features.py). One function, early_features(), does the
maths for both sources: a BioLogic .mpr file (features_from_cycles) and a cell
from the public Severson dataset (features_from_severson_cell), so a model
trained on one can score the other.

Units match the Severson data: capacity in Ah, charge time in minutes.
Cycle N is the N-th full cycle from group_full_cycles(), counting from 1.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

V_POINTS = 1000  # points on the ΔQ(V) voltage grid, as in the paper
EARLY_CYCLE = 10
LATE_CYCLE = 100
MIN_CE, MAX_CE = 0.5, 1.1

FEATURE_COLUMNS = (
    "log_var_dq",
    "log_abs_min_dq",
    "log_abs_skew_dq",
    "log_abs_kurt_dq",
    "qd_cycle2",
    "qd_max_minus_cycle2",
    "fade_slope",
    "fade_intercept",
    "charge_time_2to6",
    "temp_max",
    "ir_min",
    "ir_change_2to100",
)
# What a .mpr file can give: no temperature or internal resistance.
REDUCED_FEATURES = FEATURE_COLUMNS[:9]


class FeatureError(ValueError):
    """The cycles cannot give the features; the message says why."""


# --- Per-cycle summary ---------------------------------------------------------


def end_capacity(part):
    """Capacity at the end of a half-cycle in mAh, or None if there is none."""
    if not part:
        return None
    return part["capacity_mah"][-1]


def duration_s(part):
    """Length of a half-cycle in seconds, or None without a time column."""
    if not part or not part.get("time_s"):
        return None
    return part["time_s"][-1] - part["time_s"][0]


def cycle_summary(cycles):
    """One row per full cycle: cycle_no, discharge_mah, charge_mah,
    coulombic_eff (discharge / charge) and charge_time_s. Missing values are
    None.
    """
    rows = []
    for cycle in cycles:
        charge = end_capacity(cycle.get("charge"))
        discharge = end_capacity(cycle.get("discharge"))
        efficiency = None
        if charge and discharge is not None:
            efficiency = discharge / charge
        rows.append(
            {
                "cycle_no": cycle["cycle"],
                "discharge_mah": discharge,
                "charge_mah": charge,
                "coulombic_eff": efficiency,
                "charge_time_s": duration_s(cycle.get("charge")),
            }
        )
    return rows


# --- Quality gate --------------------------------------------------------------


def gate_step(check, rows_in, rows_out, passed, detail=""):
    """One line of a quality gate report. passed means the data got through."""
    return {
        "check": check,
        "passed": bool(passed),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "detail": detail,
    }


def parts_of(cycle):
    return [cycle[name] for name in ("charge", "discharge") if name in cycle]


def all_finite(cycle):
    for part in parts_of(cycle):
        values = part["capacity_mah"] + part["voltage_v"] + (part.get("time_s") or [])
        if not np.all(np.isfinite(np.asarray(values, dtype=float))):
            return False
    return True


def efficiency_ok(cycle):
    """True if the coulombic efficiency is in range, or cannot be worked out."""
    charge = end_capacity(cycle.get("charge"))
    discharge = end_capacity(cycle.get("discharge"))
    if not charge or discharge is None:
        return True
    return MIN_CE <= discharge / charge <= MAX_CE


def time_forward(cycle):
    for part in parts_of(cycle):
        times = part.get("time_s")
        if times and np.any(np.diff(times) < 0):
            return False
    return True


def check_cycles(cycles):
    """Drop cycles that are unfit for features. Returns (kept cycles, report),
    where the report is a list of gate_step() lines. Kept cycles keep their
    original numbers, so a dropped cycle leaves a gap.
    """
    report = []
    steps = (
        ("Cycle has a discharge", lambda c: "discharge" in c, "no discharge"),
        ("No missing values", all_finite, "NaN or infinite values"),
        (
            f"Coulombic efficiency {MIN_CE}-{MAX_CE}",
            efficiency_ok,
            "efficiency out of range",
        ),
        ("Time moves forward", time_forward, "clock goes backwards"),
    )
    kept = list(cycles)
    for check, keep_it, reason in steps:
        before = len(kept)
        kept = [c for c in kept if keep_it(c)]
        dropped = before - len(kept)
        detail = f"{dropped} dropped: {reason}" if dropped else ""
        report.append(gate_step(check, before, len(kept), len(kept) > 0, detail))

    numbers = set()
    for cycle in kept:
        numbers.add(cycle["cycle"])
    missing = []
    for n in range(2, LATE_CYCLE + 1):
        if n not in numbers:
            missing.append(n)
    detail = f"first missing: cycle {missing[0]}" if missing else ""
    report.append(
        gate_step(
            f"Cycles 2-{LATE_CYCLE} all present",
            len(kept),
            len(kept),
            not missing,
            detail,
        )
    )
    return kept, report


# --- Early-life features -------------------------------------------------------


def despike(x, window=5):
    """Rolling median: removes one-cycle logging spikes, keeps real trends."""
    return pd.Series(x).rolling(window, center=True, min_periods=1).median().to_numpy()


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def early_features(dq, qd, charge_time_min=None, ir=None, temp=None):
    """The 12 early-life features, as in the research code.

    dq: the ΔQ(V) curve in Ah. qd, charge_time_min, ir, temp: one value per
    cycle where index 0 is cycle 1; only cycles 2-100 are used. charge_time_min,
    ir and temp may be None, and give None features. A value that is not finite
    (e.g. log10 of 0) is also None.
    """
    dq = np.asarray(dq, dtype=float)
    qd = despike(np.asarray(qd, dtype=float)[1:LATE_CYCLE])
    cycle_numbers = np.arange(2, LATE_CYCLE + 1)
    fade_slope, fade_intercept = np.polyfit(cycle_numbers, qd, 1)

    # A flat ΔQ(V) gives log10(0) or NaN; those become None below
    with np.errstate(divide="ignore", invalid="ignore"):
        shape = {
            "log_var_dq": np.log10(np.var(dq)),
            "log_abs_min_dq": np.log10(np.abs(dq.min())),
            "log_abs_skew_dq": np.log10(np.abs(skew(dq))),
            "log_abs_kurt_dq": np.log10(np.abs(kurtosis(dq))),
        }
    features = {
        # ΔQ(V) curve shape: the paper's strongest group
        **shape,
        # Capacity fade over cycles 2-100
        "qd_cycle2": qd[0],
        "qd_max_minus_cycle2": qd.max() - qd[0],
        "fade_slope": fade_slope,
        "fade_intercept": fade_intercept,
        # Charging, temperature, internal resistance
        "charge_time_2to6": None,
        "temp_max": None,
        "ir_min": None,
        "ir_change_2to100": None,
    }
    if charge_time_min is not None:
        features["charge_time_2to6"] = np.mean(np.asarray(charge_time_min)[1:6])
    if temp is not None:
        features["temp_max"] = np.max(np.asarray(temp, dtype=float)[1:LATE_CYCLE])
    if ir is not None:
        ir = despike(np.asarray(ir, dtype=float)[1:LATE_CYCLE])
        nonzero = ir[ir > 0]  # a few IR readings are logged as 0
        if nonzero.size:
            features["ir_min"] = nonzero.min()
        features["ir_change_2to100"] = ir[-1] - ir[0]

    result = {}
    for name in FEATURE_COLUMNS:
        result[name] = finite_or_none(features[name])
    return result


def discharge_curve(cycle):
    """(voltage in V, capacity in Ah) of a cycle's discharge."""
    part = cycle["discharge"]
    voltage = np.asarray(part["voltage_v"], dtype=float)
    capacity = np.asarray(part["capacity_mah"], dtype=float) / 1000
    return voltage, capacity


def capacity_on_grid(voltage, capacity, grid):
    """Discharge capacity at each grid voltage. Voltage falls during a
    discharge; a running minimum removes small rises so it can be interpolated.
    """
    falling = np.minimum.accumulate(voltage)
    first = np.concatenate([[True], np.diff(falling) < 0])
    return np.interp(grid, falling[first][::-1], capacity[first][::-1])


def delta_q_curve(cycles):
    """(voltage grid, ΔQ in Ah): capacity at cycle 100 minus cycle 10.

    The paper uses a fixed 3.6-2.0 V grid for its LFP cells. Here the grid is
    the voltage range both discharges cover, so it suits any chemistry.
    """
    by_number = {}
    for cycle in cycles:
        by_number[cycle["cycle"]] = cycle
    for n in (EARLY_CYCLE, LATE_CYCLE):
        if "discharge" not in by_number.get(n, {}):
            raise FeatureError(f"Cycle {n} has no usable discharge.")
    v_early, q_early = discharge_curve(by_number[EARLY_CYCLE])
    v_late, q_late = discharge_curve(by_number[LATE_CYCLE])
    top = min(v_early.max(), v_late.max())
    bottom = max(v_early.min(), v_late.min())
    if top <= bottom:
        raise FeatureError(
            f"Cycles {EARLY_CYCLE} and {LATE_CYCLE} share no voltage range."
        )
    grid = np.linspace(top, bottom, V_POINTS)
    late = capacity_on_grid(v_late, q_late, grid)
    early = capacity_on_grid(v_early, q_early, grid)
    return grid, late - early


def features_from_cycles(cycles, ir=None, temp=None):
    """Features for cycles from group_full_cycles() (after check_cycles()).

    Needs cycles 2-100, each with a discharge. Charge time needs the time
    column (read_half_cycles(..., with_time=True)). Adds v_grid_max and
    v_grid_min, the ΔQ(V) voltage range. Raises FeatureError if it cannot.
    """
    by_number = {}
    for cycle in cycles:
        by_number[cycle["cycle"]] = cycle
    for n in range(2, LATE_CYCLE + 1):
        if "discharge" not in by_number.get(n, {}):
            raise FeatureError(f"Cycle {n} is missing or failed the quality gate.")

    grid, dq = delta_q_curve(cycles)
    qd = [np.nan]  # cycle 1 is never used
    charge_time = [np.nan]
    for n in range(2, LATE_CYCLE + 1):
        qd.append(end_capacity(by_number[n]["discharge"]) / 1000)
        seconds = duration_s(by_number[n].get("charge"))
        charge_time.append(np.nan if seconds is None else seconds / 60)
    if not np.all(np.isfinite(charge_time[1:6])):
        charge_time = None

    features = early_features(dq, qd, charge_time, ir=ir, temp=temp)
    features["v_grid_max"] = float(grid[0])
    features["v_grid_min"] = float(grid[-1])
    return features


def features_from_severson_cell(cell):
    """Features for one cell of the research repo's Severson data: cycles keyed
    by position ("0" is cycle 1) holding Qdlin on the paper's voltage grid, and
    summary arrays QD, chargetime, IR and Tmax (index 0 is cycle 1).
    """
    q_early = np.asarray(cell["cycles"][str(EARLY_CYCLE - 1)]["Qdlin"], dtype=float)
    q_late = np.asarray(cell["cycles"][str(LATE_CYCLE - 1)]["Qdlin"], dtype=float)
    s = cell["summary"]
    return early_features(
        q_late - q_early, s["QD"], s["chargetime"], ir=s["IR"], temp=s["Tmax"]
    )
