"""Cycle summaries, the quality gate and early-life features
(services/cycle_features.py, services/cycling.py). No database or real data needed.

pytest -q tests/test_cycle_features.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.cycle_features import (  # noqa: E402
    FEATURE_COLUMNS,
    REDUCED_FEATURES,
    V_POINTS,
    FeatureError,
    analyse_cycles,
    analyse_file,
    check_cycles,
    cycle_summary,
    delta_q_curve,
    features_from_cycles,
    features_from_severson_cell,
)
from services.cycling import group_full_cycles, split_half_cycles  # noqa: E402

V_TOP, V_BOTTOM = 3.6, 2.0
POINTS = 200


def capacity_mah(cycle):
    """Linear fade: 1100 mAh at cycle 1, losing 0.5 mAh per cycle."""
    return 1100 - 0.5 * cycle


def make_cycles(n=100, charge_minutes=10.0, power=1.0):
    """Full cycles with a discharge curve from 3.6 V to 2.0 V (a straight line
    when power is 1), in the shape group_full_cycles() returns.
    """
    cycles = []
    for k in range(1, n + 1):
        q = np.linspace(0, capacity_mah(k), POINTS)
        t = np.linspace(0, charge_minutes * 60, POINTS)
        falling = V_TOP - (V_TOP - V_BOTTOM) * (q / q[-1]) ** power
        cycles.append(
            {
                "cycle": k,
                "charge": {
                    "capacity_mah": (q / 0.99).tolist(),  # 99% efficiency
                    "voltage_v": falling[::-1].tolist(),
                    "time_s": t.tolist(),
                },
                "discharge": {
                    "capacity_mah": q.tolist(),
                    "voltage_v": falling.tolist(),
                    "time_s": t.tolist(),
                },
            }
        )
    return cycles


# --- services/cycling.py: the reader's new options ------------------------------


def make_raw(points_per_half=1000):
    """Data as galvani gives it: one charge then one discharge."""
    q = np.linspace(0, 5, points_per_half)
    return {
        "half cycle": np.repeat([0, 1], points_per_half),
        "Q charge/discharge/mA.h": np.concatenate([q, 5 - q]),
        "Ewe/V": np.concatenate([3 + q / 10, 3.5 - q / 10]),
        "time/s": np.arange(2 * points_per_half, dtype=float),
    }


def test_reader_defaults_unchanged():
    halves = split_half_cycles(make_raw())
    assert len(halves) == 2
    assert len(halves[0]["capacity_mah"]) <= 401
    assert "time_s" not in halves[0]


def test_reader_full_resolution_with_time():
    halves = split_half_cycles(make_raw(), max_points=None, with_time=True)
    assert len(halves[0]["capacity_mah"]) == 1000
    assert halves[1]["time_s"][0] == 0
    assert halves[1]["time_s"][-1] == 999
    cycles = group_full_cycles(halves)
    assert cycles[0]["charge"]["time_s"][-1] == 999


def test_reader_without_time_column():
    raw = make_raw()
    del raw["time/s"]
    halves = split_half_cycles(raw, with_time=True)
    assert halves[0]["time_s"] is None


# --- Summary and gate ----------------------------------------------------------


def test_cycle_summary():
    row = cycle_summary(make_cycles(n=2))[1]
    assert row["cycle_no"] == 2
    assert row["discharge_mah"] == pytest.approx(1099)
    assert row["coulombic_eff"] == pytest.approx(0.99)
    assert row["charge_time_s"] == pytest.approx(600)


def test_gate_passes_clean_cycles():
    kept, report = check_cycles(make_cycles())
    assert len(kept) == 100
    for step in report:
        assert step["passed"], step


def test_gate_drops_bad_cycles():
    cycles = make_cycles(n=120)
    cycles[49]["discharge"]["voltage_v"][5] = float("nan")
    cycles[109]["discharge"]["capacity_mah"][-1] = 10  # efficiency far too low
    kept, report = check_cycles(cycles)
    by_name = {step["check"]: step for step in report}
    assert by_name["No missing values"]["rows_out"] == 119
    assert by_name["Coulombic efficiency 0.5-1.1"]["rows_out"] == 118
    assert not by_name["Cycles 2-100 all present"]["passed"]
    assert "cycle 50" in by_name["Cycles 2-100 all present"]["detail"]
    assert len(kept) == 118


def test_gate_fails_short_test():
    _, report = check_cycles(make_cycles(n=60))
    assert not report[-1]["passed"]


# --- Features ------------------------------------------------------------------


def test_delta_q_of_straight_line_curves():
    """Q(V) is a straight line, so ΔQ(V) = (Q100 - Q10) * (3.6 - V) / 1.6
    and its variance is known exactly."""
    grid, dq = delta_q_curve(make_cycles())
    assert len(grid) == V_POINTS
    assert grid[0] == pytest.approx(V_TOP)
    assert grid[-1] == pytest.approx(V_BOTTOM)
    change_ah = (capacity_mah(100) - capacity_mah(10)) / 1000
    expected = change_ah**2 * np.var(np.linspace(0, 1, V_POINTS))
    assert np.var(dq) == pytest.approx(expected, rel=1e-3)


def test_features_from_cycles():
    f = features_from_cycles(make_cycles())
    # Ah per cycle; the rolling median flattens the two ends a little
    assert f["fade_slope"] == pytest.approx(-0.0005, rel=1e-2)
    assert f["qd_cycle2"] == pytest.approx(capacity_mah(3) / 1000)  # despiked
    assert f["charge_time_2to6"] == pytest.approx(10.0)  # minutes
    for name in ("temp_max", "ir_min", "ir_change_2to100"):
        assert f[name] is None
    for name in REDUCED_FEATURES:
        assert f[name] is not None, name


def test_optional_ir_and_temperature():
    ir = np.full(100, 0.02)
    ir[:50] = 0.021
    f = features_from_cycles(make_cycles(), ir=ir, temp=np.full(100, 31.5))
    assert f["temp_max"] == 31.5
    assert f["ir_min"] == pytest.approx(0.02)
    assert f["ir_change_2to100"] == pytest.approx(-0.001)


def test_one_cycle_spike_is_ignored():
    clean = features_from_cycles(make_cycles())
    cycles = make_cycles()
    cycles[49]["discharge"]["capacity_mah"][-1] = 5000
    spiky = features_from_cycles(cycles)
    assert spiky["qd_max_minus_cycle2"] == pytest.approx(clean["qd_max_minus_cycle2"])


def test_no_charge_time_without_time_column():
    cycles = make_cycles()
    for cycle in cycles:
        cycle["charge"]["time_s"] = None
    assert features_from_cycles(cycles)["charge_time_2to6"] is None


def test_too_few_cycles():
    with pytest.raises(FeatureError, match="Cycle 61"):
        features_from_cycles(make_cycles(n=60))


def test_flat_delta_q_gives_none_not_inf():
    cycles = make_cycles()
    cycles[99]["discharge"] = cycles[9]["discharge"]
    f = features_from_cycles(cycles)
    assert f["log_var_dq"] is None


def test_severson_cell_matches_mpr_path():
    """The same cell as Severson-style arrays gives the same features, so a
    model trained on that data scores .mpr cells on the same scale."""
    cycles = make_cycles(power=0.7)  # curved, so ΔQ(V) has a clear skew
    severson_grid = np.linspace(V_TOP, V_BOTTOM, V_POINTS)
    cell = {"cycles": {}, "summary": {}}
    for i, cycle in enumerate(cycles):
        q = np.asarray(cycle["discharge"]["capacity_mah"]) / 1000
        v = np.asarray(cycle["discharge"]["voltage_v"])
        cell["cycles"][str(i)] = {"Qdlin": np.interp(severson_grid, v[::-1], q[::-1])}
    cell["summary"] = {
        "QD": [capacity_mah(k) / 1000 for k in range(1, 101)],
        "chargetime": [10.0] * 100,
        "IR": [0.02] * 100,
        "Tmax": [31.5] * 100,
    }
    from_mpr = features_from_cycles(cycles, ir=[0.02] * 100, temp=[31.5] * 100)
    from_severson = features_from_severson_cell(cell)
    for name in FEATURE_COLUMNS:
        a, b = from_mpr[name], from_severson[name]
        assert math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-9), name


# --- What gets stored ----------------------------------------------------------


def test_analysis_marks_gate_failures_and_keeps_every_cycle():
    cycles = make_cycles(n=120)
    cycles[109]["discharge"]["capacity_mah"][-1] = 10  # efficiency far too low
    analysis = analyse_cycles(cycles)
    assert len(analysis["summary"]) == 120
    failed = []
    for row in analysis["summary"]:
        if not row["passed_gate"]:
            failed.append(row["cycle_no"])
    assert failed == [110]
    assert analysis["features"]["fade_slope"] < 0
    assert analysis["error"] is None


def test_analysis_of_a_short_test_gives_the_reason():
    analysis = analyse_cycles(make_cycles(n=60))
    assert analysis["features"] is None
    assert "Cycle 61" in analysis["error"]
    assert len(analysis["summary"]) == 60


def test_file_outside_the_upload_folder_is_not_read():
    analysis = analyse_file("/etc/passwd")
    assert analysis["summary"] == []
    assert not analysis["gate"][0]["passed"]
    assert "outside the upload folder" in analysis["error"]
