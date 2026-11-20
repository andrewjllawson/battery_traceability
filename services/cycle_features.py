"""Cycle summaries, a quality gate and early-life features for cycle-life models.

The features follow Severson et al. (2019), as in the battery-cycle-life
research code (src/features.py). One function, early_features(), does the
maths for both sources: a BioLogic .mpr file (features_from_cycles) and a cell
from the public Severson dataset (features_from_severson_cell), so a model
trained on one can score the other.

Units match the Severson data: capacity in Ah, charge time in minutes.
Cycle N is the N-th full cycle from group_full_cycles(), counting from 1.

analyse_file() and save_analysis() fill tbl_cycle_summary and tbl_cell_features
(scripts/extract_cycle_features.py); everything above them needs no database.
"""

import numpy as np
import pandas as pd
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor, execute_values
from scipy.stats import kurtosis, skew

from core.db import get_connection
from services.cycling import group_full_cycles, read_half_cycles, stored_file

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


# --- Filling the tables --------------------------------------------------------

SUMMARY_COLUMNS = (
    "cycle_no",
    "discharge_mah",
    "charge_mah",
    "coulombic_eff",
    "charge_time_s",
)


def analyse_cycles(cycles):
    """Everything stored for one file, from its full cycles: {"summary": rows
    with passed_gate, "gate": report, "features": dict or None, "error": why
    there are no features, or None, "n_cycles"}.
    """
    kept, gate = check_cycles(cycles)
    passed = set()
    for cycle in kept:
        passed.add(cycle["cycle"])
    summary = cycle_summary(cycles)
    for row in summary:
        row["passed_gate"] = row["cycle_no"] in passed

    features, error = None, None
    try:
        features = features_from_cycles(kept)
    except FeatureError as e:
        error = str(e)
    return {
        "summary": summary,
        "gate": gate,
        "features": features,
        "error": error,
        "n_cycles": len(cycles),
    }


def analyse_file(file_path):
    """analyse_cycles() for a stored .mpr file. A file that cannot be read gives a
    one-line failed gate and no cycles."""
    path = stored_file(file_path)
    try:
        if path is None:
            raise ValueError("The stored path is outside the upload folder.")
        halves = read_half_cycles(path, max_points=None, with_time=True)
    except Exception as e:  # unreadable, missing, or not a BioLogic file
        step = gate_step("Readable .mpr file", None, 0, False, str(e))
        return {
            "summary": [],
            "gate": [step],
            "features": None,
            "error": f"Could not read the file: {e}",
            "n_cycles": 0,
        }
    analysis = analyse_cycles(group_full_cycles(halves))
    cycles = analysis["n_cycles"]
    analysis["gate"].insert(0, gate_step("Readable .mpr file", cycles, cycles, True))
    return analysis


def cycling_files(record_id=None):
    """(cycling_id, record_id, file_path) of every stored file, or one record's."""
    query = "SELECT cycling_id, record_id, file_path FROM tbl_cycling_data"
    values = []
    if record_id:
        query += " WHERE record_id = %s"
        values.append(record_id)
    conn = get_connection(read_only=True)
    cur = conn.cursor()
    cur.execute(query + " ORDER BY cycling_id", values)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def save_analysis(cycling_id, record_id, analysis, conn=None):
    """Replace a file's rows in tbl_cycle_summary and tbl_cell_features. Commits,
    unless conn is given (then the caller commits or rolls back)."""
    summary_rows = []
    for row in analysis["summary"]:
        values = [cycling_id, record_id]
        for name in SUMMARY_COLUMNS:
            values.append(row[name])
        values.append(row["passed_gate"])
        summary_rows.append(tuple(values))

    features = analysis["features"] or {}
    feature_values = []
    for name in REDUCED_FEATURES:
        feature_values.append(features.get(name))
    grid = (features.get("v_grid_max"), features.get("v_grid_min"))
    columns = [
        "cycling_id",
        "record_id",
        "n_cycles",
        *REDUCED_FEATURES,
        "v_grid_max",
        "v_grid_min",
        "gate",
        "feature_error",
    ]
    values = (
        cycling_id,
        record_id,
        analysis["n_cycles"],
        *feature_values,
        *grid,
        Json(analysis["gate"]),
        analysis["error"],
    )
    updates = []
    for name in columns[1:]:
        updates.append(sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(name)))
    upsert = sql.SQL(
        "INSERT INTO tbl_cell_features ({}) VALUES ({}) "
        "ON CONFLICT (cycling_id) DO UPDATE SET {}, computed_at = now()"
    ).format(
        sql.SQL(", ").join(sql.Identifier(name) for name in columns),
        sql.SQL(", ").join(sql.Placeholder() for _ in columns),
        sql.SQL(", ").join(updates),
    )

    own = conn is None
    if own:
        conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM tbl_cycle_summary WHERE cycling_id = %s", (cycling_id,)
        )
        if summary_rows:
            execute_values(
                cur,
                "INSERT INTO tbl_cycle_summary (cycling_id, record_id, cycle_no, "
                "discharge_mah, charge_mah, coulombic_eff, charge_time_s, "
                "passed_gate) VALUES %s",
                summary_rows,
            )
        cur.execute(upsert, values)
        if own:
            conn.commit()
    finally:
        cur.close()
        if own:
            conn.close()


def analyse_and_save(cycling_id, record_id, file_path):
    """Analyse one stored file and save the result. Returns (number of cycles,
    number that passed the gate, "features saved" or why there are none)."""
    analysis = analyse_file(file_path)
    save_analysis(cycling_id, record_id, analysis)
    kept = 0
    for row in analysis["summary"]:
        kept += row["passed_gate"]
    result = "features saved" if analysis["features"] else analysis["error"]
    return analysis["n_cycles"], kept, result


def load_fade_cells(record_ids=None, conn=None):
    """Gate-passed discharge capacity of each record's newest file, for the fade
    model: {record_id: (cycle numbers, capacities in mAh)}. record_ids=None
    loads every record."""
    query = (
        "SELECT s.record_id, s.cycle_no, s.discharge_mah FROM tbl_cycle_summary s "
        "JOIN (SELECT DISTINCT ON (record_id) record_id, cycling_id "
        "FROM tbl_cycling_data ORDER BY record_id, uploaded_at DESC, cycling_id DESC)"
        " newest ON newest.cycling_id = s.cycling_id "
        "WHERE s.passed_gate"
    )
    values = []
    if record_ids is not None:
        query += " AND s.record_id = ANY(%s)"
        values.append(list(record_ids))
    own = conn is None
    if own:
        conn = get_connection(read_only=True)
    cur = conn.cursor()
    try:
        cur.execute(query + " ORDER BY s.record_id, s.cycle_no", values)
        rows = cur.fetchall()
    finally:
        cur.close()
        if own:
            conn.close()

    cells = {}
    for record_id, cycle_no, capacity in rows:
        numbers, capacities = cells.setdefault(record_id, ([], []))
        numbers.append(cycle_no)
        capacities.append(capacity)
    return cells


def stored_analysis(record_id):
    """What extract_cycle_features stored for a record's newest file: the file,
    its gate report, its features and its cycles (one list per column). None if
    the record has no file; "features" is missing if the file is not extracted.
    """
    conn = get_connection(read_only=True)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            "SELECT cycling_id, filename, uploaded_at FROM tbl_cycling_data "
            "WHERE record_id = %s ORDER BY uploaded_at DESC, cycling_id DESC LIMIT 1",
            (record_id,),
        )
        newest = cur.fetchone()
        if newest is None:
            return None
        result = {
            "record_id": record_id,
            "cycling_id": newest["cycling_id"],
            "filename": newest["filename"],
            "uploaded_at": newest["uploaded_at"].isoformat(),
            "extracted": False,
        }
        # Column names come from REDUCED_FEATURES in this file, never from a request
        columns = [
            "n_cycles",
            *REDUCED_FEATURES,
            "v_grid_max",
            "v_grid_min",
            "gate",
            "feature_error",
            "computed_at",
        ]
        cur.execute(
            sql.SQL("SELECT {} FROM tbl_cell_features WHERE cycling_id = %s").format(
                sql.SQL(", ").join(sql.Identifier(name) for name in columns)
            ),
            (newest["cycling_id"],),
        )
        stored = cur.fetchone()
        if stored is None:
            return result

        cur.execute(
            "SELECT cycle_no, discharge_mah, charge_mah, coulombic_eff, "
            "charge_time_s, passed_gate FROM tbl_cycle_summary "
            "WHERE cycling_id = %s ORDER BY cycle_no",
            (newest["cycling_id"],),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    features = None
    if stored["feature_error"] is None:
        features = {}
        for name in REDUCED_FEATURES:
            features[name] = stored[name]
    cycles = {}
    for name in (*SUMMARY_COLUMNS, "passed_gate"):
        cycles[name] = []
    for row in rows:
        for name in cycles:
            cycles[name].append(row[name])
    result.update(
        {
            "extracted": True,
            "computed_at": stored["computed_at"].isoformat(),
            "n_cycles": stored["n_cycles"],
            "gate": stored["gate"],
            "features": features,
            "feature_error": stored["feature_error"],
            "delta_q_voltage_range_v": [stored["v_grid_max"], stored["v_grid_min"]],
            "units": {
                "capacity": "mAh (features: Ah)",
                "charge_time": "s (features: min)",
            },
            "cycles": cycles,
        }
    )
    return result


def list_cycling_cells():
    """Every record with a cycling file, newest file first per record: record_id,
    cycling_id, filename, uploaded_at, n_cycles, extracted and has_features.
    Callers filter by project (record_visible)."""
    conn = get_connection(read_only=True)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            "SELECT DISTINCT ON (d.record_id) d.record_id, d.cycling_id, d.filename, "
            "d.uploaded_at, f.n_cycles, f.cycling_id IS NOT NULL AS extracted, "
            "f.cycling_id IS NOT NULL AND f.feature_error IS NULL AS has_features "
            "FROM tbl_cycling_data d "
            "LEFT JOIN tbl_cell_features f ON f.cycling_id = d.cycling_id "
            "ORDER BY d.record_id, d.uploaded_at DESC, d.cycling_id DESC"
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()
    cells = []
    for row in rows:
        cell = dict(row)
        cell["uploaded_at"] = row["uploaded_at"].isoformat()
        cells.append(cell)
    return cells


MAX_CURVE_POINTS = 200  # keeps the ΔQ(V) JSON small


def delta_q_for_file(file_path):
    """The ΔQ(V) curve (cycle 100 minus cycle 10) of a stored .mpr file, thinned
    for a chart: {"voltage_v", "delta_q_mah"}. Raises FeatureError if the file
    cannot give it."""
    path = stored_file(file_path)
    if path is None:
        raise FeatureError("The stored path is outside the upload folder.")
    try:
        halves = read_half_cycles(path, max_points=None)
    except Exception as e:  # unreadable, missing, or not a BioLogic file
        raise FeatureError(f"Could not read the file: {e}") from None
    kept, _ = check_cycles(group_full_cycles(halves))
    grid, dq = delta_q_curve(kept)
    pick = np.linspace(0, len(grid) - 1, MAX_CURVE_POINTS).astype(int)
    return {
        "early_cycle": EARLY_CYCLE,
        "late_cycle": LATE_CYCLE,
        "voltage_v": grid[pick].round(4).tolist(),
        "delta_q_mah": (1000 * dq[pick]).round(5).tolist(),
    }
