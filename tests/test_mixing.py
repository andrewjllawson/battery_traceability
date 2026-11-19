"""Mixer log parsing and stage detection (services/mixing.py). No database needed."""

import io
import sys
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.mixing import (
    MixingError,
    check_mixer_log,
    clean_ingredients,
    find_stages,
    parse_mixer_xlsx,
    run_visible,
)


def make_xlsx(rotor, pan=None, drop_clock=False) -> bytes:
    """A small log: one row per second starting 09:00:00."""
    n = len(rotor)
    rows = {
        "Date [yyyy-MM-dd]": [datetime(2024, 4, 9)] * n,
        "Time [hh:mm:ss]": [time(9, s // 60, s % 60) for s in range(n)],
        "RotorRev [1/min]": rotor,
        "RotorPower [W]": [r / 100 for r in rotor],
        "MixingPanRev [1/min]": pan or [85 if r else 0 for r in rotor],
        "MixingPanPower [W]": [0] * n,
        "Temperature [°C]": [25] * n,
        "Total Power": [r / 50 for r in rotor],
        "Cumulative Energy (J)": list(range(n)),
    }
    if drop_clock:
        rows["Time [hh:mm:ss]"][-1] = rows["Time [hh:mm:ss]"][-3]  # clock goes back
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False)
    return buf.getvalue()


def test_two_stages_and_pause():
    rotor = [3500] * 20 + [0] * 10 + [2400] * 30
    summary, log = parse_mixer_xlsx(make_xlsx(rotor))
    stages = find_stages(log)
    assert list(stages["stage_no"]) == [1, 2]
    assert list(stages["duration_s"]) == [20, 30]
    assert stages.loc[1, "pause_before_s"] == 10
    assert stages.loc[0, "rotor_rpm"] == 3500
    assert summary["duration_s"] == 59


def test_short_blip_is_not_a_stage():
    rotor = [0] * 10 + [3000] * 2 + [0] * 10 + [3000] * 20
    _, log = parse_mixer_xlsx(make_xlsx(rotor))
    assert len(find_stages(log)) == 1


def test_clock_going_back_drops_the_row():
    rotor = [3500] * 30
    _, log = parse_mixer_xlsx(make_xlsx(rotor, drop_clock=True))
    assert log["t_s"].is_monotonic_increasing
    assert len(log) == 29


def test_missing_columns_rejected():
    buf = io.BytesIO()
    pd.DataFrame({"a": [1, 2]}).to_excel(buf, index=False)
    with pytest.raises(MixingError):
        parse_mixer_xlsx(buf.getvalue())


def test_gate_matches_the_parser():
    """The gate report follows the rows parse_mixer_xlsx() and find_stages() keep."""
    rotor = [3500] * 20 + [0] * 10 + [2400] * 30
    data = make_xlsx(rotor, drop_clock=True)
    report = check_mixer_log(data)
    _, log = parse_mixer_xlsx(data)
    by_name = {step["check"]: step for step in report}
    assert all(step["passed"] for step in report)
    assert by_name["Time moves forward"]["rows_in"] == 60
    assert by_name["Time moves forward"]["rows_out"] == len(log) == 59
    assert by_name["Rotor-on stages found"]["rows_out"] == 49  # 20 s + 29 s
    assert by_name["Rotor-on stages found"]["detail"] == "2 stages"


def test_gate_stops_at_missing_columns():
    buf = io.BytesIO()
    pd.DataFrame({"a": [1, 2]}).to_excel(buf, index=False)
    report = check_mixer_log(buf.getvalue())
    assert [step["check"] for step in report] == [
        "Readable workbook",
        "Required columns",
    ]
    assert not report[-1]["passed"]


def test_gate_rejects_a_non_workbook():
    report = check_mixer_log(b"not a spreadsheet")
    assert len(report) == 1 and not report[0]["passed"]


def test_ingredients_are_cleaned():
    rows = [("CAT-NMC-001", "Binder", " 12.5 "), ("", "", ""), ("AN-GR-001", "", "3")]
    cleaned, error = clean_ingredients(rows)
    assert error is None
    assert cleaned == [("CAT-NMC-001", "Binder", 12.5), ("AN-GR-001", None, 3.0)]


@pytest.mark.parametrize(
    "rows",
    [
        [("", "Binder", "5")],  # no material
        [("A", "", "abc")],  # mass not a number
        [("A", "", "0")],  # mass not above zero
        [("A", "Glue", "5")],  # role not in the list
        [("A", "", "5"), ("A", "", "6")],  # listed twice
    ],
)
def test_bad_ingredients_rejected(rows):
    cleaned, error = clean_ingredients(rows)
    assert cleaned is None and error


def test_run_visibility():
    assert run_visible("LEAP", None)  # admin
    assert run_visible("LEAP", ["LEAP"])
    assert not run_visible("FAST", ["LEAP"])
    assert run_visible(None, ["LEAP"])  # no project: visible to everyone


def test_mixing_result_in_the_shared_shape():
    from services.model_registry import mixing_to_standard

    result = {
        "runs": 2,
        "stages": 6,
        "seconds": 900,
        "r2": 0.91,
        "mae_w": 120.0,
        "baseline_r2": 0.0,
        "baseline_mae_w": 600.0,
        "importance": [{"feature": "Rotor speed", "value": 0.5}],
        "scatter": {"actual": [1.0], "predicted": [1.1]},
        "per_run": {},
    }
    shared = mixing_to_standard(result)
    assert shared["data_type"] == "mixing"
    assert shared["metric"] == 120.0 and shared["baseline_metric"] == 600.0
    assert shared["n_groups"] == 6 and shared["n_test"] == 900
    assert shared["n_train"] is None  # leave-one-stage-out has no fixed training set
    assert shared["importance"] == result["importance"]
