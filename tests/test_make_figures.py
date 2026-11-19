"""The figure script (scripts/make_figures.py). No database or real data needed:
every figure is drawn from made-up inputs into a temporary folder.

pytest -q tests/test_make_figures.py
"""

import io
import sys
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import make_figures  # noqa: E402
from services import fade_model  # noqa: E402
from services.mixing import check_mixer_log  # noqa: E402


def mixer_xlsx(rotor) -> bytes:
    """A small mixer log: one row per second starting 09:00:00."""
    n = len(rotor)
    times = []
    for s in range(n):
        times.append(time(9, s // 60, s % 60))
    rows = {
        "Date [yyyy-MM-dd]": [datetime(2024, 4, 9)] * n,
        "Time [hh:mm:ss]": times,
        "RotorRev [1/min]": rotor,
        "MixingPanRev [1/min]": [85] * n,
        "Total Power": [r / 50 for r in rotor],
    }
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False)
    return buf.getvalue()


def fading_cells():
    """Made-up cells that pass 80% of early capacity, with a little noise."""
    rng = np.random.default_rng(0)
    cycles = np.arange(1, 301)
    cells = {}
    for i in range(4):
        q = 2.0 - (0.0015 + 0.0005 * i) * cycles
        noise = 1 + rng.normal(0, 0.002, len(cycles))
        cells[f"CELL-{i}"] = (list(cycles), list(q * noise))
    return cells


def registry_runs(cells):
    runs = fade_model.score_cells(cells)
    for run_id, run in enumerate(runs, start=1):
        run["run_id"] = run_id
        run["data_source"] = "made up"
    return runs


def assert_saved(out, name):
    assert (out / f"{name}.png").stat().st_size > 0
    assert (out / f"{name}.svg").stat().st_size > 0


def quantities(numbers, name):
    found = []
    for row in numbers:
        if row["figure"] == name:
            found.append(row["quantity"])
    return found


def test_mixing_gate_figure(tmp_path):
    reports = {
        "good.xlsx": check_mixer_log(mixer_xlsx([3500] * 20 + [0] * 10 + [2400] * 30)),
        "bad.xlsx": check_mixer_log(b"not a spreadsheet"),
    }
    numbers = []
    make_figures.mixing_gate(reports, "made up", numbers, tmp_path)
    name = "mixing_validation_gate_rows_in_out"
    assert_saved(tmp_path, name)
    by_quantity = {}
    for row in numbers:
        by_quantity[row["quantity"]] = row["value"]
    assert by_quantity["Mixer logs checked"] == "2"
    assert by_quantity["Readable workbook: logs failed"] == "1"
    assert by_quantity["Rotor-on stages found: rows out"] == "50"


def test_mixing_model_figures(tmp_path):
    actual = list(np.linspace(100, 4000, 50))
    run = {
        "run_id": 7,
        "data_source": "made up",
        "n_test": 900,
        "n_groups": 6,
        "metric": 120.0,
        "baseline_metric": 600.0,
        "r2": 0.9,
        "baseline_r2": 0.0,
        "scatter": {"actual": actual, "predicted": actual},
    }
    numbers = []
    make_figures.mixing_scatter(run, numbers, tmp_path)
    make_figures.mixing_error(run, numbers, tmp_path)
    assert_saved(tmp_path, "mixing_model_vs_baseline_predicted_vs_measured")
    assert_saved(tmp_path, "mixing_model_vs_baseline_error")
    runs = set()
    for row in numbers:
        runs.add(row["run"])
    assert runs == {"7"}  # every number traces back to the run


def test_cycling_figures(tmp_path):
    cells = fading_cells()
    runs = registry_runs(cells)
    numbers = []
    make_figures.cycling_comparison(runs, numbers, tmp_path)
    make_figures.cycling_error_vs_fraction(runs, numbers, tmp_path)
    make_figures.cycling_example(
        "CELL-0", *cells["CELL-0"], "made up", numbers, tmp_path
    )
    make_figures.cycling_backtest(cells, "made up", numbers, tmp_path)
    make_figures.cycling_time_saved(cells, "made up", numbers, tmp_path)
    make_figures.cycling_fade_curves(cells, "made up", numbers, tmp_path)
    grid = np.linspace(4.2, 3.0, 100)
    curves = {"CELL-0": (grid, -0.001 * (4.2 - grid) ** 2, "made up")}
    make_figures.cycling_delta_q(curves, numbers, tmp_path)
    for name in (
        "cycling_model_comparison_forecast_error",
        "cycling_forecast_error_vs_share_of_test_used",
        "cycling_example_fade_forecast",
        "cycling_predicted_vs_measured_end_of_life",
        "cycling_cycler_time_saved_vs_dangerous_stops",
        "cycling_capacity_fade_curves",
        "cycling_delta_q_curves_cycle100_minus_cycle10",
    ):
        assert_saved(tmp_path, name)
    comparison = quantities(numbers, "cycling_model_comparison_forecast_error")
    assert f"{fade_model.BASELINE}: {fade_model.METRIC_NAME}" in comparison


def test_stopping_points_count_dangerous_stops():
    """Fade that speeds up after cycle 150: fits on the first 30% cannot see it,
    so stopping there is dangerous for every cell."""
    cycles = np.arange(1, 301)
    q = 2.0 - 0.001 * cycles - 0.0002 * np.clip(cycles - 150, 0, None) ** 1.5
    points = make_figures.stopping_points({"CELL": (list(cycles), list(q))})
    first = points[0]
    assert first[0] == fade_model.CUTS[0]
    assert first[1] > 0  # cycler time saved
    assert first[2] == 100  # dangerous stops (% of cells)


def test_figures_need_cells_that_reached_end_of_life(tmp_path):
    cycles = list(range(1, 101))
    cells = {"CELL": (cycles, [2.0 - 0.0001 * k for k in cycles])}
    with pytest.raises(ValueError, match="80%"):
        make_figures.cycling_backtest(cells, "made up", [], tmp_path)
    with pytest.raises(ValueError, match="80%"):
        make_figures.cycling_time_saved(cells, "made up", [], tmp_path)


def test_numbers_file(tmp_path):
    numbers = []
    make_figures.note(numbers, "fig", "A value", 1.23456789, "input|file", 3)
    make_figures.note(numbers, "fig", "Missing", None, "input")
    make_figures.write_numbers(numbers, [("g", "no data")], tmp_path)
    text = (tmp_path / "numbers.md").read_text()
    assert "| fig | A value | 1.235 | input/file | 3 |" in text
    assert "| fig | Missing | - | input |  |" in text
    assert "- `g`: no data" in text


def test_missing_inputs_are_skipped_not_fatal(tmp_path, monkeypatch):
    """With no logs, no runs and no cells, every figure is skipped with a reason."""
    monkeypatch.setattr(make_figures, "MIXING_DIR", tmp_path / "no-such-folder")
    monkeypatch.setattr(make_figures, "list_runs", lambda *a, **k: [])
    monkeypatch.setattr(make_figures, "load_fade_cells", lambda: {})
    monkeypatch.setattr(make_figures, "cycling_files", lambda: [])
    out = tmp_path / "figures"
    make_figures.main(out)
    text = (out / "numbers.md").read_text()
    assert "## Skipped" in text
    for name in ("`a`", "`b`", "`c, e`", "`d, f, g fade`", "`g ΔQ(V)`"):
        assert name in text
    assert not list(out.glob("*.png"))
