"""The capacity fade forecast (services/fade_model.py). No database or real data
needed; the cells are made-up fade curves with known answers.

pytest -q tests/test_fade_model.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.fade_model import (  # noqa: E402
    BASELINE,
    CUTS,
    EXPONENTIAL,
    FORMS,
    LINEAR,
    SQUARE_ROOT,
    FadeError,
    end_of_life_cycle,
    fit_form,
    forecast_cell,
    score_cell,
    score_cells,
)

CYCLES = np.arange(1, 201)


def linear_cell(slope=-0.002, start=2.0):
    return CYCLES, start + slope * CYCLES


def test_linear_cell_is_forecast_exactly():
    n, q = linear_cell()
    result = forecast_cell(n, q)
    assert result["model"] == LINEAR
    # Early capacity is cycle 1 (1.998 mAh); 80% of it is reached at cycle 200.8
    assert result["eol_cycle"] == pytest.approx(200.8, rel=1e-6)
    assert result["eol_low"] == pytest.approx(200.8, rel=1e-6)
    assert result["eol_high"] == pytest.approx(200.8, rel=1e-6)
    assert result["measured_eol_cycle"] is None
    assert not result["extrapolated"]
    assert result["scores_mape"][LINEAR] < 1e-6


def test_square_root_cell_picks_square_root():
    q = 2.0 - 0.05 * np.sqrt(CYCLES)
    assert forecast_cell(CYCLES, q)["model"] == SQUARE_ROOT


def test_exponential_cell_picks_exponential():
    q = 2.0 * np.exp(-0.002 * CYCLES)
    assert forecast_cell(CYCLES, q)["model"] == EXPONENTIAL


def test_baseline_loses_on_a_fading_cell():
    n, q = linear_cell()
    scores = forecast_cell(n, q)["scores_mape"]
    assert scores[BASELINE] > 10 * scores[LINEAR]


def test_later_cycles_are_never_seen_by_the_fit():
    """Changing the cycles after a cut leaves that cut's predictions unchanged."""
    n, q = linear_cell()
    changed = q.copy()
    k = int(round(len(n) * CUTS[0]))
    changed[k:] = changed[k:] * 0.5
    before = score_cell(n, q)
    after = score_cell(n, changed)
    for a, b in zip(before, after, strict=True):
        if a["cut"] == CUTS[0]:
            np.testing.assert_allclose(a["predicted"], b["predicted"])


def test_noise_gives_a_band_around_the_forecast():
    rng = np.random.default_rng(0)
    n, q = linear_cell()
    q = q * (1 + rng.normal(0, 0.005, len(q)))
    result = forecast_cell(n, q)
    assert result["eol_low"] < result["eol_cycle"] < result["eol_high"]


def test_measured_end_of_life_is_reported():
    n, q = linear_cell(slope=-0.004)  # falls below 80% near cycle 100
    result = forecast_cell(n, q)
    assert result["measured_eol_cycle"] == 101


def test_cell_that_does_not_fade_is_flagged():
    q = np.full(len(CYCLES), 2.0) + 0.0001 * CYCLES
    result = forecast_cell(CYCLES, q)
    assert result["eol_cycle"] is None
    assert result["extrapolated"]


def test_exactly_flat_cell_is_not_fading():
    """Rounding gives a flat cell a tiny negative slope; that must not become an
    end of life at some absurd cycle number."""
    result = forecast_cell(CYCLES, np.full(len(CYCLES), 2.0))
    assert result["eol_cycle"] is None
    assert result["eol_low"] is None and result["eol_high"] is None
    assert result["extrapolated"]


def test_far_forecast_is_flagged_as_extrapolation():
    n, q = linear_cell(slope=-0.0001)  # 80% at about cycle 4000
    result = forecast_cell(n, q)
    assert result["eol_cycle"] > 3 * 200
    assert result["extrapolated"]


def test_missing_values_are_dropped():
    n, q = linear_cell()
    q = list(q)
    q[50] = None
    q[60] = float("nan")
    q[70] = 0
    result = forecast_cell(n, q)
    assert result["n_cycles"] == 197
    assert result["model"] == LINEAR


def test_too_few_cycles():
    with pytest.raises(FadeError, match="at least 15"):
        forecast_cell(range(1, 13), np.linspace(2, 1.9, 12))


def test_end_of_life_of_each_form():
    n, q = linear_cell()
    for form in FORMS:
        params = fit_form(form, n, q)
        cycle = end_of_life_cycle(form, params, 1.7)
        assert 140 < cycle < 160, form  # the linear cell reaches 1.7 mAh at 150


def test_score_cells_gives_one_shared_result_per_model():
    cells = {"CELL-A": linear_cell(), "CELL-B": linear_cell(slope=-0.003)}
    results = score_cells(cells)
    assert [r["model"] for r in results] == [*FORMS, BASELINE]
    keys = {
        "data_type",
        "model",
        "target",
        "unit",
        "split_method",
        "n_groups",
        "n_train",
        "n_test",
        "metric_name",
        "metric",
        "baseline_metric",
        "r2",
        "baseline_r2",
        "importance",
        "scatter",
        "note",
    }
    for result in results:
        assert keys <= set(result)
        assert result["n_groups"] == 2
        assert result["baseline_metric"] == results[-1]["metric"]
    linear = results[0]
    assert linear["metric"] < linear["baseline_metric"]
    assert linear["r2"] == pytest.approx(1.0)
    assert set(linear["metrics"]["mape_by_cell"]) == {"CELL-A", "CELL-B"}


def test_score_cells_skips_short_cells():
    cells = {"LONG": linear_cell(), "SHORT": (range(1, 6), [2, 2, 2, 2, 2])}
    assert score_cells(cells)[0]["n_groups"] == 1
    with pytest.raises(FadeError):
        score_cells({"SHORT": (range(1, 6), [2, 2, 2, 2, 2])})
