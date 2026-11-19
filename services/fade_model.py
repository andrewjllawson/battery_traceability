"""A capacity fade forecast for cells, from the cycles already in the database.

Question it answers: if a test stopped now, when would the cell reach 80% of its
early capacity? Three simple curves are fitted to discharge capacity against cycle
number. Each is scored by fitting on the first part of a test and predicting the
cycles after it, so a fit never sees the cycles it is scored on. The result is
compared with assuming the capacity stops fading.

Needs no database: callers pass cycle numbers and capacities in mAh (the cycles of
tbl_cycle_summary that passed the quality gate).
"""

import numpy as np

CUTS = (0.3, 0.5, 0.7)  # fraction of a test's cycles used for fitting
MAIN_CUT = 0.5  # the cut reported as the headline metric
MIN_FIT_CYCLES = 10
MIN_TEST_CYCLES = 5
END_OF_LIFE = 0.8  # fraction of early capacity
EARLY_CYCLES = 10  # early capacity is the best of the first 10 cycles
MIN_FADE = 0.001  # a fit falling less than 0.1% of early capacity is not fading
BASELINE_WINDOW = 5  # the baseline holds the median of the last 5 fitted cycles
MAX_EXTRAPOLATION = 3  # a forecast beyond 3x the last cycle is flagged
BAND = (10, 90)  # percentiles of the held-out errors: an 80% band
MAX_SCATTER_POINTS = 800
CURVE_POINTS = 200

LINEAR = "Linear fade"
SQUARE_ROOT = "Square-root fade"
EXPONENTIAL = "Exponential fade"
FORMS = (LINEAR, SQUARE_ROOT, EXPONENTIAL)
BASELINE = "No further fade"

TARGET = "Discharge capacity of later cycles"
SPLIT_METHOD = "Fit on the first 30/50/70% of each test's cycles, score the rest"
METRIC_NAME = "MAPE (%) at the 50% cut"


class FadeError(ValueError):
    """Too few usable cycles; the message is shown to the user."""


# --- Fitting -------------------------------------------------------------------


def clean_cycles(cycle_no, capacity_mah):
    """Sorted arrays of cycle number and capacity, without missing or zero values."""
    pairs = []
    for n, q in zip(cycle_no, capacity_mah, strict=True):
        if n is None or q is None:
            continue
        if np.isfinite(q) and q > 0:
            pairs.append((float(n), float(q)))
    pairs.sort()
    n = np.array([p[0] for p in pairs])
    q = np.array([p[1] for p in pairs])
    return n, q


def x_of(form, n):
    """The x the form is a straight line in: the square root for square-root fade."""
    n = np.asarray(n, dtype=float)
    return np.sqrt(n) if form == SQUARE_ROOT else n


def fit_form(form, n, q):
    """(slope, intercept) of the straight line behind the form. Exponential fade is
    a straight line in log(capacity)."""
    y = np.log(q) if form == EXPONENTIAL else q
    slope, intercept = np.polyfit(x_of(form, n), y, 1)
    return float(slope), float(intercept)


def predict_form(form, params, n):
    slope, intercept = params
    y = intercept + slope * x_of(form, n)
    return np.exp(y) if form == EXPONENTIAL else y


def end_of_life_cycle(form, params, threshold_mah):
    """Cycle at which the fitted curve falls to threshold_mah. None if the curve
    does not fall."""
    slope, intercept = params
    if slope >= 0 or threshold_mah <= 0:
        return None
    target = np.log(threshold_mah) if form == EXPONENTIAL else threshold_mah
    x = max((target - intercept) / slope, 0.0)
    return float(x**2) if form == SQUARE_ROOT else float(x)


def baseline_level(q_fit):
    return float(np.median(q_fit[-BASELINE_WINDOW:]))


def percent_errors(actual, predicted):
    """Signed % error per cycle (+ = capacity over-predicted)."""
    return 100 * (predicted - actual) / actual


# --- Scoring -------------------------------------------------------------------


def score_cell(n, q):
    """Fit on the first part of the test and predict the rest, at each cut.

    Returns one row per cut and model: {"cut", "model", "n_train", "n_test",
    "mape", "errors_pct", "actual", "predicted"}. Cuts that leave too few cycles
    on either side are skipped.
    """
    rows = []
    for cut in CUTS:
        k = int(round(len(n) * cut))
        if k < MIN_FIT_CYCLES or len(n) - k < MIN_TEST_CYCLES:
            continue
        n_fit, q_fit = n[:k], q[:k]
        n_test, q_test = n[k:], q[k:]
        predictions = {BASELINE: np.full(len(q_test), baseline_level(q_fit))}
        for form in FORMS:
            predictions[form] = predict_form(form, fit_form(form, n_fit, q_fit), n_test)
        for model, predicted in predictions.items():
            errors = percent_errors(q_test, predicted)
            rows.append(
                {
                    "cut": cut,
                    "model": model,
                    "n_train": k,
                    "n_test": len(q_test),
                    "mape": float(np.mean(np.abs(errors))),
                    "errors_pct": errors,
                    "actual": q_test,
                    "predicted": predicted,
                }
            )
    return rows


def mean_mape(rows, model):
    values = []
    for row in rows:
        if row["model"] == model:
            values.append(row["mape"])
    return float(np.mean(values)) if values else None


# --- One cell ------------------------------------------------------------------


def forecast_cell(cycle_no, capacity_mah):
    """The fade forecast for one cell.

    Picks the form with the lowest held-out error, refits it on every cycle and
    finds where it falls to 80% of early capacity. The 80% band comes from the
    10th and 90th percentiles of that form's held-out % errors, so it shows how
    well the curve has followed this cell, not how other cells behave.
    """
    n, q = clean_cycles(cycle_no, capacity_mah)
    rows = score_cell(n, q)
    if not rows:
        needed = MIN_FIT_CYCLES + MIN_TEST_CYCLES
        raise FadeError(
            f"Need at least {needed} usable cycles to forecast; found {len(n)}."
        )

    scores = {}
    for model in (*FORMS, BASELINE):
        scores[model] = mean_mape(rows, model)
    best = min(FORMS, key=lambda form: scores[form])
    params = fit_form(best, n, q)

    reference = float(q[:EARLY_CYCLES].max())
    threshold = END_OF_LIFE * reference
    # A flat cell's slope is rounding noise, which would put end of life at an
    # absurd cycle number, so a fit that barely falls counts as not fading
    fitted_ends = predict_form(best, params, [n[0], n[-1]])
    fading = fitted_ends[0] - fitted_ends[1] >= MIN_FADE * reference
    eol = end_of_life_cycle(best, params, threshold) if fading else None

    errors = []
    for row in rows:
        if row["model"] == best:
            errors.extend(row["errors_pct"])
    low_error, high_error = np.percentile(errors, BAND)
    eol_early = None
    eol_late = None
    if eol is not None:
        # A curve that reads low reaches the threshold sooner
        eol_early = end_of_life_cycle(best, params, threshold / (1 + low_error / 100))
        eol_late = end_of_life_cycle(best, params, threshold / (1 + high_error / 100))

    below = n[q < threshold]
    measured = float(below[0]) if len(below) else None
    last_cycle = float(n[-1])
    extrapolated = eol is None or eol > MAX_EXTRAPOLATION * last_cycle

    end = max(last_cycle, eol or 0) if not extrapolated else last_cycle
    grid = np.linspace(n[0], end, CURVE_POINTS)
    fit = predict_form(best, params, grid)
    return {
        "model": best,
        "params": {"slope": params[0], "intercept": params[1]},
        "scores_mape": scores,
        "reference_mah": reference,
        "threshold_mah": threshold,
        "first_cycle": float(n[0]),
        "last_cycle": last_cycle,
        "n_cycles": len(n),
        "eol_cycle": eol,
        "eol_low": eol_early,
        "eol_high": eol_late,
        "measured_eol_cycle": measured,
        "extrapolated": bool(extrapolated),
        "curve": {
            "cycle": grid.round(1).tolist(),
            "fit": fit.round(5).tolist(),
            "low": (fit * (1 + low_error / 100)).round(5).tolist(),
            "high": (fit * (1 + high_error / 100)).round(5).tolist(),
        },
    }


# --- Many cells, for the model registry -----------------------------------------


def r_squared(actual, predicted):
    total = np.sum((actual - np.mean(actual)) ** 2)
    if total == 0:
        return None
    return float(1 - np.sum((actual - predicted) ** 2) / total)


def score_cells(cells):
    """Score every model across cells. cells: {record_id: (cycle_no, capacity_mah)}.

    Returns one result per model (the three forms and the baseline) in the shared
    model-result shape used for the registry, the pages and the figures. The
    headline metric is the mean MAPE across cells at the 50% cut.
    """
    rows_by_cell = {}
    for record_id, (cycle_no, capacity) in cells.items():
        n, q = clean_cycles(cycle_no, capacity)
        rows = score_cell(n, q)
        if rows:
            rows_by_cell[record_id] = rows
    if not rows_by_cell:
        raise FadeError("No cell has enough usable cycles to score.")

    results = []
    for model in (*FORMS, BASELINE):
        by_cut = {}
        for cut in CUTS:
            by_cut[cut] = {"mape": [], "n_train": 0, "n_test": 0}
        actual, predicted, per_cell = [], [], {}
        for record_id, rows in rows_by_cell.items():
            for row in rows:
                if row["model"] != model:
                    continue
                part = by_cut[row["cut"]]
                part["mape"].append(row["mape"])
                part["n_train"] += row["n_train"]
                part["n_test"] += row["n_test"]
                if row["cut"] == MAIN_CUT:
                    actual.extend(row["actual"])
                    predicted.extend(row["predicted"])
                    per_cell[record_id] = round(row["mape"], 3)
        results.append(
            standard_result(
                model, by_cut, np.array(actual), np.array(predicted), per_cell
            )
        )

    baseline = results[-1]
    for result in results:
        result["baseline_metric"] = baseline["metric"]
        result["baseline_r2"] = baseline["r2"]
    return results


def standard_result(model, by_cut, actual, predicted, per_cell):
    """One model's scores in the shared model-result shape."""
    mape_by_cut = {}
    for cut, part in by_cut.items():
        if part["mape"]:
            mape_by_cut[str(cut)] = round(float(np.mean(part["mape"])), 4)
    main = by_cut[MAIN_CUT]
    metric = float(np.mean(main["mape"])) if main["mape"] else None
    pick = np.linspace(0, len(actual) - 1, min(len(actual), MAX_SCATTER_POINTS))
    pick = pick.astype(int)
    return {
        "data_type": "cycling",
        "model": model,
        "target": TARGET,
        "unit": "mAh",
        "split_method": SPLIT_METHOD,
        "features": ["cycle_no"],
        "n_groups": len(per_cell),
        "n_train": main["n_train"],
        "n_test": main["n_test"],
        "metric_name": METRIC_NAME,
        "metric": metric,
        "baseline_metric": None,
        "r2": r_squared(actual, predicted) if len(actual) else None,
        "baseline_r2": None,
        "importance": [],
        "scatter": {
            "actual": actual[pick].round(4).tolist() if len(actual) else [],
            "predicted": predicted[pick].round(4).tolist() if len(actual) else [],
        },
        "metrics": {"mape_by_cut": mape_by_cut, "mape_by_cell": per_cell},
        "note": "Each cell is fitted on its own early cycles; "
        "no cycle after the cut is seen by the fit.",
    }
