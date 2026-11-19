#!/usr/bin/env python3
# Raw figures for talks and posters. One figure per file in figures/, as a 300 dpi PNG
# and an SVG, with no title (the file name says what it shows), plus figures/numbers.md:
# every number a figure shows, with the input and model run it came from.
#   Mixing: the mixer logs in data/approved/mixing/*.xlsx and the latest mixing run.
#   Cycling: tbl_cycle_summary, the stored .mpr files and the latest cycling runs.
# Run scripts.extract_cycle_features and scripts.train_cycle_life first.
#   python3 -m scripts.make_figures

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scripts.train_cycle_life import (  # noqa: E402
    MIXING_DIR,
    approved_files,
    code_version,
    sha256_of,
)
from services import fade_model  # noqa: E402
from services.cycle_features import (  # noqa: E402
    FeatureError,
    check_cycles,
    cycling_files,
    delta_q_curve,
    load_fade_cells,
)
from services.cycling import (  # noqa: E402
    group_full_cycles,
    read_half_cycles,
    stored_file,
)
from services.mixing import check_mixer_log  # noqa: E402
from services.model_registry import list_runs  # noqa: E402

OUT = Path("figures")
SCRIPT = "scripts/make_figures.py"
DPI = 300
SIZE = (4.5, 3.4)  # inches; resized freely later, the text stays readable

# One colour per thing, the same in every figure
COLOURS = {
    fade_model.LINEAR: "#1f77b4",
    fade_model.SQUARE_ROOT: "#ff7f0e",
    fade_model.EXPONENTIAL: "#2ca02c",
    fade_model.BASELINE: "#7f7f7f",
    "Gradient boosting": "#9467bd",
    "Mixing baseline": "#7f7f7f",
    "measured": "#000000",
    "rows_in": "#c7c7c7",
    "rows_out": "#4d4d4d",
    "failed": "#d62728",
}
MODEL_ORDER = (*fade_model.FORMS, fade_model.BASELINE)


# --- Saving --------------------------------------------------------------------


def new_figure():
    fig, ax = plt.subplots(figsize=SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def save(fig, name, out):
    """name.png (300 dpi) and name.svg in out/."""
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=DPI)
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)


def note(numbers, figure, quantity, value, source, run=""):
    """One line of numbers.md."""
    if isinstance(value, float):
        value = f"{value:.4g}"
    numbers.append(
        {
            "figure": figure,
            "quantity": quantity,
            "value": "-" if value is None else str(value),
            "source": source,
            "run": str(run),
        }
    )


def write_numbers(numbers, skipped, out):
    made = f"Made by `{SCRIPT}` on {datetime.now():%Y-%m-%d %H:%M}"
    lines = [
        "# Figure numbers",
        "",
        f"{made} from code version `{code_version()}`.",
        "",
        "| Figure | Quantity | Value | Input | Model run |",
        "|---|---|---|---|---|",
    ]
    for row in numbers:
        cells = []
        for key in ("figure", "quantity", "value", "source", "run"):
            cells.append(row[key].replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    if skipped:
        lines += ["", "## Skipped", ""]
        for name, reason in skipped:
            lines.append(f"- `{name}`: {reason}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "numbers.md").write_text("\n".join(lines) + "\n")


# --- Mixing --------------------------------------------------------------------


def mixing_gate(reports, source, numbers, out):
    """(a) Each gate check: rows in and out, summed over the logs. A check that
    rejected any log is drawn in red. reports: {file name: check_mixer_log()}."""
    name = "mixing_validation_gate_rows_in_out"
    checks, rows_in, rows_out, failures = [], {}, {}, {}
    for report in reports.values():
        for step in report:
            check = step["check"]
            if check not in rows_in:
                checks.append(check)
                rows_in[check], rows_out[check], failures[check] = 0, 0, 0
            rows_in[check] += step["rows_in"] or 0
            rows_out[check] += step["rows_out"] or 0
            failures[check] += not step["passed"]

    fig, ax = new_figure()
    x = np.arange(len(checks))
    ins, outs, colours = [], [], []
    for check in checks:
        ins.append(rows_in[check])
        outs.append(rows_out[check])
        colours.append(COLOURS["failed"] if failures[check] else COLOURS["rows_out"])
    ax.bar(x - 0.2, ins, 0.4, color=COLOURS["rows_in"])
    ax.bar(x + 0.2, outs, 0.4, color=colours)
    ax.set_xticks(x, checks, rotation=30, ha="right")
    ax.set_ylabel("Log rows (1 per second)")
    legend = [
        Patch(color=COLOURS["rows_in"], label="Rows in"),
        Patch(color=COLOURS["rows_out"], label="Rows out"),
        Patch(color=COLOURS["failed"], label="A log rejected"),
    ]
    ax.legend(
        handles=legend, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1)
    )
    save(fig, name, out)

    note(numbers, name, "Mixer logs checked", len(reports), source)
    for check in checks:
        note(numbers, name, f"{check}: rows in", rows_in[check], source)
        note(numbers, name, f"{check}: rows out", rows_out[check], source)
        note(numbers, name, f"{check}: logs failed", failures[check], source)


def mixing_scatter(run, numbers, out):
    """(b) Held-out predicted against measured power."""
    name = "mixing_model_vs_baseline_predicted_vs_measured"
    actual = np.array(run["scatter"]["actual"])
    predicted = np.array(run["scatter"]["predicted"])
    fig, ax = new_figure()
    top = max(actual.max(), predicted.max()) * 1.05
    ax.plot([0, top], [0, top], color=COLOURS["measured"], lw=0.8, ls="--")
    ax.scatter(actual, predicted, s=4, alpha=0.5, color=COLOURS["Gradient boosting"])
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel("Measured total power (W)")
    ax.set_ylabel("Predicted total power (W)")
    save(fig, name, out)
    source, run_id = run["data_source"], run["run_id"]
    note(numbers, name, "Seconds plotted (sample)", len(actual), source, run_id)
    note(numbers, name, "Seconds predicted", run["n_test"], source, run_id)
    note(numbers, name, "Stages (groups)", run["n_groups"], source, run_id)


def mixing_error(run, numbers, out):
    """(b) Model against baseline: mean absolute error."""
    name = "mixing_model_vs_baseline_error"
    fig, ax = new_figure()
    colours = [COLOURS["Gradient boosting"], COLOURS["Mixing baseline"]]
    values = [run["metric"], run["baseline_metric"]]
    ax.bar(["Model", "Average power"], values, color=colours, width=0.6)
    ax.set_ylabel("Mean absolute error (W)")
    save(fig, name, out)
    source, run_id = run["data_source"], run["run_id"]
    note(numbers, name, "Model MAE (W)", run["metric"], source, run_id)
    note(numbers, name, "Baseline MAE (W)", run["baseline_metric"], source, run_id)
    note(numbers, name, "Model R²", run["r2"], source, run_id)
    note(numbers, name, "Baseline R²", run["baseline_r2"], source, run_id)


# --- Cycling -------------------------------------------------------------------


def runs_by_model(runs):
    """The registry runs in MODEL_ORDER, for those that exist."""
    found = {}
    for run in runs:
        found[run["model"]] = run
    ordered = []
    for model in MODEL_ORDER:
        if model in found:
            ordered.append(found[model])
    return ordered


def short_label(model):
    """'Linear fade' -> 'Linear'; the baseline becomes 'No fade'."""
    if model == fade_model.BASELINE:
        return "No fade"
    return model.replace(" fade", "")


def cycling_comparison(runs, numbers, out):
    """(c) Forecast error on later cycles per model, at the 50% cut."""
    name = "cycling_model_comparison_forecast_error"
    runs = runs_by_model(runs)
    if not runs:
        raise ValueError("no fade model runs in the registry")
    fig, ax = new_figure()
    labels, values, colours = [], [], []
    for run in runs:
        labels.append(short_label(run["model"]))
        values.append(run["metric"])
        colours.append(COLOURS[run["model"]])
    ax.bar(labels, values, color=colours, width=0.6)
    ax.set_xlabel("Model ('No fade' is the baseline)")
    ax.set_ylabel("Error on later cycles, MAPE (%)")
    save(fig, name, out)
    for run in runs:
        source, run_id = run["data_source"], run["run_id"]
        quantity = f"{run['model']}: {run['metric_name']}"
        note(numbers, name, quantity, run["metric"], source, run_id)
        note(numbers, name, f"{run['model']}: cells", run["n_groups"], source, run_id)


def cycling_error_vs_fraction(runs, numbers, out):
    """(e) Forecast error against how much of each test was used for fitting."""
    name = "cycling_forecast_error_vs_share_of_test_used"
    runs = runs_by_model(runs)
    if not runs:
        raise ValueError("no fade model runs in the registry")
    fig, ax = new_figure()
    for run in runs:
        by_cut = run["metrics"].get("mape_by_cut", {})
        shares, values = [], []
        for cut in fade_model.CUTS:
            if str(cut) not in by_cut:
                continue
            shares.append(100 * cut)
            values.append(by_cut[str(cut)])
            quantity = f"{run['model']}: MAPE (%) fitting on {100 * cut:.0f}%"
            note(
                numbers,
                name,
                quantity,
                by_cut[str(cut)],
                run["data_source"],
                run["run_id"],
            )
        colour = COLOURS[run["model"]]
        ax.plot(
            shares, values, marker="o", color=colour, label=short_label(run["model"])
        )
    ax.set_xlabel("Share of each test used for fitting (%)")
    ax.set_ylabel("Error on later cycles, MAPE (%)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, name, out)


def cycling_example(record_id, cycle_no, capacity, source, numbers, out):
    """(d) One cell: measured capacity, the chosen fit, its 80% band and the
    end-of-life line."""
    name = "cycling_example_fade_forecast"
    result = fade_model.forecast_cell(cycle_no, capacity)
    n, q = fade_model.clean_cycles(cycle_no, capacity)
    curve = result["curve"]
    colour = COLOURS[result["model"]]
    fig, ax = new_figure()
    ax.scatter(n, q, s=4, color=COLOURS["measured"], label="Measured")
    ax.plot(curve["cycle"], curve["fit"], color=colour, label=result["model"])
    ax.fill_between(
        curve["cycle"], curve["low"], curve["high"], color=colour, alpha=0.2, lw=0
    )
    ax.axhline(result["threshold_mah"], color=COLOURS["failed"], lw=0.8, ls="--")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Discharge capacity (mAh)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, name, out)
    values = (
        ("Cell", record_id),
        ("Cycles used", result["n_cycles"]),
        ("Chosen curve", result["model"]),
        ("80% of early capacity (mAh)", result["threshold_mah"]),
        ("Forecast end of life (cycle)", result["eol_cycle"]),
        ("80% band, early end (cycle)", result["eol_low"]),
        ("80% band, late end (cycle)", result["eol_high"]),
        ("Measured end of life (cycle)", result["measured_eol_cycle"]),
        ("Flagged as extrapolation", result["extrapolated"]),
    )
    for quantity, value in values:
        note(numbers, name, quantity, value, source)


def band_ends(row):
    """(low, high) of a backtest row, falling back to the forecast itself."""
    low = row["low"] if row["low"] is not None else row["forecast"]
    high = row["high"] if row["high"] is not None else row["forecast"]
    return low, high


def cycling_backtest(cells, source, numbers, out):
    """(d) Cells that reached end of life: forecast from the first half of each test
    against when they really did, with 80% bands."""
    name = "cycling_predicted_vs_measured_end_of_life"
    rows = []
    for row in fade_model.backtest_end_of_life(cells):
        if row["forecast"] is not None:
            rows.append(row)
    if not rows:
        raise ValueError("no cell has reached 80% of its early capacity yet")
    fig, ax = new_figure()
    top = 0
    for row in rows:
        low, high = band_ends(row)
        ax.errorbar(
            row["measured"],
            row["forecast"],
            yerr=[[row["forecast"] - low], [high - row["forecast"]]],
            fmt="o",
            ms=4,
            capsize=2,
            color=COLOURS[row["model"]],
        )
        top = max(top, row["measured"], high)
        value = f"{row['measured']:.0f} / {row['forecast']:.0f} / {low:.0f}-{high:.0f}"
        quantity = f"{row['record_id']}: measured / forecast / band (cycle)"
        note(numbers, name, quantity, value, source)
    top *= 1.05
    ax.plot([0, top], [0, top], color=COLOURS["measured"], lw=0.8, ls="--")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Measured end of life (cycle)")
    ax.set_ylabel("Forecast end of life (cycle)")
    save(fig, name, out)
    note(numbers, name, "Cells", len(rows), source)
    note(
        numbers,
        name,
        "Share of each test fitted (%)",
        100 * fade_model.MAIN_CUT,
        source,
    )


def stopping_points(cells):
    """For each cut: (share fitted, cycler time saved %, dangerous stops %, cells).
    A stop is dangerous when the cell reached end of life before the band's early
    end, so stopping its test would have overstated its life."""
    points = []
    for share in fade_model.CUTS:
        saved, total, dangerous, counted = 0.0, 0.0, 0, 0
        for row in fade_model.backtest_end_of_life(cells, share):
            if row["forecast"] is None:
                continue
            low, _ = band_ends(row)
            counted += 1
            saved += row["measured"] - row["fitted_cycles"]
            total += row["measured"]
            dangerous += row["measured"] < low
        if counted:
            points.append(
                (share, 100 * saved / total, 100 * dangerous / counted, counted)
            )
    return points


def cycling_time_saved(cells, source, numbers, out):
    """(f) Stopping each test early: cycler time saved against dangerous stops."""
    name = "cycling_cycler_time_saved_vs_dangerous_stops"
    points = stopping_points(cells)
    if not points:
        raise ValueError("no cell has reached 80% of its early capacity yet")
    fig, ax = new_figure()
    for share, saved_pct, dangerous_pct, counted in points:
        ax.scatter(saved_pct, dangerous_pct, color=COLOURS[fade_model.LINEAR])
        ax.annotate(
            f"fit on {100 * share:.0f}%",
            (saved_pct, dangerous_pct),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
        label = f"Stop after {100 * share:.0f}% of cycles"
        note(numbers, name, f"{label}: cycler time saved (%)", saved_pct, source)
        note(numbers, name, f"{label}: dangerous stops (%)", dangerous_pct, source)
        note(numbers, name, f"{label}: cells", counted, source)
    ax.set_xlabel("Cycler time saved (% of cycles to end of life)")
    ax.set_ylabel("Dangerous stops (% of cells)")
    ax.set_ylim(bottom=0)
    save(fig, name, out)


def cycling_fade_curves(cells, source, numbers, out):
    """(g) Every cell's discharge capacity as a share of its early capacity."""
    name = "cycling_capacity_fade_curves"
    fig, ax = new_figure()
    plotted = 0
    for cycle_no, capacity in cells.values():
        n, q = fade_model.clean_cycles(cycle_no, capacity)
        if not len(n):
            continue
        early = q[: fade_model.EARLY_CYCLES].max()
        ax.plot(n, 100 * q / early, color=COLOURS["measured"], lw=0.8, alpha=0.5)
        plotted += 1
    ax.axhline(100 * fade_model.END_OF_LIFE, color=COLOURS["failed"], lw=0.8, ls="--")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Discharge capacity (% of early capacity)")
    save(fig, name, out)
    note(numbers, name, "Cells", plotted, source)
    note(numbers, name, "End-of-life line (%)", 100 * fade_model.END_OF_LIFE, source)


def cycling_delta_q(curves, numbers, out):
    """(g) ΔQ(V), capacity at cycle 100 minus cycle 10, for cells with 100 cycles.
    curves: {record_id: (voltage grid, ΔQ in Ah, source)}."""
    name = "cycling_delta_q_curves_cycle100_minus_cycle10"
    if not curves:
        raise ValueError("no cell has 100 cycles that passed the quality gate")
    fig, ax = new_figure()
    for record_id, (grid, dq, source) in curves.items():
        ax.plot(grid, 1000 * dq, color=COLOURS["measured"], lw=0.8, alpha=0.6)
        variance = float(np.var(dq))
        if variance > 0:
            quantity = f"{record_id}: log10 variance of ΔQ (Ah²)"
            note(numbers, name, quantity, float(np.log10(variance)), source)
    ax.axhline(0, color=COLOURS["rows_in"], lw=0.8)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("ΔQ, cycle 100 minus cycle 10 (mAh)")
    save(fig, name, out)
    note(numbers, name, "Cells", len(curves), "stored .mpr files")


def delta_q_curves():
    """ΔQ(V) of each record's newest stored file that has 100 usable cycles."""
    newest = {}
    for cycling_id, record_id, file_path in cycling_files():
        newest[record_id] = (cycling_id, file_path)
    curves = {}
    for record_id, (cycling_id, file_path) in newest.items():
        path = stored_file(file_path)
        if path is None:
            continue
        try:
            halves = read_half_cycles(path, max_points=None)
            kept, _ = check_cycles(group_full_cycles(halves))
            grid, dq = delta_q_curve(kept)
        except (FeatureError, ValueError, OSError):
            continue
        curves[record_id] = (grid, dq, f"tbl_cycling_data {cycling_id}")
    return curves


# --- Running -------------------------------------------------------------------


def make(name, job, skipped):
    """Make one figure. One whose input is missing is skipped, not fatal."""
    try:
        job()
        print(f"  made {name}")
    except Exception as e:  # missing input, or too little data
        skipped.append((name, str(e)))
        print(f"  skipped {name}: {e}")


def mixing_figures(numbers, skipped, out):
    logs = approved_files(MIXING_DIR, "*.xlsx") if MIXING_DIR.exists() else []
    reports, sources = {}, []
    for path in logs:
        reports[path.name] = check_mixer_log(path.read_bytes())
        sources.append(f"{path} sha256={sha256_of(path)[:12]}")
    if reports:
        source = "; ".join(sources)
        make("a", lambda: mixing_gate(reports, source, numbers, out), skipped)
    else:
        skipped.append(("a", f"no mixer logs in {MIXING_DIR}/"))

    runs = list_runs("mixing", latest_only=True)
    if not runs:
        skipped.append(
            ("b", "no mixing run in the registry (scripts.train_cycle_life)")
        )
        return
    make("b scatter", lambda: mixing_scatter(runs[0], numbers, out), skipped)
    make("b error", lambda: mixing_error(runs[0], numbers, out), skipped)


def cycling_figures(numbers, skipped, out):
    runs = list_runs("cycling", latest_only=True)
    if runs:
        make("c", lambda: cycling_comparison(runs, numbers, out), skipped)
        make("e", lambda: cycling_error_vs_fraction(runs, numbers, out), skipped)
    else:
        skipped.append(("c, e", "no cycling runs in the registry"))

    cells = load_fade_cells()
    if cells:
        source = f"tbl_cycle_summary, newest file of {len(cells)} records"
        longest = max(cells, key=lambda record_id: len(cells[record_id][0]))
        cycle_no, capacity = cells[longest]

        def example():
            cycling_example(longest, cycle_no, capacity, source, numbers, out)

        make("d example", example, skipped)
        make(
            "d backtest", lambda: cycling_backtest(cells, source, numbers, out), skipped
        )
        make("f", lambda: cycling_time_saved(cells, source, numbers, out), skipped)
        make(
            "g fade", lambda: cycling_fade_curves(cells, source, numbers, out), skipped
        )
    else:
        skipped.append(("d, f, g fade", "tbl_cycle_summary is empty"))
    make("g ΔQ(V)", lambda: cycling_delta_q(delta_q_curves(), numbers, out), skipped)


def main(out=OUT):
    numbers, skipped = [], []
    print("Mixing:")
    mixing_figures(numbers, skipped, out)
    print("Cycling:")
    cycling_figures(numbers, skipped, out)
    write_numbers(numbers, skipped, out)
    print(f"Wrote {out}/numbers.md ({len(numbers)} numbers, {len(skipped)} skipped)")


if __name__ == "__main__":
    main()
