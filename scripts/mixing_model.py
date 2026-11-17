"""Train a basic model on one or more mixer logs and draw the results.

    python -m scripts.mixing_model Mix_1_Data.xlsx [more.xlsx ...] --out mixing_model.png

Question the model answers: given how the mixer was set and how far through the
run it is, how much power does it draw? Power at a fixed speed rises with slurry
viscosity, so a model that follows the power through the run is following the
slurry as solvent is added.

Validation is leave-one-stage-out: every stage is predicted by a model that never saw
that stage, so a second of power is never predicted from its own neighbours.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.mixing import find_stages, parse_mixer_xlsx  # noqa: E402

FEATURES = ["rotor_rpm", "t_in_stage_s", "pause_before_s", "energy_before_kj", "stage_no"]
LABELS = {
    "rotor_rpm": "Rotor speed",
    "t_in_stage_s": "Time in stage",
    "pause_before_s": "Pause before stage",
    "energy_before_kj": "Energy used so far",
    "stage_no": "Stage number",
}
DARK, GREY, YELLOW, RED, BLUE = "#2B2C2D", "#6B6C6D", "#FBB034", "#E0201B", "#0090D0"


def build_table(paths):
    """One row per second of rotor-on time, with run and stage identifiers."""
    parts, stage_tables = [], []
    for run, path in enumerate(paths, 1):
        summary, log = parse_mixer_xlsx(Path(path).read_bytes())
        stages = find_stages(log)
        stages["run"] = run
        stage_tables.append(stages)
        energy_so_far = 0.0
        for s in stages.itertuples():
            part = log[(log["t_s"] >= s.start_s) & (log["t_s"] < s.start_s + s.duration_s)].copy()
            part["run"], part["stage_no"] = run, s.stage_no
            part["t_in_stage_s"] = part["t_s"] - s.start_s
            part["pause_before_s"] = s.pause_before_s
            part["energy_before_kj"] = energy_so_far
            part["group"] = f"{run}-{s.stage_no}"
            energy_so_far += s.energy_kj
            parts.append(part)
    return pd.concat(parts, ignore_index=True), pd.concat(stage_tables, ignore_index=True)


def make_model():
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="mixing_model.png")
    args = ap.parse_args()

    data, stages = build_table(args.logs)
    X, y, groups = data[FEATURES], data["total_power_w"], data["group"]
    cv = LeaveOneGroupOut()
    pred = cross_val_predict(make_model(), X, y, groups=groups, cv=cv)
    base = cross_val_predict(DummyRegressor(), X, y, groups=groups, cv=cv)
    data["pred"] = pred

    r2, mae = r2_score(y, pred), mean_absolute_error(y, pred)
    r2_b, mae_b = r2_score(y, base), mean_absolute_error(y, base)
    print(f"{len(data)} seconds in {data['group'].nunique()} stages from {len(args.logs)} run(s)")
    print(f"Model     R2 {r2:.2f}   MAE {mae:.1f} W")
    print(f"Mean only R2 {r2_b:.2f}   MAE {mae_b:.1f} W")

    final = make_model().fit(X, y)
    imp = permutation_importance(final, X, y, n_repeats=10, random_state=0)
    order = np.argsort(imp.importances_mean)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    fig.suptitle("Mixer log model: power draw from settings and progress", fontsize=16, fontweight="bold", color=DARK)

    # A. the run
    a = ax[0, 0]
    log = data.sort_values("t_s")
    a.plot(log["t_s"] / 60, log["total_power_w"], ".", ms=2, color=GREY, label="Measured power")
    a.plot(log["t_s"] / 60, log["pred"], ".", ms=2, color=RED, label="Predicted (stage held out)")
    for s in stages[stages["run"] == 1].itertuples():
        a.axvspan(s.start_s / 60, (s.start_s + s.duration_s) / 60, color=YELLOW, alpha=0.18, lw=0)
    a.set_xlabel("Time (min)"); a.set_ylabel("Total power (W)")
    a.set_title("A  The run: shaded bands are rotor-on stages", loc="left", fontsize=12)
    a.legend(loc="upper right", markerscale=5, frameon=False)

    # B. stage summary
    b = ax[0, 1]
    st = stages[stages["run"] == 1]
    colours = [RED if r > 3000 else BLUE for r in st["rotor_rpm"]]
    b.bar(st["stage_no"], st["mean_power_w"], color=colours)
    b.set_xlabel("Stage"); b.set_ylabel("Mean power (W)")
    b.set_title("B  Mean power per stage (red 3600 rpm, blue lower speeds)", loc="left", fontsize=12)
    b.set_xticks(st["stage_no"])

    # C. actual vs predicted
    c = ax[1, 0]
    c.scatter(y, pred, s=5, alpha=0.4, color=BLUE)
    lim = [0, max(y.max(), pred.max()) * 1.05]
    c.plot(lim, lim, "--", color=DARK, lw=1)
    c.set_xlim(lim); c.set_ylim(lim)
    c.set_xlabel("Measured power (W)"); c.set_ylabel("Predicted power (W)")
    c.set_title(f"C  Held-out stages: R² {r2:.2f}, MAE {mae:.0f} W (mean only: R² {r2_b:.2f})", loc="left", fontsize=12)

    # D. importance
    d = ax[1, 1]
    d.barh([LABELS[FEATURES[i]] for i in order], imp.importances_mean[order], color=YELLOW, edgecolor=DARK)
    d.set_xlabel("Drop in R² when the feature is shuffled")
    d.set_title("D  What the model relies on (fitted on all stages)", loc="left", fontsize=12)

    for a in ax.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
