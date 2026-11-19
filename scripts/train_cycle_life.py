#!/usr/bin/env python3
# Score the models and write every run to the model registry (tbl_model_runs).
#   Cycling: the capacity fade forecast (services/fade_model.py) on every record's
#   newest file in tbl_cycle_summary (fill it first with
#   scripts.extract_cycle_features).
#   Mixing: the power model (services/mixing_model.py) on the mixer logs in
#   data/approved/mixing/*.xlsx. Only files under data/approved/ are read.
#   python3 -m scripts.train_cycle_life [--only cycling|mixing] [--dry-run]

import argparse
import hashlib
import subprocess
from pathlib import Path

from services import fade_model
from services.cycle_features import load_fade_cells
from services.mixing import find_stages, parse_mixer_xlsx
from services.mixing_model import ModelError, make_model, train_and_score
from services.model_registry import mixing_to_standard, record_run

APPROVED = Path("data/approved")
MIXING_DIR = APPROVED / "mixing"


def code_version():
    """The short git commit, with "-dirty" if there are uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if changes else commit


def approved_files(folder, pattern):
    """Files matching pattern in folder, which must be inside data/approved/."""
    folder = Path(folder).resolve()
    if not folder.is_relative_to(APPROVED.resolve()):
        raise ValueError(f"{folder} is not inside {APPROVED}/")
    return sorted(folder.glob(pattern))


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def show(result):
    metric = result["metric"]
    baseline = result["baseline_metric"]
    print(
        f"  {result['model']:<20} {result['metric_name']}: "
        f"{'-' if metric is None else f'{metric:.3f}'} "
        f"(baseline {'-' if baseline is None else f'{baseline:.3f}'}), "
        f"{result['n_groups']} groups, {result['n_test']} test rows"
    )


def cycling_runs():
    """The fade model's results and the registry fields that go with them."""
    cells = load_fade_cells()
    results = fade_model.score_cells(cells)
    params = {
        "cuts": list(fade_model.CUTS),
        "main_cut": fade_model.MAIN_CUT,
        "end_of_life": fade_model.END_OF_LIFE,
        "early_cycles": fade_model.EARLY_CYCLES,
        "baseline_window": fade_model.BASELINE_WINDOW,
        "min_fit_cycles": fade_model.MIN_FIT_CYCLES,
        "min_test_cycles": fade_model.MIN_TEST_CYCLES,
    }
    source = (
        f"tbl_cycle_summary: gate-passed cycles of the newest file of "
        f"{len(cells)} records"
    )
    return results, params, source


def mixing_runs(folder=MIXING_DIR):
    """The mixing model's result from the approved mixer logs."""
    files = approved_files(folder, "*.xlsx")
    if not files:
        raise ModelError(f"No mixer logs (*.xlsx) in {folder}/")
    runs, sources = [], []
    for path in files:
        data = path.read_bytes()
        _, log = parse_mixer_xlsx(data)
        runs.append((path.stem, log, find_stages(log)))
        sources.append(f"{path} sha256={sha256_of(path)}")
    result = mixing_to_standard(train_and_score(runs))
    return [result], make_model().get_params(), "; ".join(sources)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("cycling", "mixing"))
    parser.add_argument(
        "--dry-run", action="store_true", help="print the scores, save nothing"
    )
    args = parser.parse_args(argv)

    version = code_version()
    jobs = {"cycling": cycling_runs, "mixing": mixing_runs}
    for data_type, job in jobs.items():
        if args.only and args.only != data_type:
            continue
        print(f"{data_type}:")
        try:
            results, params, source = job()
        except (fade_model.FadeError, ModelError, ValueError) as e:
            print(f"  skipped: {e}")
            continue
        for result in results:
            show(result)
            if not args.dry_run:
                run_id = record_run(result, params, source, version)
                print(f"    saved as run {run_id}")


if __name__ == "__main__":
    main()
