"""The model registry: every training run, for any data type, in tbl_model_runs.

Each model's own output is put into one shared result shape first (the keys the
mixing Model tab shows), so pages, endpoints and figures read every data type the
same way:

    {data_type, model, target, unit, split_method, features, n_groups, n_train,
     n_test, metric_name, metric, baseline_metric, r2, baseline_r2,
     importance: [{feature, value}], scatter: {actual, predicted}, metrics, note}

services/fade_model.py returns this shape already; mixing_to_standard() converts
the mixing model's result. The mixing page itself is unchanged.
"""

from psycopg2.extras import Json, RealDictCursor

from core.db import get_connection
from services.mixing_model import FEATURES as MIXING_FEATURES

# Shared-shape keys kept in the metrics column alongside the model's own scores
EXTRA_KEYS = ("unit", "n_groups", "r2", "baseline_r2", "importance", "scatter", "note")
RUN_COLUMNS = (
    "run_id",
    "data_type",
    "model",
    "target",
    "features",
    "split_method",
    "n_train",
    "n_test",
    "metric_name",
    "metric",
    "baseline_metric",
    "metrics",
    "params",
    "data_source",
    "code_version",
    "trained_at",
)


def mixing_to_standard(result):
    """The shared shape for a train_and_score() result (services/mixing_model.py).

    Every second is predicted once, by a model that never saw its stage, so
    n_test is every second and there is no fixed training set (n_train is None).
    """
    return {
        "data_type": "mixing",
        "model": "Gradient boosting",
        "target": "Total mixer power",
        "unit": "W",
        "split_method": "Leave one stage out",
        "features": list(MIXING_FEATURES),
        "n_groups": result["stages"],
        "n_train": None,
        "n_test": result["seconds"],
        "metric_name": "MAE (W)",
        "metric": result["mae_w"],
        "baseline_metric": result["baseline_mae_w"],
        "r2": result["r2"],
        "baseline_r2": result["baseline_r2"],
        "importance": result["importance"],
        "scatter": result["scatter"],
        "metrics": {"runs": result["runs"], "stages": result["stages"]},
        "note": "Each stage is predicted by a model that never saw it; "
        "the baseline guesses the average power.",
    }


def record_run(result, params=None, data_source=None, code_version=None, conn=None):
    """Save one shared-shape result and return its run_id. Commits, unless conn is
    given (then the caller commits or rolls back)."""
    metrics = dict(result.get("metrics") or {})
    for key in EXTRA_KEYS:
        metrics[key] = result.get(key)
    row = {
        "data_type": result["data_type"],
        "model": result["model"],
        "target": result["target"],
        "features": list(result.get("features") or []),
        "split_method": result["split_method"],
        "n_train": result.get("n_train"),
        "n_test": result.get("n_test"),
        "metric_name": result["metric_name"],
        "metric": result.get("metric"),
        "baseline_metric": result.get("baseline_metric"),
        "metrics": Json(metrics),
        "params": Json(params or {}),
        "data_source": data_source,
        "code_version": code_version,
    }
    own = conn is None
    if own:
        conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO tbl_model_runs (data_type, model, target, features, "
            "split_method, n_train, n_test, metric_name, metric, baseline_metric, "
            "metrics, params, data_source, code_version) VALUES (%(data_type)s, "
            "%(model)s, %(target)s, %(features)s, %(split_method)s, %(n_train)s, "
            "%(n_test)s, %(metric_name)s, %(metric)s, %(baseline_metric)s, "
            "%(metrics)s, %(params)s, %(data_source)s, %(code_version)s) "
            "RETURNING run_id",
            row,
        )
        run_id = cur.fetchone()[0]
        if own:
            conn.commit()
        return run_id
    finally:
        cur.close()
        if own:
            conn.close()


def list_runs(data_type=None, latest_only=False, conn=None):
    """Registry rows, newest first, as dicts with the shared-shape keys lifted out
    of metrics. latest_only keeps the newest run of each (data_type, model)."""
    query = f"SELECT {', '.join(RUN_COLUMNS)} FROM tbl_model_runs"
    values = []
    if data_type:
        query += " WHERE data_type = %s"
        values.append(data_type)
    query += " ORDER BY trained_at DESC, run_id DESC"

    own = conn is None
    if own:
        conn = get_connection(read_only=True)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(query, values)
        rows = cur.fetchall()
    finally:
        cur.close()
        if own:
            conn.close()

    runs, seen = [], set()
    for row in rows:
        key = (row["data_type"], row["model"])
        if latest_only and key in seen:
            continue
        seen.add(key)
        run = dict(row)
        run["trained_at"] = row["trained_at"].isoformat()
        for name in EXTRA_KEYS:
            run[name] = run["metrics"].get(name)
        runs.append(run)
    return runs
