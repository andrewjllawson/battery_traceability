"""Schema checks for tbl_cycle_summary, tbl_cell_features and tbl_model_runs
(migrations/005_cycling_ml.sql). Needs a database; skipped if none can be
reached. Each test rolls back, so nothing is left behind.

pytest -q tests/test_model_registry.py
"""

import os
import sys
from pathlib import Path

import pytest
from psycopg2 import OperationalError, errors
from psycopg2.extras import Json

from core.db import get_admin_connection, get_connection
from services.cycle_features import REDUCED_FEATURES

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

TEST_RECORD = "TEST-CYC-1"
NEW_TABLES = ("tbl_cycle_summary", "tbl_cell_features", "tbl_model_runs")


def database_available() -> bool:
    try:
        get_connection().close()
        return True
    except OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not database_available(), reason="no database reachable (set DATABASE_URL)"
)


@pytest.fixture
def cur():
    """A cursor with a test cycling file inserted. Rolled back afterwards."""
    conn = get_connection()
    cursor = conn.cursor()
    yield cursor
    conn.rollback()
    cursor.close()
    conn.close()


def insert_file(cur):
    cur.execute(
        "INSERT INTO tbl_cycling_data (record_id, file_path) VALUES (%s, %s) "
        "RETURNING cycling_id",
        (TEST_RECORD, "uploads/cycling_data/test.mpr"),
    )
    return cur.fetchone()[0]


def insert_cycle(cur, cycling_id, cycle_no=1, discharge=1.05, efficiency=0.99):
    cur.execute(
        "INSERT INTO tbl_cycle_summary "
        "(cycling_id, record_id, cycle_no, discharge_mah, charge_mah, "
        "coulombic_eff, charge_time_s) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (cycling_id, TEST_RECORD, cycle_no, discharge, 1.06, efficiency, 3600),
    )


def insert_run(cur, **changes):
    row = {
        "data_type": "cycling",
        "model": "Linear fade",
        "target": "discharge capacity (later cycles)",
        "features": ["cycle_no"],
        "split_method": "fit early cycles, score later",
        "n_train": 50,
        "n_test": 50,
        "metric_name": "MAPE (%)",
        "metric": 1.2,
        "baseline_metric": 3.4,
        "metrics": Json({"cut_50": 1.2}),
        "params": Json({"slope": -0.0005}),
    }
    row.update(changes)
    cur.execute(
        "INSERT INTO tbl_model_runs (data_type, model, target, features, "
        "split_method, n_train, n_test, metric_name, metric, baseline_metric, "
        "metrics, params) VALUES (%(data_type)s, %(model)s, %(target)s, "
        "%(features)s, %(split_method)s, %(n_train)s, %(n_test)s, %(metric_name)s, "
        "%(metric)s, %(baseline_metric)s, %(metrics)s, %(params)s) "
        "RETURNING run_id, trained_at",
        row,
    )
    return cur.fetchone()


@pytest.mark.parametrize("table", NEW_TABLES)
def test_tables_exist(cur, table):
    cur.execute("SELECT to_regclass(%s)", (table,))
    assert cur.fetchone()[0] is not None


@pytest.mark.parametrize("table", NEW_TABLES)
def test_tables_are_unprivileged(table):
    """Listed for the other roles. The list itself is admin-only."""
    conn = get_admin_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT 1 FROM tbl_unprivileged_tables WHERE table_name = %s", (table,)
        )
        assert cur.fetchone() is not None
    finally:
        cur.close()
        conn.close()


def test_cycle_rows_go_with_their_file(cur):
    cycling_id = insert_file(cur)
    insert_cycle(cur, cycling_id, 1)
    insert_cycle(cur, cycling_id, 2)
    cur.execute("DELETE FROM tbl_cycling_data WHERE cycling_id = %s", (cycling_id,))
    cur.execute(
        "SELECT count(*) FROM tbl_cycle_summary WHERE record_id = %s", (TEST_RECORD,)
    )
    assert cur.fetchone()[0] == 0


def test_cycle_number_is_unique_per_file(cur):
    cycling_id = insert_file(cur)
    insert_cycle(cur, cycling_id, 1)
    with pytest.raises(errors.UniqueViolation):
        insert_cycle(cur, cycling_id, 1)


@pytest.mark.parametrize(
    "changes",
    [
        {"cycle_no": 0},
        {"discharge": -1},
        {"efficiency": 0},
    ],
)
def test_bad_cycle_values_rejected(cur, changes):
    cycling_id = insert_file(cur)
    with pytest.raises(errors.CheckViolation):
        insert_cycle(cur, cycling_id, **changes)


def test_feature_columns_match_the_code(cur):
    """The table holds exactly the features a .mpr can give."""
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'tbl_cell_features'"
    )
    columns = set()
    for (name,) in cur.fetchall():
        columns.add(name)
    for name in REDUCED_FEATURES:
        assert name in columns, name
    for name in ("temp_max", "ir_min", "ir_change_2to100"):
        assert name not in columns, name


def test_short_test_stores_reason_not_features(cur):
    cycling_id = insert_file(cur)
    cur.execute(
        "INSERT INTO tbl_cell_features (cycling_id, record_id, n_cycles, gate, "
        "feature_error) VALUES (%s, %s, %s, %s, %s)",
        (cycling_id, TEST_RECORD, 40, Json([]), "Cycle 41 is missing"),
    )
    cur.execute(
        "SELECT log_var_dq, feature_error FROM tbl_cell_features WHERE cycling_id = %s",
        (cycling_id,),
    )
    assert cur.fetchone() == (None, "Cycle 41 is missing")


def test_model_run_inserts_with_defaults(cur):
    run_id, trained_at = insert_run(cur)
    assert run_id and trained_at
    cur.execute(
        "SELECT features, metrics->>'cut_50', metric FROM tbl_model_runs "
        "WHERE run_id = %s",
        (run_id,),
    )
    features, cut_50, metric = cur.fetchone()
    assert features == ["cycle_no"]
    assert cut_50 == "1.2"
    assert isinstance(metric, float)


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"n_test": -1}, errors.CheckViolation),
        ({"model": None}, errors.NotNullViolation),
        ({"metric_name": None}, errors.NotNullViolation),
    ],
)
def test_bad_model_runs_rejected(cur, changes, error):
    with pytest.raises(error):
        insert_run(cur, **changes)


def test_read_only_role_can_read_but_not_write():
    conn = get_connection(read_only=True)
    cur = conn.cursor()
    try:
        cur.execute("SELECT count(*) FROM tbl_model_runs")
        denied = (errors.InsufficientPrivilege, errors.ReadOnlySqlTransaction)
        with pytest.raises(denied):
            cur.execute(
                "INSERT INTO tbl_model_runs (data_type, model, target, "
                "split_method, metric_name) VALUES ('x', 'x', 'x', 'x', 'x')"
            )
    finally:
        conn.rollback()
        cur.close()
        conn.close()
