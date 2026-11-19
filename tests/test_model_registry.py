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
from scripts.train_cycle_life import approved_files
from services.cycle_features import REDUCED_FEATURES, load_fade_cells, save_analysis
from services.fade_model import score_cells
from services.model_registry import list_runs, record_run

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


# --- Saving and loading (services/cycle_features.py, services/model_registry.py)


@pytest.fixture
def conn():
    """A read/write connection rolled back afterwards."""
    connection = get_connection()
    yield connection
    connection.rollback()
    connection.close()


def fake_analysis(n=30, failed=()):
    summary = []
    for k in range(1, n + 1):
        summary.append(
            {
                "cycle_no": k,
                "discharge_mah": 2.0 - 0.002 * k,
                "charge_mah": 2.02 - 0.002 * k,
                "coulombic_eff": 0.99,
                "charge_time_s": 3600.0,
                "passed_gate": k not in failed,
            }
        )
    return {
        "summary": summary,
        "gate": [{"check": "x", "passed": True, "rows_in": n, "rows_out": n}],
        "features": None,
        "error": f"Cycle {n + 1} is missing or failed the quality gate.",
        "n_cycles": n,
    }


def test_save_analysis_replaces_a_files_rows(conn):
    cur = conn.cursor()
    cycling_id = insert_file(cur)
    save_analysis(cycling_id, TEST_RECORD, fake_analysis(30), conn=conn)
    save_analysis(cycling_id, TEST_RECORD, fake_analysis(20), conn=conn)
    cur.execute(
        "SELECT count(*) FROM tbl_cycle_summary WHERE cycling_id = %s", (cycling_id,)
    )
    assert cur.fetchone()[0] == 20
    cur.execute(
        "SELECT n_cycles, feature_error, gate FROM tbl_cell_features "
        "WHERE cycling_id = %s",
        (cycling_id,),
    )
    n_cycles, error, gate = cur.fetchone()
    assert n_cycles == 20 and "Cycle 21" in error
    assert gate[0]["rows_in"] == 20
    cur.close()


def test_fade_cells_use_gate_passed_cycles_of_the_newest_file(conn):
    cur = conn.cursor()
    old_file = insert_file(cur)
    save_analysis(old_file, TEST_RECORD, fake_analysis(30), conn=conn)
    new_file = insert_file(cur)
    save_analysis(new_file, TEST_RECORD, fake_analysis(25, failed={3}), conn=conn)
    cur.close()
    cells = load_fade_cells([TEST_RECORD], conn=conn)
    numbers, capacities = cells[TEST_RECORD]
    assert len(numbers) == 24 and 3 not in numbers
    assert capacities[0] == pytest.approx(1.998)


def test_registry_round_trip(conn):
    cycles = list(range(1, 101))
    capacities = []
    for k in cycles:
        capacities.append(2 - 0.002 * k)
    result = score_cells({TEST_RECORD: (cycles, capacities)})[0]
    run_id = record_run(result, {"cuts": [0.3]}, "test", "abc123", conn=conn)
    runs = list_runs("cycling", conn=conn)
    saved = runs[0]
    assert saved["run_id"] == run_id
    assert saved["model"] == result["model"]
    assert saved["metric"] == pytest.approx(result["metric"])
    assert saved["r2"] == pytest.approx(result["r2"])
    assert saved["scatter"] == result["scatter"]
    assert saved["params"] == {"cuts": [0.3]}
    assert saved["code_version"] == "abc123"


def test_latest_only_keeps_newest_run_per_model(conn):
    first = record_run(standard_row(), conn=conn)
    second = record_run(standard_row(), conn=conn)
    run_ids = []
    for run in list_runs("test-type", latest_only=True, conn=conn):
        run_ids.append(run["run_id"])
    assert run_ids == [second] and first != second


def standard_row():
    return {
        "data_type": "test-type",
        "model": "Linear fade",
        "target": "x",
        "split_method": "x",
        "metric_name": "x",
        "metric": 1.0,
    }


def test_training_only_reads_approved_files(tmp_path):
    with pytest.raises(ValueError, match="not inside"):
        approved_files(tmp_path, "*.xlsx")
