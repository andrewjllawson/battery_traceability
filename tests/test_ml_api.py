"""Cycle feature, fade forecast and model registry endpoints (routers/cycling.py,
routers/ml.py).

Needs a database with the dev seed (dev / dev login); skipped if none can be
reached. Fixtures add made-up cells and registry runs and delete them afterwards.

pytest -q tests/test_ml_api.py
"""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from psycopg2 import OperationalError

import main
import routers.cycling
import routers.ml
from core.db import get_connection
from services.model_registry import record_run

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Contains -CC so detect_record_type sees a coin cell; no tbl_coincell row, so visible
CELL = "TEST-ML-CC1"
SHORT_CELL = "TEST-ML-CC2"
FLAT_CELL = "TEST-ML-CC3"
RAW_CELL = "TEST-ML-CC4"  # a file that has not been analysed
TEST_TYPE = "test-api"


def database_available() -> bool:
    try:
        get_connection().close()
        return True
    except OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not database_available(), reason="no database reachable (set DATABASE_URL)"
)


@pytest.fixture(scope="module")
def client():
    """A logged-in browser session (dev / dev from the dev seed)."""
    c = TestClient(main.app, follow_redirects=False)
    response = c.post("/login", data={"username": "dev", "password": "dev"})
    assert response.status_code == 303, "dev login failed - is the dev seed loaded?"
    return c


def add_cell(cur, record_id, capacities):
    """A stored file with one gate-passed summary row per capacity (mAh)."""
    from services.cycle_features import save_analysis

    cur.execute(
        "INSERT INTO tbl_cycling_data (record_id, file_path, filename) "
        "VALUES (%s, %s, %s) RETURNING cycling_id",
        (record_id, "uploads/cycling_data/fake.mpr", "fake.mpr"),
    )
    cycling_id = cur.fetchone()[0]
    if capacities is None:
        return
    summary = []
    for k, capacity in enumerate(capacities, start=1):
        summary.append(
            {
                "cycle_no": k,
                "discharge_mah": capacity,
                "charge_mah": capacity / 0.99,
                "coulombic_eff": 0.99,
                "charge_time_s": 3600.0,
                "passed_gate": True,
            }
        )
    analysis = {
        "summary": summary,
        "gate": [{"check": "Cycle has a discharge", "passed": True}],
        "features": None,
        "error": f"Cycle {len(capacities) + 1} is missing or failed the quality gate.",
        "n_cycles": len(capacities),
    }
    save_analysis(cycling_id, record_id, analysis, conn=cur.connection)


@pytest.fixture(scope="module")
def cells():
    """Made-up cells: fading, too short, flat, and not analysed. Deleted afterwards
    (their summary and feature rows go with the file)."""
    fading = []
    for k in range(1, 121):
        fading.append(2 - 0.003 * k)
    conn = get_connection()
    cur = conn.cursor()
    add_cell(cur, CELL, fading)
    add_cell(cur, SHORT_CELL, [2.0] * 12)
    add_cell(cur, FLAT_CELL, [2.0] * 120)
    add_cell(cur, RAW_CELL, None)
    conn.commit()
    try:
        yield
    finally:
        cur.execute(
            "DELETE FROM tbl_cycling_data WHERE record_id = ANY(%s)",
            ([CELL, SHORT_CELL, FLAT_CELL, RAW_CELL],),
        )
        conn.commit()
        cur.close()
        conn.close()


@pytest.fixture
def run_ids():
    """Two registry runs of a test data type, deleted afterwards."""
    result = {
        "data_type": TEST_TYPE,
        "model": "Linear fade",
        "target": "Discharge capacity of later cycles",
        "split_method": "x",
        "metric_name": "MAPE (%)",
        "metric": 1.5,
        "baseline_metric": 4.0,
        "r2": 0.9,
        "scatter": {"actual": [1.0], "predicted": [1.1]},
    }
    ids = [record_run(result), record_run(result)]
    try:
        yield ids
    finally:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM tbl_model_runs WHERE data_type = %s", (TEST_TYPE,))
        conn.commit()
        cur.close()
        conn.close()


@pytest.mark.parametrize(
    "url",
    [
        f"/api/cycling/{CELL}/features",
        f"/api/ml/cycle-life/{CELL}",
        "/api/ml/models",
        f"/api/cycling/{CELL}/delta-q",
        "/cycling",
    ],
)
def test_login_required(url):
    response = TestClient(main.app, follow_redirects=False).get(url)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_features_of_an_analysed_file(client, cells):
    response = client.get(f"/api/cycling/{CELL}/features")
    assert response.status_code == 200
    data = response.json()
    assert data["extracted"] and data["filename"] == "fake.mpr"
    assert data["n_cycles"] == 120
    assert len(data["cycles"]["cycle_no"]) == 120
    assert data["cycles"]["discharge_mah"][0] == pytest.approx(1.997)
    assert data["features"] is None  # 120 cycles of made-up data, no ΔQ(V)
    assert "Cycle 121" in data["feature_error"]
    assert data["gate"][0]["check"] == "Cycle has a discharge"


def test_features_of_a_file_not_yet_analysed(client, cells):
    response = client.get(f"/api/cycling/{RAW_CELL}/features")
    assert response.status_code == 404
    assert "extract_cycle_features" in response.json()["error"]


def test_forecast(client, cells):
    response = client.get(f"/api/ml/cycle-life/{CELL}")
    assert response.status_code == 200
    data = response.json()
    assert data["record_id"] == CELL
    assert data["model"] == "Linear fade"
    # 80% of 1.997 mAh is reached at (2 - 0.8 * 1.997) / 0.003 = 134.1 cycles
    assert data["eol_cycle"] == pytest.approx(134.1, abs=0.1)
    assert data["eol_low"] <= data["eol_cycle"] <= data["eol_high"]
    assert data["warning"] is None
    assert "held-out" in data["method"]
    assert len(data["curve"]["cycle"]) == len(data["curve"]["fit"])


def test_flat_cell_gets_a_warning(client, cells):
    data = client.get(f"/api/ml/cycle-life/{FLAT_CELL}").json()
    assert data["eol_cycle"] is None
    assert data["extrapolated"] and data["warning"]


def test_short_test_cannot_be_forecast(client, cells):
    response = client.get(f"/api/ml/cycle-life/{SHORT_CELL}")
    assert response.status_code == 422
    assert "at least" in response.json()["error"]


@pytest.mark.parametrize(
    "url",
    [
        "/api/cycling/NO-SUCH-CC9/features",
        "/api/ml/cycle-life/NO-SUCH-CC9",
        f"/api/ml/cycle-life/{RAW_CELL}",
    ],
)
def test_missing_data_is_not_found(client, cells, url):
    assert client.get(url).status_code == 404


@pytest.mark.parametrize(
    "module, url",
    [
        (routers.cycling, f"/api/cycling/{CELL}/features"),
        (routers.ml, f"/api/ml/cycle-life/{CELL}"),
        (routers.cycling, f"/api/cycling/{CELL}/delta-q"),
    ],
)
def test_hidden_record_is_not_found(client, cells, monkeypatch, module, url):
    """The endpoint asks record_visible about this record and hides it if told to."""
    asked = []

    def not_visible(record_id, allowed):
        asked.append(record_id)
        return False

    monkeypatch.setattr(module, "record_visible", not_visible)
    response = client.get(url)
    assert response.status_code == 404
    assert "error" in response.json()
    assert asked == [CELL]


def test_models_list(client, run_ids):
    response = client.get(f"/api/ml/models?data_type={TEST_TYPE}")
    assert response.status_code == 200
    runs = response.json()["runs"]
    assert [run["run_id"] for run in runs] == list(reversed(run_ids))
    run = runs[0]
    assert run["metric"] == 1.5 and run["baseline_metric"] == 4.0
    assert run["r2"] == 0.9
    assert run["scatter"] == {"actual": [1.0], "predicted": [1.1]}


def test_models_latest_only(client, run_ids):
    runs = client.get(f"/api/ml/models?data_type={TEST_TYPE}&latest=1").json()["runs"]
    assert [run["run_id"] for run in runs] == [run_ids[1]]


# --- The cycling page (routers/cycling.py, templates/cycling.html) ---------------


def test_cycling_page_lists_cells(client, cells):
    response = client.get(f"/cycling?cell={CELL}")
    assert response.status_code == 200
    assert f'data-id="{CELL}"' in response.text
    assert f'data-selected="{CELL}"' in response.text
    assert "not analysed yet" in response.text  # RAW_CELL
    assert "/static/js/cycling.js" in response.text


def test_cycling_page_hides_cells_from_other_projects(client, cells, monkeypatch):
    def only_others(record_id, allowed):
        return record_id != CELL

    monkeypatch.setattr(routers.cycling, "record_visible", only_others)
    response = client.get("/cycling")
    assert f'data-id="{CELL}"' not in response.text
    assert f'data-id="{SHORT_CELL}"' in response.text


def test_nav_links_to_the_cycling_page(client):
    assert 'href="/cycling"' in client.get("/mixing").text


def test_delta_q_of_an_unreadable_file(client, cells):
    """The made-up cells have no file on disk, so the curve cannot be read."""
    response = client.get(f"/api/cycling/{CELL}/delta-q")
    assert response.status_code == 422
    assert "Could not read the file" in response.json()["error"]


def test_delta_q_of_a_cell_without_data(client):
    assert client.get("/api/cycling/NO-SUCH-CC9/delta-q").status_code == 404
