"""Cycling upload works out the features straight away (routers/cycling.py,
services/cycle_features.analyse_and_save).

Needs a database with the dev seed (dev / dev login); skipped if none can be
reached. galvani and the file analysis are replaced by fakes, so no real .mpr is
needed. Uploaded rows and files are deleted afterwards.

pytest -q tests/test_cycling_upload.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from galvani import BioLogic
from psycopg2 import Error as Psycopg2Error
from psycopg2 import OperationalError

import main
import services.cycle_features
from core.db import get_connection
from scripts import extract_cycle_features
from services.cycle_features import REDUCED_FEATURES
from services.cycling import UPLOAD_DIR

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Contains -CC so detect_record_type sees a coin cell; no tbl_coincell row, so visible
CELL = "TEST-UP-CC1"
FILENAME = "fake.mpr"


def database_available() -> bool:
    try:
        get_connection().close()
        return True
    except OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not database_available(), reason="no database reachable (set DATABASE_URL)"
)


class FakeMPR:
    """Stands in for BioLogic.MPRfile: 4 points over 2 half-cycles."""

    def __init__(self, path):
        self.data = pd.DataFrame(
            {
                "Q charge/discharge/mA.h": [0.0, 1.5, 2.0, 1.0],
                "half cycle": [0, 0, 1, 1],
            }
        )


def fake_analysis(features):
    """What analyse_file gives for a 2-cycle file, with or without features."""
    summary = []
    for k in (1, 2):
        summary.append(
            {
                "cycle_no": k,
                "discharge_mah": 2.0,
                "charge_mah": 2.02,
                "coulombic_eff": 0.99,
                "charge_time_s": 3600.0,
                "passed_gate": k == 1,
            }
        )
    return {
        "summary": summary,
        "gate": [{"check": "Readable .mpr file", "passed": True}],
        "features": features,
        "error": None if features else "Cycle 3 is missing or failed the quality gate.",
        "n_cycles": 2,
    }


def full_features():
    features = {"v_grid_max": 4.2, "v_grid_min": 3.0}
    for name in REDUCED_FEATURES:
        features[name] = 1.0
    return features


@pytest.fixture(scope="module")
def client():
    """A logged-in browser session (dev / dev from the dev seed)."""
    c = TestClient(main.app, follow_redirects=False)
    response = c.post("/login", data={"username": "dev", "password": "dev"})
    assert response.status_code == 303, "dev login failed - is the dev seed loaded?"
    return c


@pytest.fixture(autouse=True)
def clean_up(monkeypatch):
    """Fake galvani; delete the test cell's rows and saved file afterwards (its
    summary and feature rows go with the file)."""
    monkeypatch.setattr(BioLogic, "MPRfile", FakeMPR)
    try:
        yield
    finally:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM tbl_cycling_data WHERE record_id = %s", (CELL,))
        conn.commit()
        cur.close()
        conn.close()
        (UPLOAD_DIR / f"{CELL}_{FILENAME}").unlink(missing_ok=True)


def upload(client):
    return client.post(
        "/cycling/upload",
        data={"record_id": CELL},
        files={"file": (FILENAME, b"not really an mpr")},
    )


def stored_rows():
    """(files, summary rows, feature error) stored for the test cell."""
    conn = get_connection(read_only=True)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM tbl_cycling_data WHERE record_id = %s", (CELL,))
    files = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM tbl_cycle_summary WHERE record_id = %s", (CELL,))
    summary = cur.fetchone()[0]
    cur.execute(
        "SELECT feature_error FROM tbl_cell_features WHERE record_id = %s", (CELL,)
    )
    features = cur.fetchall()
    cur.close()
    conn.close()
    return files, summary, features


def test_upload_saves_features(client, monkeypatch):
    monkeypatch.setattr(
        services.cycle_features,
        "analyse_file",
        lambda p: fake_analysis(full_features()),
    )
    response = upload(client)
    assert response.status_code == 200
    assert "Analysis: features saved" in response.text
    assert stored_rows() == (1, 2, [(None,)])


def test_upload_without_features_says_why(client, monkeypatch):
    monkeypatch.setattr(
        services.cycle_features, "analyse_file", lambda p: fake_analysis(None)
    )
    response = upload(client)
    assert "Analysis: Cycle 3 is missing or failed the quality gate." in response.text
    reason = "Cycle 3 is missing or failed the quality gate."
    assert stored_rows() == (1, 2, [(reason,)])


def test_failed_analysis_keeps_upload(client, monkeypatch):
    def broken_save(*args, **kwargs):
        raise Psycopg2Error("database went away")

    monkeypatch.setattr(
        services.cycle_features, "analyse_file", lambda p: fake_analysis(None)
    )
    monkeypatch.setattr(services.cycle_features, "save_analysis", broken_save)
    response = upload(client)
    assert response.status_code == 200
    assert "Uploaded and parsed 4 data points" in response.text
    assert f"scripts.extract_cycle_features --record {CELL}" in response.text
    assert stored_rows() == (1, 0, [])


def test_script_output_unchanged(client, monkeypatch, capsys):
    monkeypatch.setattr(
        services.cycle_features,
        "analyse_file",
        lambda p: fake_analysis(full_features()),
    )
    upload(client)
    capsys.readouterr()
    extract_cycle_features.main(["--record", CELL])
    line = capsys.readouterr().out.strip()
    assert line.startswith(f"{CELL} (file ")
    assert line.endswith("): 2 cycles, 1 passed the gate; features saved")
