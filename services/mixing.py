"""Eirich mixer logs: read the spreadsheet, split a run into stages, save it.

Reading and stage detection need no database (parse_mixer_xlsx, find_stages), so they
can be tested on their own. save_mixing_run writes the four tbl_mixing_* tables.
"""

import hashlib
import io

import pandas as pd

ROTOR_ON_RPM = 100  # above this the rotor counts as running
MIN_STAGE_S = 5  # shorter blips are ignored

# Spreadsheet column (units stripped) -> database column
LOG_COLUMNS = {
    "RotorRev": "rotor_rpm",
    "RotorPower": "rotor_power_w",
    "MixingPanRev": "pan_rpm",
    "MixingPanPower": "pan_power_w",
    "Temperature": "temperature_c",
    "Total Power": "total_power_w",
    "Cumulative Energy (J)": "cumulative_energy_j",
}

# Summary block (columns P:Q) -> tbl_mixing_run column
SUMMARY_LABELS = {
    "Total Energy": "total_energy_kj",
    "1st Addition Energy": "first_addition_energy_kj",
    "Knead Energy": "knead_energy_kj",
    "Peak Power": "peak_power_w",
}


class MixingError(ValueError):
    """The file is not a usable mixer log; the message is shown to the user."""


def parse_mixer_xlsx(data: bytes):
    """Returns (summary dict, log DataFrame with t_s and the LOG_COLUMNS names).

    Rows without a timestamp are dropped, and so is any row whose time does not
    move forward (the export repeats or rewinds the clock at the very end).
    """
    try:
        sheet = pd.read_excel(io.BytesIO(data), header=0)
    except Exception as e:  # unreadable or not a workbook
        raise MixingError(f"Could not read the spreadsheet: {e}") from None

    sheet.columns = [str(c).split(" [")[0].strip() for c in sheet.columns]
    needed = {"Date", "Time", "RotorRev", "Total Power"}
    missing = needed - set(sheet.columns)
    if missing:
        raise MixingError(f"Missing columns: {', '.join(sorted(missing))}.")

    log = sheet.dropna(subset=["Date", "Time"]).copy()
    stamp = pd.to_datetime(
        log["Date"].dt.strftime("%Y-%m-%d") + " " + log["Time"].astype(str)
    )
    log["t_s"] = (stamp - stamp.iloc[0]).dt.total_seconds().round().astype(int)
    log = log[log["t_s"] > log["t_s"].cummax().shift(fill_value=-1)]
    if len(log) < 10:
        raise MixingError("The log has fewer than 10 usable rows.")

    summary = {}
    if {"Unnamed: 15", "Unnamed: 16"} <= set(sheet.columns):
        for label, value in zip(sheet["Unnamed: 15"], sheet["Unnamed: 16"]):
            if label in SUMMARY_LABELS and pd.notna(value):
                summary[SUMMARY_LABELS[label]] = float(value)
    summary["started_at"] = stamp.iloc[0].to_pydatetime()
    summary["duration_s"] = int(log["t_s"].iloc[-1])

    keep = ["t_s", *[c for c in LOG_COLUMNS if c in log.columns]]
    log = log[keep].rename(columns=LOG_COLUMNS).reset_index(drop=True)
    return summary, log


def find_stages(log: pd.DataFrame) -> pd.DataFrame:
    """One row per rotor-on period: stage_no, start_s, duration_s, pause_before_s,
    rotor_rpm, pan_rpm, mean_power_w, peak_power_w, energy_kj."""
    on = log["rotor_rpm"] > ROTOR_ON_RPM
    block = (on != on.shift()).cumsum()
    rows, last_end = [], -1
    for _, part in log[on].groupby(block[on]):
        if len(part) < MIN_STAGE_S:
            continue
        start, end = int(part["t_s"].iloc[0]), int(part["t_s"].iloc[-1])
        power = part["total_power_w"]
        rows.append(
            {
                "stage_no": len(rows) + 1,
                "start_s": start,
                "duration_s": end - start + 1,
                "pause_before_s": max(start - last_end - 1, 0),
                "rotor_rpm": float(part["rotor_rpm"].median()),
                "pan_rpm": float(part["pan_rpm"].median()),
                "mean_power_w": float(power.mean()),
                "peak_power_w": float(power.max()),
                "energy_kj": float(power.sum() / 1000),  # 1 s per row
            }
        )
        last_end = end
    return pd.DataFrame(rows)


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_mixing_run(conn, mixing_id, project, made_by, filename, data, ingredients=()):
    """Parse the file and write the run, its stages, its log and its ingredients in
    one transaction. ingredients: (material_id, role, mass_g) tuples.
    Raises MixingError for a bad file or one that was loaded before.
    """
    from psycopg2.extras import execute_values

    summary, log = parse_mixer_xlsx(data)
    stages = find_stages(log)
    digest = file_hash(data)
    cur = conn.cursor()
    try:
        cur.execute("SELECT mixing_id FROM tbl_mixing_run WHERE source_sha256 = %s", (digest,))
        if cur.fetchone():
            raise MixingError("This log has already been loaded.")
        cols = ["mixing_id", "project", "made_by", "source_filename", "source_sha256", *summary]
        cur.execute(
            f"INSERT INTO tbl_mixing_run ({', '.join(cols)}) "
            f"VALUES ({', '.join(['%s'] * len(cols))})",
            [mixing_id, project, made_by, filename, digest, *summary.values()],
        )
        if len(stages):
            execute_values(
                cur,
                "INSERT INTO tbl_mixing_stage (mixing_id, stage_no, start_s, duration_s, "
                "pause_before_s, rotor_rpm, pan_rpm, mean_power_w, peak_power_w, energy_kj) VALUES %s",
                [(mixing_id, *(r.tolist())) for r in stages[
                    ["stage_no", "start_s", "duration_s", "pause_before_s", "rotor_rpm",
                     "pan_rpm", "mean_power_w", "peak_power_w", "energy_kj"]].itertuples(index=False)],
            )
        log_cols = ["t_s", *[c for c in LOG_COLUMNS.values() if c in log.columns]]
        execute_values(
            cur,
            f"INSERT INTO tbl_mixing_log (mixing_id, {', '.join(log_cols)}) VALUES %s",
            [(mixing_id, *row) for row in log[log_cols].where(log.notna(), None).itertuples(index=False)],
            page_size=1000,
        )
        if ingredients:
            execute_values(
                cur,
                "INSERT INTO tbl_mixing_ingredient (mixing_id, material_id, role, mass_g) VALUES %s",
                [(mixing_id, *i) for i in ingredients],
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
    return {"stages": len(stages), "log_rows": len(log), **summary}
