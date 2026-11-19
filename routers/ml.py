"""Machine learning results: the model registry and the cycle-life (fade) forecast.

Read-only. Models are scored by scripts/train_cycle_life.py; the logic lives in
services/model_registry.py and services/fade_model.py.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from core.auth import get_allowed_projects, require_login
from services.cycle_features import load_fade_cells
from services.fade_model import (
    END_OF_LIFE,
    MAX_EXTRAPOLATION,
    SPLIT_METHOD,
    FadeError,
    forecast_cell,
)
from services.model_registry import list_runs
from services.records import record_visible

router = APIRouter()

NOT_FOUND = "No extracted cycling data for this record"
METHOD = (
    f"The curve that best predicted this cell's later cycles ({SPLIT_METHOD.lower()}) "
    f"is refitted on every cycle. End of life is {END_OF_LIFE:.0%} of the best of "
    "the first 10 cycles. The 80% band comes from this cell's own held-out errors, "
    "so it is not calibrated across cells."
)
EXTRAPOLATION_WARNING = (
    f"The forecast is more than {MAX_EXTRAPOLATION}x beyond the last measured cycle, "
    "or the cell is not fading yet: treat it as a rough guide."
)


@router.get("/api/ml/models")
def ml_models(request: Request, data_type: str = None, latest: bool = False):
    """Model registry runs, newest first. Aggregate scores only, no records, so
    every logged-in user may see them. ?latest=1 keeps the newest run per model.
    """
    redirect = require_login(request)
    if redirect:
        return redirect
    return {"runs": list_runs(data_type or None, latest_only=latest)}


@router.get("/api/ml/cycle-life/{record_id}")
def ml_cycle_life(request: Request, record_id: str):
    """The capacity fade forecast for one cell, from its newest extracted file."""
    redirect = require_login(request)
    if redirect:
        return redirect
    if not record_visible(record_id, get_allowed_projects(request)):
        return JSONResponse({"error": NOT_FOUND}, status_code=404)
    cells = load_fade_cells([record_id])
    if record_id not in cells:
        return JSONResponse({"error": NOT_FOUND}, status_code=404)
    cycle_no, capacity = cells[record_id]
    try:
        forecast = forecast_cell(cycle_no, capacity)
    except FadeError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    forecast["record_id"] = record_id
    forecast["method"] = METHOD
    forecast["warning"] = EXTRAPOLATION_WARNING if forecast["extrapolated"] else None
    return forecast
