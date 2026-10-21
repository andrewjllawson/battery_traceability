"""Synthesis steps on material cards: read and save.

GET /materials/{id}/synthesis   precursors and steps for one material (JSON)
PUT /materials/{id}/synthesis   replace them with what the card's editor sends (JSON)

Both need a login: synthesis routes are process know-how. The card page shows the
section (templates/_synthesis.html, static/js/synthesis.js); the step list and
checks are in services/synthesis.py.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from services.records import detect_record_type
from services.synthesis import SynthesisError, clean_synthesis, get_synthesis, save_synthesis

router = APIRouter()


def _not_logged_in(request: Request):
    """JSON endpoints answer 401 instead of redirecting to /login."""
    if not request.session.get("user_id"):
        return JSONResponse({"error": "Please log in again."}, status_code=401)
    return None


@router.get("/materials/{material_id}/synthesis")
def material_synthesis(request: Request, material_id: str):
    """Saved precursors and synthesis steps for a material."""
    denied = _not_logged_in(request)
    if denied:
        return denied
    if detect_record_type(material_id) != "material":
        return JSONResponse({"error": "Synthesis steps are only kept for materials."}, status_code=404)
    return JSONResponse({"material_id": material_id, **get_synthesis(material_id)})


@router.put("/materials/{material_id}/synthesis")
async def update_material_synthesis(request: Request, material_id: str):
    """Replace a material's precursors and steps with the editor's list."""
    denied = _not_logged_in(request)
    if denied:
        return denied
    if detect_record_type(material_id) != "material":
        return JSONResponse({"error": "Synthesis steps are only kept for materials."}, status_code=404)
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": "The steps could not be read. Reload the page and try again."}, status_code=400)
    try:
        saved = save_synthesis(material_id, clean_synthesis(body), request.session.get("full_name"))
    except SynthesisError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"material_id": material_id, **saved})
