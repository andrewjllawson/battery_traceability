"""Synthesis steps for materials: how a powder was made (routers/synthesis.py).

Each material can have a list of raw materials / precursors and an ordered list of
steps, each picked from SYNTHESIS_STEPS plus an optional free-text description
(temperature, time, atmosphere, equipment...). The whole list is saved at once, so
reordering or removing a step is a single save.

Tables: tbl_material_synthesis (precursors, one row per material) and
tbl_material_synthesis_steps (one row per step). See migrations/002_material_synthesis.sql.
"""

from core.db import get_connection

# Step catalogue, grouped as in the dropdown. Add new steps here: the database stores
# the step name as text, so the list can grow without a database change.
SYNTHESIS_STEPS = {
    "Precursor preparation": [
        "Raw material weighing", "Precursor dissolution", "Solution preparation (metal salts)",
        "pH adjustment", "Chelating agent addition (e.g. ammonia)", "Co-precipitation",
        "Precursor ageing", "Precursor filtration", "Precursor washing", "Precursor drying",
    ],
    "Mixing and size reduction": [
        "Dry mixing", "Wet mixing", "Mortar and pestle grinding", "Planetary ball milling",
        "Attritor milling", "Jet milling", "High-energy ball milling (mechanochemical)",
        "Sieving", "Pelletising",
    ],
    "Solution and wet-chemical routes": [
        "Sol-gel synthesis", "Hydrothermal synthesis", "Solvothermal synthesis", "Spray drying",
        "Freeze drying", "Solution combustion synthesis", "Molten salt synthesis", "Electrospinning",
    ],
    "Thermal treatment": [
        "Pre-calcination (pre-sintering)", "Calcination", "Sintering", "Annealing",
        "Carbothermal reduction", "Pyrolysis", "Carbonisation", "Activation (hard carbon)",
        "Graphitisation", "Quenching", "Slow cooling",
    ],
    "Atmosphere-controlled steps": [
        "Oxygen atmosphere heat treatment", "Air heat treatment",
        "Inert atmosphere (Ar/N₂) heat treatment", "Reducing atmosphere (H₂/Ar) heat treatment",
    ],
    "Post-synthesis processing": [
        "Lithiation (Li source addition)", "Sodiation (Na source addition)", "Water washing",
        "Solvent washing", "Acid washing", "Post-wash drying", "Vacuum drying", "De-agglomeration",
        "Classification (size)", "Magnetic separation (impurity removal)",
    ],
    "Modification and coating": [
        "Doping", "Surface coating (wet)", "Surface coating (dry)", "Atomic layer deposition (ALD)",
        "Carbon coating", "Core-shell formation", "Compositing (e.g. with carbon or silicon)",
    ],
    "Handling and storage": [
        "Glovebox transfer", "Vacuum sealing", "Dry-room storage",
    ],
    "Other": [
        "Other (describe)",
    ],
}

OTHER_STEP = "Other (describe)"
MAX_STEPS = 40
MAX_DESCRIPTION = 500
MAX_PRECURSORS = 500

# step name -> category, for checking a saved step and showing its group
STEP_CATEGORY = {step: category for category, steps in SYNTHESIS_STEPS.items() for step in steps}


class SynthesisError(ValueError):
    """The steps sent by the browser were rejected; the message is shown to the user."""


def catalogue():
    """The dropdown contents as a list, for the page: [{"category", "steps": [...]}, ...]."""
    return [{"category": category, "steps": steps} for category, steps in SYNTHESIS_STEPS.items()]


def clean_synthesis(data) -> dict:
    """Check what the editor sent: {"precursors": str, "steps": [{"step", "description"}]}.
    Returns the tidied version, or raises SynthesisError."""
    if not isinstance(data, dict):
        raise SynthesisError("Expected precursors and a list of steps.")
    precursors = data.get("precursors") or ""
    if not isinstance(precursors, str):
        raise SynthesisError("Raw materials must be text.")
    precursors = " ".join(precursors.split())
    if len(precursors) > MAX_PRECURSORS:
        raise SynthesisError(f"Raw materials can be at most {MAX_PRECURSORS} characters.")

    steps = data.get("steps") or []
    if not isinstance(steps, list):
        raise SynthesisError("Steps must be a list.")
    if len(steps) > MAX_STEPS:
        raise SynthesisError(f"A material can have at most {MAX_STEPS} steps.")

    cleaned = []
    for number, item in enumerate(steps, start=1):
        if not isinstance(item, dict):
            raise SynthesisError(f"Step {number} is not valid.")
        step = (item.get("step") or "").strip()
        description = item.get("description") or ""
        if not isinstance(description, str):
            raise SynthesisError(f"Step {number}: the description must be text.")
        description = description.strip()
        if not step:
            raise SynthesisError(f"Step {number}: choose a step from the list.")
        if step not in STEP_CATEGORY:
            raise SynthesisError(f"Step {number}: '{step}' is not in the list of synthesis steps.")
        if step == OTHER_STEP and not description:
            raise SynthesisError(f"Step {number}: describe the step when you choose 'Other'.")
        if len(description) > MAX_DESCRIPTION:
            raise SynthesisError(f"Step {number}: the description can be at most {MAX_DESCRIPTION} characters.")
        cleaned.append({"order": number, "category": STEP_CATEGORY[step], "step": step,
                        "description": description or None})
    return {"precursors": precursors or None, "steps": cleaned}


def material_exists(cur, material_id: str) -> bool:
    cur.execute("SELECT 1 FROM tbl_materials WHERE material_id = %s", (material_id,))
    return cur.fetchone() is not None


def get_synthesis(material_id: str) -> dict:
    """Saved precursors and steps for one material (empty if none were saved)."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT precursors, updated_by, updated_at FROM tbl_material_synthesis WHERE material_id = %s",
                    (material_id,))
        head = cur.fetchone()
        cur.execute(
            "SELECT step_order, step, description FROM tbl_material_synthesis_steps "
            "WHERE material_id = %s ORDER BY step_order", (material_id,)
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()
    steps = [{"order": int(order), "category": STEP_CATEGORY.get(step, "Other"), "step": step,
              "description": description} for order, step, description in rows]
    return {
        "precursors": head[0] if head else None,
        "updated_by": head[1] if head else None,
        "updated_at": head[2].isoformat() if head and hasattr(head[2], "isoformat") else (str(head[2]) if head and head[2] else None),
        "steps": steps,
    }


def save_synthesis(material_id: str, data: dict, user_name: str = None) -> dict:
    """Replace a material's precursors and steps with `data` (already checked by clean_synthesis),
    in one transaction. Returns the saved version. Raises SynthesisError if the material is missing."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        if not material_exists(cur, material_id):
            raise SynthesisError(f"Material {material_id} doesn't exist.")
        cur.execute(
            "INSERT INTO tbl_material_synthesis (material_id, precursors, updated_by, updated_at) "
            "VALUES (%s, %s, %s, now()) "
            "ON CONFLICT (material_id) DO UPDATE SET precursors = EXCLUDED.precursors, "
            "updated_by = EXCLUDED.updated_by, updated_at = now()",
            (material_id, data["precursors"], user_name),
        )
        cur.execute("DELETE FROM tbl_material_synthesis_steps WHERE material_id = %s", (material_id,))
        for s in data["steps"]:
            cur.execute(
                "INSERT INTO tbl_material_synthesis_steps (material_id, step_order, step, description) "
                "VALUES (%s, %s, %s, %s)",
                (material_id, s["order"], s["step"], s["description"]),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()
    return get_synthesis(material_id)
