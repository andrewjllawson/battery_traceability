"""Synthesis step checks (services/synthesis.py). No database needed.

    pytest -q tests/test_synthesis.py
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from services.synthesis import (MAX_STEPS, OTHER_STEP, STEP_CATEGORY, SYNTHESIS_STEPS,  # noqa: E402
                                SynthesisError, catalogue, clean_synthesis)


def test_catalogue_has_every_step_once():
    steps = [s for group in SYNTHESIS_STEPS.values() for s in group]
    assert len(steps) == len(set(steps)) == len(STEP_CATEGORY)
    assert [g["category"] for g in catalogue()] == list(SYNTHESIS_STEPS)


def test_clean_numbers_steps_and_adds_categories():
    out = clean_synthesis({"precursors": "  NiSO4,   MnSO4 ",
                           "steps": [{"step": "Co-precipitation", "description": " pH 11 "},
                                     {"step": "Calcination", "description": ""}]})
    assert out["precursors"] == "NiSO4, MnSO4"
    assert out["steps"] == [
        {"order": 1, "category": "Precursor preparation", "step": "Co-precipitation", "description": "pH 11"},
        {"order": 2, "category": "Thermal treatment", "step": "Calcination", "description": None},
    ]


def test_empty_list_is_allowed():
    assert clean_synthesis({"precursors": "", "steps": []}) == {"precursors": None, "steps": []}


@pytest.mark.parametrize("body", [
    None,
    {"steps": "Calcination"},                                   # not a list
    {"steps": [{"step": ""}]},                                  # no step chosen
    {"steps": [{"step": "Boiling"}]},                           # not in the list
    {"steps": [{"step": OTHER_STEP, "description": "  "}]},     # 'Other' needs a description
    {"steps": [{"step": "Sintering", "description": "x" * 501}]},
    {"steps": [{"step": "Sintering"}] * (MAX_STEPS + 1)},
    {"precursors": ["NiSO4"], "steps": []},                     # precursors must be text
])
def test_rejects_bad_input(body):
    with pytest.raises(SynthesisError):
        clean_synthesis(body)
