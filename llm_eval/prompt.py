"""The prompt that turns structured AI findings into a report impression.

Only de-identified finding fields go in: side, size, confidence. No UIDs, names,
dates or accession numbers, so nothing here needs a BAA to send to a hosted model.
Bump PROMPT_VERSION whenever the wording changes; it is recorded with every run.
"""
from __future__ import annotations

import json

from .cases import AIFinding

PROMPT_VERSION = "impression-v2"

SYSTEM = """You write the IMPRESSION section of a chest CT report from an AI nodule detector's output.

Rules:
- Use only the findings given. Do not add, remove or change any finding.
- For each nodule give the side (right or left lung) and the size in millimeters.
- Give one follow-up recommendation using the Fleischner Society 2017 guideline for a single
  incidental solid nodule in a low-risk adult aged 35 or over, not a screening study (size rounded to the nearest mm):
    < 6 mm: no routine follow-up.   6-8 mm: CT in 6-12 months.
    > 8 mm: consider CT in 3 months, PET/CT, or tissue sampling.
- If there are no findings, write exactly: No pulmonary nodule identified.
- Plain text, at most three sentences. No headings, no preamble."""


def user_message(findings: list[AIFinding]) -> str:
    payload = [{"type": "pulmonary nodule", "side": f.laterality, "size_mm": round(f.diameter_mm, 1),
                "detector_confidence": round(f.confidence, 2)} for f in findings]
    return "AI detector findings (JSON):\n" + json.dumps(payload)
