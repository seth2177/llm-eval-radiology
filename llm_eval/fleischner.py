"""Follow-up recommendation for a single incidental pulmonary nodule.

Fleischner Society 2017 guidelines (MacMahon et al., Radiology 2017;284:228-243),
single nodule, low-risk patient. Sizes are rounded to the nearest millimeter first,
as the guideline specifies.

  solid  < 6 mm    no routine follow-up
  solid  6-8 mm    CT at 6-12 months
  solid  > 8 mm    consider CT at 3 months, PET/CT, or tissue sampling
  ground-glass < 6 mm   no routine follow-up
  ground-glass >= 6 mm  CT at 6-12 months to confirm persistence

The guideline does not apply to lung-cancer screening, patients under 35, known
cancer or immunosuppression. The prompt and checker assume the low-risk incidental
case; the README lists these limits. Multiple nodules have their own table in the
guideline, which is not implemented here (the checker marks them OUT_OF_SCOPE).
"""
from __future__ import annotations

NONE = "no_routine_followup"
CT_6_12 = "ct_6_12_months"
CT_3_PET = "ct_3_months_pet_or_sampling"

LABELS = {
    NONE: "no routine follow-up",
    CT_6_12: "CT in 6-12 months",
    CT_3_PET: "CT in 3 months, PET/CT, or tissue sampling",
}


def round_mm(size_mm: float) -> int:
    """Round half up (6.5 -> 7); Python's round() would send 6.5 to 6."""
    return int(size_mm + 0.5)


def category(size_mm: float, density: str = "solid") -> str:
    mm = round_mm(size_mm)
    if density == "ground-glass":
        return NONE if mm < 6 else CT_6_12
    if mm < 6:
        return NONE
    if mm <= 8:
        return CT_6_12
    return CT_3_PET
