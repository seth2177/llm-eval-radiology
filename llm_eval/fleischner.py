"""Follow-up recommendation for incidental pulmonary nodules.

Fleischner Society 2017 guidelines (MacMahon et al., Radiology 2017;284:228-243),
low-risk patient. Sizes are rounded to the nearest millimeter first, as the
guideline specifies.

Single nodule:
  solid  < 6 mm    no routine follow-up
  solid  6-8 mm    CT at 6-12 months
  solid  > 8 mm    consider CT at 3 months, PET/CT, or tissue sampling
  ground-glass < 6 mm   no routine follow-up
  ground-glass >= 6 mm  CT at 6-12 months to confirm persistence

Multiple solid nodules (managed by the most suspicious one; with only sizes known,
that is the largest):
  largest < 6 mm   no routine follow-up
  largest 6-8 mm   CT at 3-6 months, then consider CT at 18-24 months
  largest > 8 mm   CT at 3-6 months, then consider CT at 18-24 months

The multiple subsolid table is not implemented: the detector reports no density,
and the prompt and checker assume solid nodules.

The guideline does not apply to lung-cancer screening, patients under 35, known
cancer or immunosuppression. The prompt and checker assume the low-risk incidental
case; the README lists these limits.
"""
from __future__ import annotations

NONE = "no_routine_followup"
CT_6_12 = "ct_6_12_months"
CT_3_PET = "ct_3_months_pet_or_sampling"
CT_3_6 = "ct_3_6_months"                 # multiple nodules only

LABELS = {
    NONE: "no routine follow-up",
    CT_6_12: "CT in 6-12 months",
    CT_3_PET: "CT in 3 months, PET/CT, or tissue sampling",
    CT_3_6: "CT in 3-6 months, then consider CT at 18-24 months",
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


def multiple_category(sizes_mm: list[float]) -> str:
    """Multiple solid nodules, low risk: the largest decides, and 6-8 mm and > 8 mm share one row."""
    return NONE if round_mm(max(sizes_mm)) < 6 else CT_3_6


def management(sizes_mm: list[float]) -> str | None:
    """What the guideline says for a study's solid nodules: None when there are none."""
    if not sizes_mm:
        return None
    return category(sizes_mm[0]) if len(sizes_mm) == 1 else multiple_category(sizes_mm)
