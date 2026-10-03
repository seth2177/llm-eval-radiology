"""Layer 1: how good is the detector, judged against ground truth.

Ground truth has one nodule at most per synthetic study, so matching is simple
(the model may still report several):
  TP        truth has a nodule, the model reports one on the same side
  WRONG_SIDE truth has a nodule, the model reports one only on the other side
  FN        truth has a nodule, the model reports none
  FP        truth has none, the model reports one
  TN        neither
WRONG_SIDE counts as a miss for sensitivity and is also reported on its own,
because a wrong-side finding is a different (and worse) failure than silence.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from . import fleischner as F
from .cases import Case

TP, FN, FP, TN, WRONG_SIDE = "TP", "FN", "FP", "TN", "WRONG_SIDE"


def outcome(case: Case) -> str:
    if case.truth is None:
        return FP if case.ai else TN
    if not case.ai:
        return FN
    if any(f.laterality == case.truth.laterality for f in case.ai):
        return TP
    return WRONG_SIDE


def management_ok(case: Case) -> bool:
    """A true positive whose measured size(s) still land in the right Fleischner category.

    A 5 mm nodule measured as 6.3 mm is found on the right side, but it moves the patient
    from "no routine follow-up" to "CT in 6-12 months". Attribution treats that as a
    detector error, not a correct report. Extra findings count too: with more than one,
    the multiple-nodule table applies.
    """
    if outcome(case) != TP:
        return False
    return F.management([f.diameter_mm for f in case.ai]) == F.category(case.truth.diameter_mm, case.truth.density)


def size_bin(mm: float) -> str:
    mm = F.round_mm(mm)
    if mm < 6:
        return "<6 mm"
    if mm <= 8:
        return "6-8 mm"
    return ">8 mm"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval. Small-n safe, unlike the normal approximation."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


@dataclass
class Rate:
    k: int
    n: int

    @property
    def value(self) -> float | None:
        return self.k / self.n if self.n else None

    def as_dict(self) -> dict:
        lo, hi = wilson(self.k, self.n)
        return {"k": self.k, "n": self.n, "value": self.value, "ci95": [round(lo, 3), round(hi, 3)]}


def metrics(cases: list[Case]) -> dict:
    outs = [(c, outcome(c)) for c in cases]
    pos = [(c, o) for c, o in outs if c.truth is not None]
    neg = [(c, o) for c, o in outs if c.truth is None]

    strata: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for c, o in pos:
        key = f"{c.truth.density} {size_bin(c.truth.diameter_mm)}"
        strata[key][1] += 1
        strata[key][0] += o == TP

    size_err = []
    for c, o in pos:
        if o == TP:
            f = next(f for f in c.ai if f.laterality == c.truth.laterality)
            size_err.append(f.diameter_mm - c.truth.diameter_mm)

    return {
        "n_cases": len(outs),
        "counts": {k: sum(o == k for _, o in outs) for k in (TP, FN, WRONG_SIDE, FP, TN)},
        "sensitivity": Rate(sum(o == TP for _, o in pos), len(pos)).as_dict(),
        "specificity": Rate(sum(o == TN for _, o in neg), len(neg)).as_dict(),
        "sensitivity_by_stratum": {k: Rate(v[0], v[1]).as_dict() for k, v in sorted(strata.items())},
        "size_error_mm": {
            "n": len(size_err),
            "mean": round(sum(size_err) / len(size_err), 2) if size_err else None,
            "mean_abs": round(sum(abs(e) for e in size_err) / len(size_err), 2) if size_err else None,
            "worst": round(max(size_err, key=abs), 2) if size_err else None,
        },
        "tp_wrong_followup_category": sum(o == TP and not management_ok(c) for c, o in pos),
        "tp_with_extra_findings": sum(o == TP and len(c.ai) > 1 for c, o in pos),
    }
