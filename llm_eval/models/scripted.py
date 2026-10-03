"""A deterministic stand-in for an LLM, with known, planted mistakes.

Why it exists: an evaluation harness is only trustworthy if it is itself tested.
This model writes a correct impression, then (at a chosen rate) plants exactly one
mistake of a known type and records it. The checker never sees that record; the
meta-evaluation compares what was planted with what the checker found. If the
checker misses a planted error, or flags a clean report, the test suite fails.

Phrasing varies across several templates on purpose, so the checker cannot pass by
matching one fixed sentence.
"""
from __future__ import annotations

import hashlib
import random

from .. import fleischner as F
from ..cases import AIFinding
from .base import ImpressionModel

HALLUCINATION, OMISSION, LATERALITY, SIZE, FOLLOWUP_WRONG, FOLLOWUP_MISSING = (
    "HALLUCINATION", "OMISSION", "LATERALITY", "SIZE", "FOLLOWUP_WRONG", "FOLLOWUP_MISSING")
ERROR_TYPES = (HALLUCINATION, OMISSION, LATERALITY, SIZE, FOLLOWUP_WRONG, FOLLOWUP_MISSING)
POSITIVE_ONLY = {OMISSION, LATERALITY, SIZE, FOLLOWUP_WRONG, FOLLOWUP_MISSING}

FINDING_TEMPLATES = (
    "{Side} lung nodule measuring {size} mm.",
    "{size} mm pulmonary nodule in the {side} lung.",
    "Solitary {side} pulmonary nodule, {size} mm.",
    "There is a {size} mm nodule in the {side} lung.",
    "{Side} lung nodule measuring {size} mm, no additional nodules identified.",   # negation in the same sentence
    "A {size}-mm solid nodule is present in the {side} lung, without other pulmonary nodules.",
)
FOLLOWUP_TEMPLATES = {
    F.NONE: ("No routine follow-up is recommended.", "No routine follow-up needed per Fleischner 2017.",
             "Follow-up imaging is not required."),
    F.CT_6_12: ("Recommend follow-up CT in 6-12 months.", "Follow-up chest CT in 6 to 12 months per Fleischner 2017.",
                "Recommend CT in 6-12 months; PET/CT is not needed.",               # a negated alternative
                "Follow-up CT in 6-12 months, then consider CT at 18-24 months."),
    F.CT_3_PET: ("Consider CT in 3 months, PET/CT, or tissue sampling.",
                 "Recommend CT at 3 months, PET/CT or tissue sampling."),
}
# Wrong on purpose: a recommendation that *rules out* the correct category, or names a non-Fleischner interval.
WRONG_FOLLOWUP_EXTRA = {
    F.CT_3_PET: ("PET/CT and tissue sampling are not indicated.",),
    F.CT_6_12: ("Recommend follow-up CT in 12 months.",),
    F.NONE: ("Recommend CT in 6 months.",),
}
NEGATIVE = ("No pulmonary nodule identified.", "No pulmonary nodules.", "No suspicious pulmonary nodule is seen.")

# More than one finding: one sentence per nodule (or two joined by "and"), then one recommendation for the study.
MULTI_FINDING_TEMPLATES = (FINDING_TEMPLATES[0], FINDING_TEMPLATES[1], FINDING_TEMPLATES[3],
    "{Side} lung nodule, {size} mm.",
    "A {size} mm nodule is also present in the {side} lung.",
)
MULTI_JOINED = "A {size} mm {side} lung nodule and a {size2} mm {side2} lung nodule."
MULTI_OPENERS = ("", "", "Multiple pulmonary nodules. ", "Multiple solid pulmonary nodules. ")
MULTI_FOLLOWUP_TEMPLATES = {
    F.NONE: ("No routine follow-up is recommended.", "No routine follow-up needed per Fleischner 2017 for multiple "
             "nodules.", "Follow-up imaging is not required for these nodules."),
    F.CT_3_6: ("Recommend CT in 3-6 months, then consider CT at 18-24 months.",
               "For multiple nodules, follow-up CT at 3 to 6 months is recommended.",
               "Follow-up chest CT in 3-6 months per Fleischner 2017 for multiple solid nodules; PET/CT is not needed.",
               "Recommend CT in three to six months."),
}
# Wrong on purpose. Includes the single-nodule recommendation for the largest nodule, the classic mistake.
MULTI_WRONG_FOLLOWUP = {
    F.NONE: ("Recommend CT in 3-6 months.", "Recommend CT in 6 months.", "Recommend follow-up CT in 6-12 months."),
    F.CT_3_6: ("Follow-up CT in 3-6 months is not required.", "Recommend follow-up CT in 6-12 months.",
               "Consider CT in 3 months, PET/CT, or tissue sampling.", "No routine follow-up is recommended.",
               "Recommend follow-up CT in 12 months."),
}


def _fmt(mm: float) -> str:
    return f"{mm:.1f}".rstrip("0").rstrip(".")


def _rng(case_id: str, seed: int) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()[:16], 16))


class ScriptedModel(ImpressionModel):
    def __init__(self, error_rate: float = 0.0, seed: int = 0, only: tuple[str, ...] | None = None):
        self.error_rate, self.seed = error_rate, seed
        self.only = tuple(only) if only else ERROR_TYPES
        self.planted: dict[str, str | None] = {}      # case_id -> planted error type (or None)
        self.name = f"scripted(error_rate={error_rate}, seed={seed})"

    def _choose(self, r: random.Random, positive: bool) -> str | None:
        if r.random() >= self.error_rate:
            return None
        allowed = [e for e in self.only if positive or e not in POSITIVE_ONLY]
        return r.choice(allowed) if allowed else None

    def generate(self, findings: list[AIFinding], case_id: str) -> str:
        r = _rng(case_id, self.seed)
        error = self._choose(r, positive=bool(findings))
        self.planted[case_id] = error

        if len(findings) > 1:
            return self._multiple(r, findings, error)
        if not findings:
            if error == HALLUCINATION:
                side, size = r.choice(["right", "left"]), r.choice([5.0, 7.0, 11.0])
                return self._finding(r, side, size, F.category(size))
            return r.choice(NEGATIVE)
        if error == OMISSION:
            return r.choice(NEGATIVE)

        parts = []
        for f in findings:
            side, size, cat = f.laterality, f.diameter_mm, F.category(f.diameter_mm)
            if error == LATERALITY:
                side = "left" if side == "right" else "right"
            if error == SIZE:
                # includes errors just past the 0.5 mm tolerance, not only gross ones
                size = size + r.choice([-1, 1]) * r.choice([0.6, 1.0, 3.0, 5.0])
                size = size if size > 1 else f.diameter_mm + 4.0
            if error == FOLLOWUP_WRONG:
                if r.random() < 0.5:
                    parts.append(self._finding(r, side, size, None) + " " + r.choice(WRONG_FOLLOWUP_EXTRA[cat]))
                    continue
                cat = r.choice([c for c in (F.NONE, F.CT_6_12, F.CT_3_PET) if c != cat])
            parts.append(self._finding(r, side, size, None if error == FOLLOWUP_MISSING else cat))
            if error == HALLUCINATION:           # an extra nodule the detector never reported
                other = "left" if f.laterality == "right" else "right"
                extra = r.choice([4.0, 7.0, 12.0])
                parts.append(self._finding(r, other, extra, F.category(extra)))
        return " ".join(parts)

    def _multiple(self, r: random.Random, findings: list[AIFinding], error: str | None) -> str:
        """One planted error at most, in exactly one nodule (or in the study's recommendation). A changed size
        or side never lands within tolerance of another finding, so the planted report is never equivalent to
        a correct one."""
        nodules = [(f.laterality, round(f.diameter_mm, 1)) for f in findings]
        shown = [s for _, s in nodules]
        i = r.randrange(len(nodules))
        if error == OMISSION:
            del nodules[i]
        elif error == LATERALITY:
            side, size = nodules[i]
            nodules[i] = ("left" if side == "right" else "right", size)
        elif error == SIZE:
            side, size = nodules[i]
            # includes sizes just past the tolerance, as for a single nodule
            options = [size + d * k for d in (0.6, 1.0, 3.0, 5.0) for k in (-1, 1)]
            nodules[i] = (side, r.choice([x for x in options if x > 1 and all(abs(x - y) > 0.55 for y in shown)]))
        elif error == HALLUCINATION:
            extra = r.choice([x for x in (3.0, 4.0, 7.0, 10.0, 12.0, 15.0) if all(abs(x - y) > 1.0 for y in shown)])
            nodules.insert(r.randrange(len(nodules) + 1), (r.choice(["right", "left"]), extra))

        cat = F.multiple_category(shown)
        if error == FOLLOWUP_MISSING:
            followup = ""
        elif error == FOLLOWUP_WRONG:
            followup = r.choice(MULTI_WRONG_FOLLOWUP[cat])
        else:
            followup = r.choice(MULTI_FOLLOWUP_TEMPLATES[cat])

        text = r.choice(MULTI_OPENERS)
        if len(nodules) == 2 and r.random() < 0.3:
            (side, size), (side2, size2) = nodules
            text += MULTI_JOINED.format(side=side, size=_fmt(size), side2=side2, size2=_fmt(size2))
        else:
            text += " ".join(r.choice(MULTI_FINDING_TEMPLATES).format(side=side, Side=side.capitalize(),
                                                                      size=_fmt(size)) for side, size in nodules)
        return f"{text} {followup}".strip()

    def _finding(self, r: random.Random, side: str, size: float, cat: str | None) -> str:
        s = r.choice(FINDING_TEMPLATES).format(side=side, Side=side.capitalize(), size=_fmt(size))
        return s if cat is None else f"{s} {r.choice(FOLLOWUP_TEMPLATES[cat])}"
