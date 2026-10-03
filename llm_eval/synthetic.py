"""Multi-nodule detector outputs for the checker self-test.

dicom-ai-router's synthetic studies carry one nodule at most, and its detector has not reported two on one
study, so the router data never exercises the multiple-nodule rules. These variants do: for every chest
study where the detector reported one nodule, a copy with one or two more findings added (or, for some, two
small findings in place of it, so the "all under 6 mm" row is tested too). They exist only to
test the checker (`selftest`); `run` never scores them, and they say nothing about any detector.

Findings in a variant are at least 1.5 mm apart, so a planted size or side error can't turn into a
description of a different finding.
"""
from __future__ import annotations

import hashlib
import random

from .cases import AIFinding, Case, nodule_cases

MIN_GAP_MM = 1.5


def multi_nodule_variants(cases: list[Case]) -> list[Case]:
    out = []
    for c in nodule_cases(cases):
        if len(c.ai) != 1:
            continue
        r = random.Random(int(hashlib.sha256(f"multi:{c.study_uid}".encode()).hexdigest()[:16], 16))
        if r.random() < 0.4:
            # every nodule small: the "no routine follow-up" row, including the 5.4 / 5.5 mm rounding boundary
            f = c.ai[0]
            ai = [AIFinding(f.laterality, r.choice([round(r.uniform(2.5, 3.9), 1), 5.4, 5.5]), f.confidence)]
            largest, want = 5.4, 2
        else:
            ai = list(c.ai)
            largest, want = 16.0, (2 if r.random() < 0.65 else 3)
        while len(ai) < want:
            size = round(r.uniform(2.5, largest), 1)
            if all(abs(size - f.diameter_mm) >= MIN_GAP_MM for f in ai):
                ai.append(AIFinding(r.choice(["right", "left"]), size, round(r.uniform(0.55, 0.95), 2)))
        out.append(Case(f"{c.study_uid}.multi", c.kind, c.truth, c.model, ai))
    return out
