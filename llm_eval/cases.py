"""Load what the router's AI said next to what was really there.

dicom-ai-router writes one pair of files per study:

  results/<StudyInstanceUID>.json   the model's output (findings, sizes, confidence)
  truth/<StudyInstanceUID>.json     the synthetic scanner's ground truth

A truth file with no results file is a study the router did not send to a model
(for example a head CT that matched no routing rule). Those are counted, not scored.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Nodule:
    laterality: str          # "right" | "left"
    diameter_mm: float
    density: str = "solid"   # "solid" | "ground-glass"


@dataclass(frozen=True)
class AIFinding:
    laterality: str
    diameter_mm: float
    confidence: float


@dataclass
class Case:
    study_uid: str
    kind: str                                  # "chest" | "qa" | "head" ...
    truth: Nodule | None                       # None = no nodule
    model: str | None = None                   # None = not routed
    ai: list[AIFinding] = field(default_factory=list)

    @property
    def routed(self) -> bool:
        return self.model is not None


def _nodule(t: dict) -> Nodule | None:
    n = t.get("nodule")
    if not n:
        return None
    return Nodule(n["laterality"], float(n["diameter_mm"]), n.get("density", "solid"))


def _findings(ai: dict) -> list[AIFinding]:
    out = []
    for f in ai.get("findings", []):
        if f.get("type") == "pulmonary_nodule" and f.get("present"):
            out.append(AIFinding(f["laterality"], float(f["diameter_mm"]), float(f.get("confidence", 0.0))))
    return out


def load(results_dir: Path, truth_dir: Path) -> list[Case]:
    """Every truth file becomes a Case; the matching results file, if any, fills in the AI side."""
    results_dir, truth_dir = Path(results_dir), Path(truth_dir)
    truth_files = sorted(truth_dir.glob("*.json"))
    if not truth_files:
        raise FileNotFoundError(f"no truth/*.json under {truth_dir}")
    cases = []
    for tf in truth_files:
        t = json.loads(tf.read_text(encoding="utf-8"))
        case = Case(study_uid=t["study_uid"], kind=t.get("kind", "chest"), truth=_nodule(t))
        rf = results_dir / tf.name
        if rf.exists():
            ai = json.loads(rf.read_text(encoding="utf-8"))["ai"]
            case.model = ai.get("model")
            case.ai = _findings(ai)
        cases.append(case)
    return cases


def nodule_cases(cases: list[Case]) -> list[Case]:
    """Chest studies the lung-nodule model actually saw: the population the report layer is scored on."""
    return [c for c in cases if c.kind == "chest" and c.model == "lung-nodule"]
