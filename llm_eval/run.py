"""Run both layers over a set of cases and put the blame where it belongs.

For every chest study the nodule model saw:
  1. detector outcome against ground truth (TP / FN / FP / TN / WRONG_SIDE)
  2. the LLM writes an impression from the detector's findings only
  3. the checker scores that impression against the detector's findings

Crossing 1 and 3 answers the question that matters in production: when the final
report is wrong, was it the detector or the language model?
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import detector as D
from .cases import Case, nodule_cases
from .checker import OUT_OF_SCOPE, check
from .detector import Rate
from .models.base import ImpressionModel, ModelError
from .models.scripted import ScriptedModel
from .prompt import PROMPT_VERSION

REPORT_RIGHT, LLM_FAULT, DETECTOR_FAULT, BOTH_FAULT = (
    "report right", "LLM error (detector was right)", "detector error (LLM was faithful)", "both wrong")
NOT_SCORED = "not scored (model error or out of scope)"
MODEL_ERROR = "MODEL_ERROR"


@dataclass
class CaseResult:
    study_uid: str
    truth: dict | None
    ai: list[dict]
    detector: str
    impression: str
    errors: list[str]
    planted: str | None
    attribution: str
    ms: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def detector_right(case: Case) -> bool:
    """Right side AND a size that keeps the patient in the correct follow-up category."""
    o = D.outcome(case)
    return o == D.TN or (o == D.TP and D.management_ok(case))


def _attribution(case: Case, errors: list[str]) -> str:
    if MODEL_ERROR in errors or OUT_OF_SCOPE in errors:
        return NOT_SCORED
    faithful = not errors
    if detector_right(case):
        return REPORT_RIGHT if faithful else LLM_FAULT
    return DETECTOR_FAULT if faithful else BOTH_FAULT


def evaluate(cases: list[Case], model: ImpressionModel, limit: int | None = None, sink=None) -> list[CaseResult]:
    """Score each case. A model failure on one case is recorded, not fatal; `sink` gets each
    result as it lands, so a long paid run keeps everything done before an interruption."""
    out = []
    for c in nodule_cases(cases)[:limit]:
        t0 = time.perf_counter()
        try:
            text = model.generate(c.ai, c.study_uid)
            errors = check(text, c.ai).errors
        except ModelError as e:
            text, errors = f"[model error] {e}", [MODEL_ERROR]
        ms = (time.perf_counter() - t0) * 1000
        r = CaseResult(
            study_uid=c.study_uid,
            truth=c.truth.__dict__ if c.truth else None,
            ai=[f.__dict__ for f in c.ai],
            detector=D.outcome(c),
            impression=text,
            errors=errors,
            planted=model.planted.get(c.study_uid) if isinstance(model, ScriptedModel) else None,
            attribution=_attribution(c, errors),
            ms=round(ms, 1),
        )
        out.append(r)
        if sink is not None:
            sink.write(json.dumps(r.as_dict()) + "\n")
            sink.flush()
    return out


def meta_eval(results: list[CaseResult]) -> dict:
    """Scripted model only: did the checker find exactly what was planted?"""
    per_type: dict[str, list[int]] = {}
    false_alarms = 0
    exact = 0
    for r in results:
        want = {r.planted} if r.planted else set()
        got = set(r.errors)
        exact += want == got
        if r.planted:
            hit = per_type.setdefault(r.planted, [0, 0])
            hit[1] += 1
            hit[0] += r.planted in got
        elif got:
            false_alarms += 1
    clean = sum(1 for r in results if not r.planted)
    return {
        "exact_agreement": Rate(exact, len(results)).as_dict(),
        "recall_by_error_type": {k: Rate(*v).as_dict() for k, v in sorted(per_type.items())},
        "false_alarms_on_clean_reports": {"k": false_alarms, "n": clean},
    }


def summarize(cases: list[Case], results: list[CaseResult], model: ImpressionModel) -> dict:
    nc = nodule_cases(cases)
    scored = [r for r in results if r.attribution != NOT_SCORED]
    faithful = sum(not r.errors for r in scored)
    error_counts = Counter(e for r in results for e in r.errors)
    summary = {
        "model": model.name,
        "prompt_version": getattr(model, "prompt_version", PROMPT_VERSION),
        "temperature": getattr(model, "temperature", None),
        **({"region": model.region} if getattr(model, "region", None) else {}),
        "population": {
            "studies": len(cases),
            "not_routed": sum(not c.routed for c in cases),
            "routed_to_qa": sum(c.model == "ct-qa" for c in cases),
            "scored_chest": len(nc),
            "scored_in_this_run": len(results),
            "note": "Layer 1 covers every scored chest study; layer 2 and attribution cover this run's cases only.",
        },
        "detector": D.metrics(nc),
        "llm": {
            "faithful": Rate(faithful, len(scored)).as_dict(),
            "errors": dict(sorted(error_counts.items())),
            "reports_with_errors": sum(bool(r.errors) for r in scored),
            "not_scored": len(results) - len(scored),
            "median_ms": sorted(r.ms for r in results)[len(results) // 2] if results else None,
        },
        "attribution": dict(Counter(r.attribution for r in results)),
    }
    if isinstance(model, ScriptedModel):
        summary["checker_meta_eval"] = meta_eval(results)
    return summary


def write(out_dir: Path, summary: dict, results: list[CaseResult]) -> None:
    """metrics.json and report.md; cases.jsonl is written case by case during evaluate()."""
    from .report import markdown
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(markdown(summary, results), encoding="utf-8")
