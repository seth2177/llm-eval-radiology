"""Render a run as a Markdown report a radiologist or a reviewer can read in two minutes."""
from __future__ import annotations

from .run import BOTH_FAULT, DETECTOR_FAULT, LLM_FAULT, REPORT_RIGHT, CaseResult


def _pct(rate: dict) -> str:
    if rate["value"] is None:
        return "n/a"
    lo, hi = rate["ci95"]
    return f"{rate['value'] * 100:.0f}% ({rate['k']}/{rate['n']}; 95% CI {lo * 100:.0f}-{hi * 100:.0f}%)"


def markdown(s: dict, results: list[CaseResult]) -> str:
    p, d, llm = s["population"], s["detector"], s["llm"]
    lines = [
        "# LLM report evaluation",
        "",
        f"Model: `{s['model']}`" + (f" · region `{s['region']}`" if s.get("region") else "")
        + f" · prompt `{s['prompt_version']}`",
        "",
        f"{p['studies']} studies: {p['not_routed']} not routed (no rule matched), {p['routed_to_qa']} QA phantoms, "
        f"{p['scored_chest']} chest CTs scored ({p['scored_in_this_run']} in this run).",
        *([f"Layer 1 covers all {p['scored_chest']} chest CTs; layer 2 and attribution cover the "
           f"{p['scored_in_this_run']} in this run."] if p["scored_in_this_run"] != p["scored_chest"] else []),
        "",
        "## Layer 1: detector vs ground truth",
        "",
        f"- Sensitivity: {_pct(d['sensitivity'])}",
        f"- Specificity: {_pct(d['specificity'])}",
        f"- Wrong-side detections: {d['counts']['WRONG_SIDE']}",
        f"- Size error on true positives: mean {d['size_error_mm']['mean']} mm, "
        f"mean absolute {d['size_error_mm']['mean_abs']} mm, worst {d['size_error_mm']['worst']} mm",
        f"- True positives whose measured size puts the patient in the wrong Fleischner follow-up category: "
        f"{d['tp_wrong_followup_category']}",
        "",
        "| Stratum | Sensitivity |",
        "|---|---|",
        *[f"| {k} | {_pct(v)} |" for k, v in d["sensitivity_by_stratum"].items()],
        "",
        "## Layer 2: impression vs detector findings",
        "",
        f"- Faithful reports: {_pct(llm['faithful'])}",
        *([f"- Not scored (model error, or more than six findings): {llm['not_scored']}"] if llm["not_scored"] else []),
        "",
        "| Error | Reports |",
        "|---|---|",
        *([f"| {k} | {v} |" for k, v in llm["errors"].items()] or ["| none | 0 |"]),
        "",
        "## Who caused the wrong reports",
        "",
        "| Outcome | Reports |",
        "|---|---|",
        *[f"| {k} | {s['attribution'].get(k, 0)} |" for k in (REPORT_RIGHT, LLM_FAULT, DETECTOR_FAULT, BOTH_FAULT)],
        "",
        "The detector counts as right only when it found the nodule on the correct side at a size that keeps the "
        "patient in the correct follow-up category.",
        "",
    ]
    if "checker_meta_eval" in s:
        m = s["checker_meta_eval"]
        lines += [
            "## Checker self-test (planted errors)",
            "",
            f"- Exact agreement with what was planted: {_pct(m['exact_agreement'])}",
            f"- False alarms on clean reports: {m['false_alarms_on_clean_reports']['k']} of "
            f"{m['false_alarms_on_clean_reports']['n']}",
            "",
            "| Planted error | Caught |",
            "|---|---|",
            *[f"| {k} | {_pct(v)} |" for k, v in m["recall_by_error_type"].items()],
            "",
        ]
    bad = [r for r in results if r.errors][:15]
    if bad:
        lines += ["## Examples of flagged reports", "", "| Detector said | Report said | Flagged |", "|---|---|---|"]
        for r in bad:
            ai = "; ".join(f"{f['laterality']} {f['diameter_mm']:.1f} mm" for f in r.ai) or "nothing"
            text = " ".join(r.impression.replace("|", "/").split())
            lines.append(f"| {ai} | {text} | {', '.join(r.errors)} |")
        lines.append("")
    return "\n".join(lines)
