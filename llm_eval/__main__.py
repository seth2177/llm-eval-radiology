"""Command line.

  python -m llm_eval run --data examples/router-150 --model scripted --error-rate 0.3 --out out/scripted
  python -m llm_eval run --data examples/router-150 --model anthropic --out out/claude
  python -m llm_eval run --data examples/router-150 --model ollama --model-name llama3.1:8b --out out/local
  python -m llm_eval selftest --data examples/router-150

--data is a folder holding results/ and truth/ (a dicom-ai-router workdir works as is).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cases import load
from .models import AnthropicModel, ModelError, OllamaModel, OpenAIModel, ScriptedModel
from .run import evaluate, summarize, write


def _model(a):
    if a.model == "scripted":
        return ScriptedModel(error_rate=a.error_rate, seed=a.seed)
    if a.model == "anthropic":
        return AnthropicModel(model=a.model_name)
    if a.model == "openai":
        return OpenAIModel(model=a.model_name)
    return OllamaModel(model=a.model_name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="llm_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "selftest"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True, help="folder with results/ and truth/")
    r = sub.choices["run"]
    r.add_argument("--model", choices=["scripted", "anthropic", "openai", "ollama"], default="scripted")
    r.add_argument("--model-name", help="provider model id (defaults: anthropic claude-sonnet-5; others required)")
    r.add_argument("--error-rate", type=float, default=0.3,
                   help="scripted model: share of reports with a planted error")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--limit", type=int, help="score only the first N chest studies (to cap API spend)")
    r.add_argument("--out", type=Path, default=Path("out"))
    a = ap.parse_args(argv)

    try:
        cases = load(a.data / "results", a.data / "truth")
    except FileNotFoundError as e:
        print(f"llm_eval: {e}", file=sys.stderr)
        return 2

    if a.cmd == "selftest":
        return _selftest(cases)

    try:
        model = _model(a)
    except ModelError as e:
        print(f"llm_eval: {e}", file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "cases.jsonl").open("w", encoding="utf-8") as sink:
        results = evaluate(cases, model, limit=a.limit, sink=sink)
    summary = summarize(cases, results, model)
    write(a.out, summary, results)
    llm = summary["llm"]["faithful"]
    print(f"{summary['model']}: {llm['k']}/{llm['n']} reports faithful to the detector; "
          f"report written to {a.out / 'report.md'}")
    return 0


def _selftest(cases) -> int:
    """Plant an error in every report, then in none. The checker must catch all and flag none."""
    ok = True
    for rate, label in ((1.0, "every report has one planted error"), (0.0, "no planted errors")):
        for seed in range(3):
            m = ScriptedModel(error_rate=rate, seed=seed)
            res = evaluate(cases, m)
            misses = [r for r in res if set(r.errors) != ({r.planted} if r.planted else set())]
            print(f"{label}, seed {seed}: {len(res) - len(misses)}/{len(res)} exact")
            for r in misses[:5]:
                print(f"   planted={r.planted} found={r.errors} :: {r.impression}")
            ok &= not misses
    print("SELFTEST PASS" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
