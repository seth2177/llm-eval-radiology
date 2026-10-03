"""Scoring impressions written elsewhere, and the prompt file an external runner works from."""
import json
from pathlib import Path

import pytest

from llm_eval.__main__ import main
from llm_eval.cases import load, nodule_cases
from llm_eval.models import ModelError, ReplayModel, ScriptedModel
from llm_eval.prompt import PROMPT_VERSION, SYSTEM, user_message
from llm_eval.run import MODEL_ERROR, NOT_SCORED, evaluate

DATA = Path(__file__).resolve().parents[1] / "examples" / "router-150"
CASES = load(DATA / "results", DATA / "truth")


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_prompts_file_holds_the_exact_prompt_per_uid(tmp_path):
    out = tmp_path / "p" / "prompts.jsonl"
    assert main(["prompts", "--data", str(DATA), "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    nc = nodule_cases(CASES)
    assert [r["uid"] for r in rows] == [c.study_uid for c in nc]
    for r, c in zip(rows, nc, strict=True):
        assert r == {"uid": c.study_uid, "prompt_version": PROMPT_VERSION, "system": SYSTEM,
                     "user": user_message(c.ai)}
        assert c.study_uid not in r["system"] + r["user"]          # the uid is the join key, never prompt text
    assert main(["prompts", "--data", str(DATA), "--out", str(out), "--limit", "5"]) == 0
    assert len(out.read_text().splitlines()) == 5


def test_replay_round_trip_through_the_cli(tmp_path):
    """An "external runner" answers every prompt but one; the missing one is a MODEL_ERROR, the rest are scored."""
    prompts = tmp_path / "prompts.jsonl"
    main(["prompts", "--data", str(DATA), "--out", str(prompts)])
    writer = ScriptedModel(error_rate=0.0)
    by_uid = {c.study_uid: c for c in CASES}
    rows = [{"uid": p["uid"], "impression": writer.generate(by_uid[p["uid"]].ai, p["uid"]), "model": "batch-x",
             "prompt_version": p["prompt_version"]}
            for p in map(json.loads, prompts.read_text().splitlines())]
    missing = rows.pop(3)["uid"]
    imp = _jsonl(tmp_path / "impressions.jsonl", rows)

    out = tmp_path / "out"
    assert main(["run", "--data", str(DATA), "--model", "replay", "--impressions", str(imp), "--out", str(out)]) == 0
    s = json.loads((out / "metrics.json").read_text())
    assert s["model"] == "replay:batch-x" and s["prompt_version"] == PROMPT_VERSION and s["temperature"] is None
    assert s["llm"]["errors"] == {MODEL_ERROR: 1} and s["llm"]["not_scored"] == 1
    assert s["llm"]["faithful"]["k"] == s["llm"]["faithful"]["n"] == len(rows)
    lines = [json.loads(x) for x in (out / "cases.jsonl").read_text().splitlines()]
    failed = [r for r in lines if r["study_uid"] == missing]
    assert failed[0]["errors"] == [MODEL_ERROR] and failed[0]["attribution"] == NOT_SCORED


def test_replayed_errors_are_still_caught(tmp_path):
    c = next(c for c in nodule_cases(CASES) if c.ai)
    imp = _jsonl(tmp_path / "i.jsonl", [{"uid": c.study_uid, "impression": "No pulmonary nodule identified."}])
    res = evaluate([c], ReplayModel(imp))
    assert res[0].errors == ["OMISSION"] and res[0].impression == "No pulmonary nodule identified."


@pytest.mark.parametrize("lines,match", [
    (['{"uid": "a"}'], "expected"),
    (["not json"], "expected"),
    (['{"uid": 1, "impression": "x"}'], "must be strings"),
    (['{"uid": "a", "impression": "x"}', '{"uid": "a", "impression": "y"}'], "appears twice"),
    ([""], "no impressions"),
])
def test_bad_impression_files_fail_loudly(tmp_path, lines, match):
    f = tmp_path / "i.jsonl"
    f.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ModelError, match=match):
        ReplayModel(f)


def test_empty_impression_is_a_model_error(tmp_path):
    m = ReplayModel(_jsonl(tmp_path / "i.jsonl", [{"uid": "a", "impression": "  "}]))
    assert m.name == "replay:unknown" and m.prompt_version == "unknown"
    with pytest.raises(ModelError, match="empty impression"):
        m.generate([], "a")


def test_replay_needs_a_file(tmp_path, capsys):
    assert main(["run", "--data", str(DATA), "--model", "replay", "--out", str(tmp_path)]) == 2
    assert "--impressions" in capsys.readouterr().err
    assert main(["run", "--data", str(DATA), "--model", "replay", "--impressions", str(tmp_path / "nope.jsonl"),
                 "--out", str(tmp_path)]) == 2
