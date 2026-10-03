import json
from pathlib import Path

import httpx
import pytest

from llm_eval import detector as D
from llm_eval import fleischner as F
from llm_eval.__main__ import main
from llm_eval.cases import AIFinding, Case, Nodule, load, nodule_cases
from llm_eval.models import AnthropicModel, ModelError, OllamaModel, OpenAIModel, ScriptedModel
from llm_eval.prompt import user_message

DATA = Path(__file__).resolve().parents[1] / "examples" / "router-150"


# --- Fleischner 2017 boundaries ---------------------------------------------------------------

@pytest.mark.parametrize("mm,expected", [
    (5.4, F.NONE), (5.5, F.CT_6_12), (6.0, F.CT_6_12), (8.0, F.CT_6_12),
    (8.4, F.CT_6_12), (8.5, F.CT_3_PET), (12.0, F.CT_3_PET),
])
def test_solid_boundaries_round_to_nearest_mm(mm, expected):
    assert F.category(mm) == expected


@pytest.mark.parametrize("sizes,expected", [
    ([5.4, 3.0], F.NONE), ([5.5, 3.0], F.CT_3_6), ([7.0, 4.2], F.CT_3_6), ([12.4, 6.3], F.CT_3_6),
    ([3.0, 4.0, 5.0], F.NONE),
])
def test_multiple_nodules_are_managed_by_the_largest(sizes, expected):
    assert F.multiple_category(sizes) == expected
    assert F.management(sizes) == expected


def test_management_picks_the_table():
    assert F.management([]) is None
    assert F.management([6.3]) == F.CT_6_12 and F.management([12.4]) == F.CT_3_PET


def test_ground_glass():
    assert F.category(5.0, "ground-glass") == F.NONE
    assert F.category(14.0, "ground-glass") == F.CT_6_12


# --- detector layer --------------------------------------------------------------------------

def _case(truth, ai):
    return Case("1.2.3", "chest", truth, "lung-nodule", ai)


def test_outcomes():
    n = Nodule("right", 8.0)
    assert D.outcome(_case(n, [AIFinding("right", 7.5, 0.9)])) == D.TP
    assert D.outcome(_case(n, [AIFinding("left", 7.5, 0.9)])) == D.WRONG_SIDE
    assert D.outcome(_case(n, [])) == D.FN
    assert D.outcome(_case(None, [AIFinding("left", 4, 0.5)])) == D.FP
    assert D.outcome(_case(None, [])) == D.TN


def test_wilson_interval_is_sane():
    lo, hi = D.wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.35          # zero hits still leaves real uncertainty
    lo, hi = D.wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.2
    assert D.wilson(0, 0) == (0.0, 1.0)


# --- loading router output -------------------------------------------------------------------

def test_load_router_output():
    cases = load(DATA / "results", DATA / "truth")
    assert len(cases) == 150
    assert sum(not c.routed for c in cases) > 0            # head CTs: truth but no result
    assert all(c.kind == "chest" for c in nodule_cases(cases))


def test_load_fails_loudly_on_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "results", tmp_path / "truth")


def test_prompt_carries_no_identifiers():
    msg = user_message([AIFinding("right", 6.3, 0.92)])
    assert "right" in msg and "6.3" in msg
    for leak in ("uid", "2.25.", "1.2.826", "patient", "accession"):
        assert leak not in msg.lower()


# --- scripted model is deterministic ---------------------------------------------------------

def test_scripted_model_reproducible():
    f = [AIFinding("left", 12.4, 0.9)]
    a, b = ScriptedModel(0.5, seed=3), ScriptedModel(0.5, seed=3)
    assert [a.generate(f, str(i)) for i in range(50)] == [b.generate(f, str(i)) for i in range(50)]
    assert a.planted == b.planted and any(a.planted.values()) and not all(a.planted.values())


# --- adapters: exact request shape, no network ------------------------------------------------

def _mock(capture, reply, status=200):
    def handler(req: httpx.Request):
        capture.append(req)
        return httpx.Response(status, json=reply)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_anthropic_request():
    seen = []
    reply = {"content": [{"type": "text", "text": " ok "}]}
    m = AnthropicModel(model="claude-x", api_key="k", client=_mock(seen, reply))
    assert m.generate([AIFinding("right", 6.3, 0.9)], "c") == "ok"
    body = json.loads(seen[0].content)
    assert seen[0].url == "https://api.anthropic.com/v1/messages"
    assert seen[0].headers["x-api-key"] == "k" and seen[0].headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-x" and body["system"] and body["messages"][0]["role"] == "user"


def test_openai_request():
    seen = []
    m = OpenAIModel(model="gpt-x", api_key="k", base_url="https://gw.example/v1",
                    client=_mock(seen, {"choices": [{"message": {"content": "ok"}}]}))
    assert m.generate([], "c") == "ok"
    assert str(seen[0].url) == "https://gw.example/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer k"


def test_ollama_request():
    seen = []
    m = OllamaModel(model="llama", host="http://box:11434", client=_mock(seen, {"message": {"content": "ok"}}))
    assert m.generate([], "c") == "ok"
    assert json.loads(seen[0].content)["stream"] is False


def test_client_errors_are_not_retried():
    seen = []
    m = AnthropicModel(model="x", api_key="k", client=_mock(seen, {"error": "bad"}, status=400), retries=4)
    with pytest.raises(ModelError, match="HTTP 400"):
        m.generate([], "c")
    assert len(seen) == 1


def test_missing_keys_fail_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ModelError, match="ANTHROPIC_API_KEY"):
        AnthropicModel()


# --- end to end ------------------------------------------------------------------------------

def test_cli_run_and_selftest(tmp_path, capsys):
    assert main(["run", "--data", str(DATA), "--model", "scripted", "--error-rate", "0.4", "--out", str(tmp_path)]) == 0
    s = json.loads((tmp_path / "metrics.json").read_text())
    meta = s["checker_meta_eval"]
    assert meta["exact_agreement"]["k"] == meta["exact_agreement"]["n"]
    assert meta["false_alarms_on_clean_reports"]["k"] == 0
    assert sum(s["attribution"].values()) == s["population"]["scored_in_this_run"]
    assert "Who caused the wrong reports" in (tmp_path / "report.md").read_text()
    assert main(["selftest", "--data", str(DATA)]) == 0
    out = capsys.readouterr().out
    assert "SELFTEST PASS" in out and "multi-nodule 57/57" in out


def test_bundled_example_is_the_default_data(tmp_path, monkeypatch, capsys):
    from llm_eval.__main__ import _data_dir
    monkeypatch.chdir(tmp_path)                                  # any folder, as after pip install
    d = _data_dir(Path("example"))
    assert (d / "truth").is_dir() and len(list((d / "truth").glob("*.json"))) == 150
    assert _data_dir(tmp_path) == tmp_path                       # a real folder is used as given
    assert main(["prompts", "--out", str(tmp_path / "p.jsonl")]) == 0
    assert "108 prompts" in capsys.readouterr().out


def test_openai_body_suits_reasoning_models():
    seen = []
    m = OpenAIModel(model="gpt-x", api_key="k", client=_mock(seen, {"choices": [{"message": {"content": "ok"}}]}))
    m.generate([], "c")
    body = json.loads(seen[0].content)
    assert "max_completion_tokens" in body and "max_tokens" not in body and "temperature" not in body


def test_openai_refusal_is_a_model_error():
    m = OpenAIModel(model="gpt-x", api_key="k", client=_mock([], {"choices": [{"message": {"content": None}}]}))
    with pytest.raises(ModelError, match="empty or refused"):
        m.generate([], "c")


# --- attribution ------------------------------------------------------------------------------

def test_right_side_wrong_management_is_a_detector_error():
    """5 mm nodule measured as 6.3 mm: found, right side, but moves the patient into CT follow-up."""
    from llm_eval.run import DETECTOR_FAULT, _attribution, detector_right
    c = _case(Nodule("right", 5.0), [AIFinding("right", 6.3, 0.9)])
    assert D.outcome(c) == D.TP and not D.management_ok(c) and not detector_right(c)
    assert _attribution(c, []) == DETECTOR_FAULT


def test_an_extra_finding_is_a_detector_error():
    """Truth has one nodule; the detector found it and invented a second. The report will carry the invented one,
    and the multiple-nodule table changes the follow-up."""
    from llm_eval.run import DETECTOR_FAULT, REPORT_RIGHT, _attribution, detector_right
    one = _case(Nodule("right", 7.0), [AIFinding("right", 7.0, 0.9)])
    two = _case(Nodule("right", 7.0), [AIFinding("right", 7.0, 0.9), AIFinding("left", 4.0, 0.6)])
    assert detector_right(one) and _attribution(one, []) == REPORT_RIGHT
    assert D.outcome(two) == D.TP and not D.management_ok(two) and not detector_right(two)
    assert _attribution(two, []) == DETECTOR_FAULT
    small = _case(Nodule("right", 4.0), [AIFinding("right", 4.0, 0.9), AIFinding("left", 3.0, 0.6)])
    assert D.management_ok(small) and not detector_right(small)      # same follow-up, still an invented nodule


def test_scripted_multi_nodule_reports_are_reproducible_and_planted():
    from llm_eval.synthetic import multi_nodule_variants
    multi = multi_nodule_variants(load(DATA / "results", DATA / "truth"))
    assert multi and all(len(c.ai) > 1 and c.study_uid.endswith(".multi") for c in multi)
    assert {F.multiple_category([f.diameter_mm for f in c.ai]) for c in multi} == {F.NONE, F.CT_3_6}
    a, b = ScriptedModel(1.0, seed=5), ScriptedModel(1.0, seed=5)
    assert [a.generate(c.ai, c.study_uid) for c in multi] == [b.generate(c.ai, c.study_uid) for c in multi]
    assert set(a.planted.values()) == {"HALLUCINATION", "OMISSION", "LATERALITY", "SIZE", "FOLLOWUP_WRONG",
                                       "FOLLOWUP_MISSING"}


class _Flaky(ScriptedModel):
    def generate(self, findings, case_id):
        if case_id.endswith("3"):
            raise ModelError("HTTP 529 overloaded")
        return super().generate(findings, case_id)


def test_one_model_failure_does_not_lose_the_run(tmp_path):
    from llm_eval.run import MODEL_ERROR, NOT_SCORED, evaluate, summarize
    cases = load(DATA / "results", DATA / "truth")
    with (tmp_path / "cases.jsonl").open("w") as sink:
        res = evaluate(cases, _Flaky(), sink=sink)
    failed = [r for r in res if MODEL_ERROR in r.errors]
    assert failed and all(r.attribution == NOT_SCORED for r in failed)
    assert len((tmp_path / "cases.jsonl").read_text().splitlines()) == len(res)
    s = summarize(cases, res, _Flaky())
    assert s["llm"]["faithful"]["n"] == len(res) - len(failed)
