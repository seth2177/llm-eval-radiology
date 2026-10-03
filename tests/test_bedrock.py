"""Bedrock Converse adapter: exact request shape, both auth modes, retries. No network, no AWS account."""
import hashlib
import hmac
import json
from urllib.parse import quote

import httpx
import pytest

from llm_eval.cases import AIFinding
from llm_eval.models import BedrockModel, ModelError
from llm_eval.prompt import PROMPT_VERSION, SYSTEM
from llm_eval.run import summarize

REPLY = {"output": {"message": {"role": "assistant", "content": [{"text": " Right lung nodule, 6 mm. "}]}},
         "stopReason": "end_turn"}


def _client(seen, replies):
    """replies: one (status, json) per attempt; the last one repeats."""
    def handler(req: httpx.Request):
        seen.append(req)
        status, body = replies[min(len(seen), len(replies)) - 1]
        return httpx.Response(status, json=body)
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _no_aws_env(monkeypatch):
    for k in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION", "AWS_DEFAULT_REGION", "LLM_EVAL_BEDROCK_MODEL",
              "LLM_EVAL_BEDROCK_TEMPERATURE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("llm_eval.models.hosted.time.sleep", lambda s: None)        # retries without the wait


def test_converse_request_with_api_key():
    seen = []
    m = BedrockModel(token="br-key", client=_client(seen, [(200, REPLY)]))
    assert m.generate([AIFinding("right", 6.3, 0.92)], "1.2.826.0.1.99") == "Right lung nodule, 6 mm."
    req = seen[0]
    assert str(req.url) == ("https://bedrock-runtime.us-east-1.amazonaws.com/model/"
                            "us.anthropic.claude-sonnet-4-6/converse")
    assert req.method == "POST" and req.headers["authorization"] == "Bearer br-key"
    assert "x-amz-date" not in req.headers
    body = json.loads(req.content)
    assert body == {
        "system": [{"text": SYSTEM}],
        "messages": [{"role": "user", "content": [{"text": 'AI detector findings (JSON):\n[{"type": "pulmonary '
                      'nodule", "side": "right", "size_mm": 6.3, "detector_confidence": 0.92}]'}]}],
        "inferenceConfig": {"maxTokens": 300, "temperature": 0},
    }


def test_no_identifiers_leave_the_machine():
    seen = []
    m = BedrockModel(token="k", client=_client(seen, [(200, REPLY)]))
    m.generate([AIFinding("left", 12.4, 0.95)], "1.2.826.0.1.3680043.8.498.123")
    body = json.loads(seen[0].content)
    assert set(body) == {"system", "messages", "inferenceConfig"}
    user = body["messages"][0]["content"][0]["text"]
    findings = json.loads(user.split("\n", 1)[1])
    assert [set(f) for f in findings] == [{"type", "side", "size_mm", "detector_confidence"}]   # nothing else
    sent = seen[0].content.decode() + str(seen[0].url)
    assert "1.2.826" not in sent and "3680043" not in sent


def test_region_model_and_temperature_from_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-key")
    monkeypatch.setenv("LLM_EVAL_BEDROCK_TEMPERATURE", "default")
    seen = []
    m = BedrockModel(model="anthropic.claude-haiku-4-5-20251001-v1:0", client=_client(seen, [(200, REPLY)]))
    m.generate([], "c")
    # the model id is one path segment: ':' is percent-encoded, as boto3 does
    assert seen[0].url.raw_path == b"/model/anthropic.claude-haiku-4-5-20251001-v1%3A0/converse"
    assert seen[0].url.host == "bedrock-runtime.eu-west-1.amazonaws.com"
    assert seen[0].headers["authorization"] == "Bearer env-key"
    assert "temperature" not in json.loads(seen[0].content)["inferenceConfig"]   # models that reject it
    assert BedrockModel(token="k", region="ap-south-1").region == "ap-south-1"     # --region wins


def test_client_errors_are_not_retried():
    seen = []
    m = BedrockModel(token="k", client=_client(seen, [(400, {"message": "ValidationException"})]), retries=4)
    with pytest.raises(ModelError, match="HTTP 400"):
        m.generate([], "c")
    assert len(seen) == 1
    seen.clear()
    m = BedrockModel(token="k", client=_client(seen, [(403, {"message": "AccessDenied"})]), retries=4)
    with pytest.raises(ModelError, match="HTTP 403"):
        m.generate([], "c")
    assert len(seen) == 1


@pytest.mark.parametrize("status", [429, 500, 503])
def test_throttling_and_server_errors_are_retried(status):
    seen = []
    m = BedrockModel(token="k", client=_client(seen, [(status, {"message": "slow down"}), (200, REPLY)]))
    assert m.generate([], "c") == "Right lung nodule, 6 mm."
    assert len(seen) == 2
    seen.clear()
    m = BedrockModel(token="k", client=_client(seen, [(status, {"message": "slow down"})]), retries=3)
    with pytest.raises(ModelError, match=f"gave up after 3 attempts: HTTP {status}"):
        m.generate([], "c")
    assert len(seen) == 3


def test_empty_or_filtered_response_is_a_model_error():
    blocked = {"output": {"message": {"role": "assistant", "content": []}}, "stopReason": "guardrail_intervened"}
    m = BedrockModel(token="k", client=_client([], [(200, blocked)]))
    with pytest.raises(ModelError, match="guardrail_intervened"):
        m.generate([], "c")
    m = BedrockModel(token="k", client=_client([], [(200, {"unexpected": 1})]))
    with pytest.raises(ModelError, match="unexpected response shape"):
        m.generate([], "c")


def test_run_metadata_records_model_region_prompt_and_temperature():
    m = BedrockModel(token="k", region="us-west-2", client=_client([], [(200, REPLY)]))
    s = summarize([], [], m)
    assert s["model"] == "bedrock:us.anthropic.claude-sonnet-4-6" and s["region"] == "us-west-2"
    assert s["prompt_version"] == PROMPT_VERSION and s["temperature"] == 0


def test_without_token_or_botocore_the_message_says_what_to_do(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "botocore", None)
    monkeypatch.setitem(sys.modules, "botocore.session", None)
    with pytest.raises(ModelError, match="AWS_BEARER_TOKEN_BEDROCK"):
        BedrockModel()


# --- SigV4, checked against an independent implementation of the AWS spec ----------------------

def _hmac(key, msg):
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _expected_signature(req: httpx.Request, secret: str, region: str) -> str:
    amz_date = req.headers["x-amz-date"]
    signed = ["content-type", "host", "x-amz-date"] + (["x-amz-security-token"] if "x-amz-security-token"
                                                       in req.headers else [])
    canonical = "\n".join([
        "POST",
        quote(req.url.raw_path.decode(), safe="/~"),            # non-S3 services: the path is encoded twice
        "",
        "".join(f"{h}:{req.headers[h].strip()}\n" for h in signed),
        ";".join(signed),
        hashlib.sha256(req.content).hexdigest(),
    ])
    scope = f"{amz_date[:8]}/{region}/bedrock/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    k = _hmac(("AWS4" + secret).encode(), amz_date[:8])
    for part in (region, "bedrock", "aws4_request"):
        k = _hmac(k, part)
    return hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()


def test_sigv4_signature_matches_the_spec():
    credentials = pytest.importorskip("botocore.credentials")
    seen = []
    creds = credentials.Credentials("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", "session-tok")
    m = BedrockModel(model="anthropic.claude-haiku-4-5-20251001-v1:0", region="us-west-2", credentials=creds,
                     client=_client(seen, [(200, REPLY)]))
    m.generate([AIFinding("right", 6.3, 0.9)], "c")
    req = seen[0]
    auth = req.headers["authorization"]
    assert auth.startswith(f"AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/{req.headers['x-amz-date'][:8]}/"
                           "us-west-2/bedrock/aws4_request, ")
    assert "SignedHeaders=content-type;host;x-amz-date;x-amz-security-token" in auth
    assert req.headers["x-amz-security-token"] == "session-tok"
    assert auth.endswith("Signature=" + _expected_signature(req, creds.secret_key, "us-west-2"))


def test_sigv4_signs_every_retry():
    credentials = pytest.importorskip("botocore.credentials")
    seen = []
    m = BedrockModel(credentials=credentials.Credentials("AKID", "SECRET"),
                     client=_client(seen, [(429, {}), (200, REPLY)]))
    m.generate([], "c")
    assert len(seen) == 2 and all(r.headers["authorization"].startswith("AWS4-HMAC-SHA256") for r in seen)


def test_sigv4_uses_the_aws_credential_chain(monkeypatch, tmp_path):
    pytest.importorskip("botocore.session")
    (tmp_path / "credentials").write_text("[radiology]\naws_access_key_id = AKIDPROFILE\n"
                                          "aws_secret_access_key = secret\n")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_PROFILE", "radiology")
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    seen = []
    m = BedrockModel(client=_client(seen, [(200, REPLY)]))
    m.generate([], "c")
    assert m.auth_mode == "sigv4" and "Credential=AKIDPROFILE/" in seen[0].headers["authorization"]
