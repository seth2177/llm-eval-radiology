"""Amazon Bedrock through the Converse API. Plain httpx, no boto3.

Auth, first match wins:
  1. a Bedrock API key in AWS_BEARER_TOKEN_BEDROCK, sent as Authorization: Bearer
  2. SigV4 with credentials from the normal AWS chain (env keys, AWS_PROFILE, SSO, instance role).
     Signing needs botocore: pip install "llm-eval-radiology[bedrock]".

Region from --region, then AWS_REGION / AWS_DEFAULT_REGION, else us-east-1.

The default model is the US cross-region inference profile for Claude Sonnet 4.6. Bedrock serves the
newer Claude models through cross-region inference profiles, not the bare model ID, and Anthropic's
Bedrock docs list Sonnet 4.6 among the Converse-capable models. Newer Claude models reject a temperature;
for those set LLM_EVAL_BEDROCK_TEMPERATURE=default and the field is left out.
"""
from __future__ import annotations

import os
from urllib.parse import quote

import httpx

from ..cases import AIFinding
from ..prompt import SYSTEM, user_message
from .base import ImpressionModel, ModelError
from .hosted import _post

DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
DEFAULT_REGION = "us-east-1"


class _SigV4(httpx.Auth):
    """Signs each attempt (retries included), so the X-Amz-Date stays fresh."""

    requires_request_body = True

    def __init__(self, credentials, region: str):
        self.credentials, self.region = credentials, region

    def auth_flow(self, request: httpx.Request):
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        # Sign only what we control; the transport may add or reorder the rest.
        aws = AWSRequest(method=request.method, url=str(request.url), data=request.content,
                         headers={"content-type": request.headers["content-type"]})
        SigV4Auth(self.credentials.get_frozen_credentials(), "bedrock", self.region).add_auth(aws)
        for k in ("Authorization", "X-Amz-Date", "X-Amz-Security-Token"):
            if k in aws.headers:
                request.headers[k] = aws.headers[k]
        yield request


def _temperature(value: float | None | str) -> float | None:
    if value != "env":
        return value
    env = os.environ.get("LLM_EVAL_BEDROCK_TEMPERATURE")
    if env is None:
        return 0
    return None if env.strip().lower() in ("", "default", "none") else float(env)


class BedrockModel(ImpressionModel):
    def __init__(self, model: str | None = None, region: str | None = None, token: str | None = None,
                 credentials=None, client: httpx.Client | None = None, retries: int = 4,
                 temperature: float | None | str = "env"):
        self.model = model or os.environ.get("LLM_EVAL_BEDROCK_MODEL", DEFAULT_MODEL)
        self.region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or DEFAULT_REGION
        self.url = f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{quote(self.model, safe='')}/converse"
        token = token or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if token:
            self.auth_mode, self.auth, self.headers = "api-key", None, {"authorization": f"Bearer {token}"}
        else:
            self.auth_mode, self.headers = "sigv4", {}
            self.auth = _SigV4(credentials or _aws_credentials(), self.region)
        self.client = client or httpx.Client(timeout=60)
        self.retries = retries
        self.temperature = _temperature(temperature)
        self.name = f"bedrock:{self.model}"

    def generate(self, findings: list[AIFinding], case_id: str) -> str:
        inference = {"maxTokens": 300}
        if self.temperature is not None:
            inference["temperature"] = self.temperature
        body = {"system": [{"text": SYSTEM}],
                "messages": [{"role": "user", "content": [{"text": user_message(findings)}]}],
                "inferenceConfig": inference}
        out = _post(self.client, self.url, self.retries, json=body, headers=self.headers, auth=self.auth)
        try:
            blocks = out["output"]["message"]["content"]
        except (KeyError, TypeError) as e:
            raise ModelError(f"unexpected response shape: {str(out)[:300]}") from e
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict)).strip()
        if not text:
            raise ModelError(f"empty or refused response (stopReason={out.get('stopReason')})")
        return text


def _aws_credentials():
    try:
        import botocore.session
    except ImportError as e:
        raise ModelError('set AWS_BEARER_TOKEN_BEDROCK, or pip install "llm-eval-radiology[bedrock]" '
                         "(botocore) to sign with your AWS credentials") from e
    creds = botocore.session.Session().get_credentials()       # honors AWS_PROFILE, SSO, instance roles
    if creds is None:
        raise ModelError("no AWS credentials found: set AWS_BEARER_TOKEN_BEDROCK, AWS_PROFILE or AWS_ACCESS_KEY_ID")
    return creds
