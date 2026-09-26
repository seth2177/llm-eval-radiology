"""HTTP adapters for hosted and local LLMs. Plain httpx, no vendor SDKs.

Every adapter takes an optional httpx.Client so tests can swap in a MockTransport
and check the exact request without touching the network.
"""
from __future__ import annotations

import os
import time

import httpx

from ..cases import AIFinding
from ..prompt import SYSTEM, user_message
from .base import ImpressionModel, ModelError

RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}


def _post(client: httpx.Client, url: str, retries: int, **kw) -> dict:
    last = "no attempt"
    for attempt in range(1, retries + 1):
        try:
            r = client.post(url, **kw)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in RETRY_STATUS:
                raise ModelError(f"HTTP {r.status_code}: {r.text[:300]}")
            last = f"HTTP {r.status_code}"
        except httpx.HTTPError as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(min(2 ** attempt, 20))
    raise ModelError(f"gave up after {retries} attempts: {last}")


class AnthropicModel(ImpressionModel):
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 client: httpx.Client | None = None, retries: int = 4):
        self.model = model or os.environ.get("LLM_EVAL_ANTHROPIC_MODEL", "claude-sonnet-5")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ModelError("set ANTHROPIC_API_KEY to use --model anthropic")
        self.client = client or httpx.Client(timeout=60)
        self.retries = retries
        self.temperature = 0
        self.name = f"anthropic:{self.model}"

    def generate(self, findings: list[AIFinding], case_id: str) -> str:
        body = {"model": self.model, "max_tokens": 300, "temperature": self.temperature, "system": SYSTEM,
                "messages": [{"role": "user", "content": user_message(findings)}]}
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        out = _post(self.client, self.URL, self.retries, json=body, headers=headers)
        return "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text").strip()


class OpenAIModel(ImpressionModel):
    """Any OpenAI-compatible /chat/completions endpoint (OpenAI, gateways, vLLM).

    Temperature is sent only when set (LLM_EVAL_OPENAI_TEMPERATURE), because reasoning models
    reject anything but the default. max_completion_tokens is the current name for the limit.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None,
                 client: httpx.Client | None = None, retries: int = 4, temperature: float | None = None):
        self.model = model or os.environ.get("LLM_EVAL_OPENAI_MODEL")
        if not self.model:
            raise ModelError("pass --model-name or set LLM_EVAL_OPENAI_MODEL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ModelError("set OPENAI_API_KEY to use --model openai")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.client = client or httpx.Client(timeout=60)
        self.retries = retries
        env_t = os.environ.get("LLM_EVAL_OPENAI_TEMPERATURE")
        self.temperature = temperature if temperature is not None else (float(env_t) if env_t else None)
        self.name = f"openai:{self.model}"

    def generate(self, findings: list[AIFinding], case_id: str) -> str:
        body = {"model": self.model, "max_completion_tokens": 2000,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": user_message(findings)}]}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        out = _post(self.client, f"{self.base_url}/chat/completions", self.retries, json=body,
                    headers={"authorization": f"Bearer {self.api_key}"})
        try:
            content = out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ModelError(f"unexpected response shape: {str(out)[:300]}") from e
        if not content:
            raise ModelError(f"empty or refused response: {str(out)[:300]}")
        return content.strip()


class OllamaModel(ImpressionModel):
    """A local model through Ollama. Nothing leaves the machine."""

    def __init__(self, model: str | None = None, host: str | None = None,
                 client: httpx.Client | None = None, retries: int = 2):
        self.model = model or os.environ.get("LLM_EVAL_OLLAMA_MODEL")
        if not self.model:
            raise ModelError("pass --model-name (e.g. the name shown by `ollama list`) or set LLM_EVAL_OLLAMA_MODEL")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.client = client or httpx.Client(timeout=300)
        self.retries = retries
        self.temperature = 0
        self.name = f"ollama:{self.model}"

    def generate(self, findings: list[AIFinding], case_id: str) -> str:
        body = {"model": self.model, "stream": False, "options": {"temperature": self.temperature},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": user_message(findings)}]}
        out = _post(self.client, f"{self.host}/api/chat", self.retries, json=body)
        try:
            return out["message"]["content"].strip()
        except (KeyError, TypeError) as e:
            raise ModelError(f"unexpected response shape: {str(out)[:300]}") from e
