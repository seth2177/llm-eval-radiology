"""Impressions written somewhere else (a batch job, another tool), scored as if a model had just written them.

One JSON object per line:

  {"uid": "<StudyInstanceUID>", "impression": "...", "model": "<optional>", "prompt_version": "<optional>"}

`python -m llm_eval prompts` writes the exact prompt for each uid, so an external runner can produce this file.
A study with no line is recorded as MODEL_ERROR, so a partial batch is still scored for what it has.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..cases import AIFinding
from .base import ImpressionModel, ModelError


class ReplayModel(ImpressionModel):
    def __init__(self, path: Path | str):
        self.path = Path(path)
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            raise ModelError(f"cannot read {self.path}: {e}") from e
        self.impressions: dict[str, str] = {}
        models, versions = set(), set()
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                uid, text = row["uid"], row["impression"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise ModelError(f'{self.path}:{n}: expected {{"uid": ..., "impression": ...}}') from e
            if not isinstance(uid, str) or not isinstance(text, str):
                raise ModelError(f"{self.path}:{n}: uid and impression must be strings")
            if uid in self.impressions:
                raise ModelError(f"{self.path}:{n}: uid {uid} appears twice")
            self.impressions[uid] = text
            models.add(str(row.get("model") or "unknown"))
            versions.add(str(row.get("prompt_version") or "unknown"))
        if not self.impressions:
            raise ModelError(f"no impressions in {self.path}")
        self.name = f"replay:{'+'.join(sorted(models))}"
        self.prompt_version = "+".join(sorted(versions))     # whatever the external run says it used
        self.temperature = None                              # not known here

    def generate(self, findings: list[AIFinding], case_id: str) -> str:
        text = self.impressions.get(case_id)
        if text is None:
            raise ModelError(f"no impression for {case_id} in {self.path.name}")
        if not text.strip():
            raise ModelError(f"empty impression for {case_id} in {self.path.name}")
        return text
