from __future__ import annotations

from abc import ABC, abstractmethod

from ..cases import AIFinding


class ModelError(RuntimeError):
    pass


class ImpressionModel(ABC):
    """Anything that turns detector findings into an impression paragraph."""

    name: str = "base"

    @abstractmethod
    def generate(self, findings: list[AIFinding], case_id: str) -> str:
        """case_id lets deterministic models (the scripted one) be reproducible per study."""
