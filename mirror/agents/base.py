from __future__ import annotations

from abc import ABC, abstractmethod

from mirror.types import AgentResult, PipelineContext


class Agent(ABC):
    name: str

    @abstractmethod
    def run(self, ctx: PipelineContext) -> AgentResult:
        raise NotImplementedError
