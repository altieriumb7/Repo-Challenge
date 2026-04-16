from __future__ import annotations

import os
from contextlib import contextmanager


class Tracer:
    def __init__(self) -> None:
        self.enabled = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

    @contextmanager
    def span(self, name: str, metadata: dict | None = None):
        _ = metadata
        yield

    def event(self, name: str, payload: dict) -> None:
        _ = (name, payload)
