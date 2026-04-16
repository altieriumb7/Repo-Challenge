from __future__ import annotations

import time
from contextlib import contextmanager


class StageProfiler:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        yield
        self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - start)


def memory_usage_mb() -> float | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(usage / 1024.0)
    except Exception:
        return None
