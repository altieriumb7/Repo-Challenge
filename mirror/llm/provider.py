from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import requests


class OpenRouterClient:
    def __init__(self, cache_dir: str | Path, timeout_seconds: int = 20, max_retries: int = 2):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _cache_path(self, model: str, prompt: str) -> Path:
        digest = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def complete(self, prompt: str, model: str) -> str:
        path = self._cache_path(model, prompt)
        if path.exists():
            return json.loads(path.read_text())["text"]
        if not self.api_key:
            return "LLM disabled (missing OPENROUTER_API_KEY)."

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}

        last_err = None
        for _ in range(self.max_retries + 1):
            try:
                r = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout_seconds)
                r.raise_for_status()
                body = r.json()
                text = body["choices"][0]["message"]["content"]
                usage = body.get("usage", {})
                self.usage["calls"] += 1
                self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
                path.write_text(json.dumps({"text": text}, ensure_ascii=True), encoding="utf-8")
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
        return f"LLM error: {last_err}"
