"""OpenAI-compatible chat client with an on-disk response cache and usage accounting.

The cache makes reruns free and reproducible: the same (model, messages, params,
sample_index) always returns the stored response. Every call reports token usage,
which the experiments need for the cost / efficiency tables.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT, ModelConfig

DEFAULT_CACHE = Path(os.environ.get("HMAS_CACHE_DIR", ROOT / ".cache")) / "llm_cache.sqlite"


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency: float
    cached: bool = False


@dataclass
class Usage:
    """Accumulates LLM usage; one per task so per-task cost can be reported."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_calls: int = 0
    latency: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, r: LLMResponse) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += r.prompt_tokens
            self.completion_tokens += r.completion_tokens
            self.cached_calls += int(r.cached)
            self.latency += r.latency

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_calls": self.cached_calls,
            "latency": round(self.latency, 3),
        }


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self._db.commit()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
            self._db.commit()


class LLMClient:
    def __init__(self, cfg: ModelConfig, cache_path: Path | None = DEFAULT_CACHE, max_retries: int = 5):
        from openai import OpenAI  # imported lazily so offline tools don't need it

        self.cfg = cfg
        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout, max_retries=0)
        self._cache = _Cache(cache_path) if cache_path else None
        self.max_retries = max_retries
        self.total = Usage()

    def _key(self, messages, temperature, max_tokens, sample_index) -> str:
        blob = json.dumps(
            {"model": self.cfg.model, "base_url": self.cfg.base_url, "messages": messages,
             "temperature": temperature, "max_tokens": max_tokens, "sample_index": sample_index,
             "extra_body": self.cfg.extra_body},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages: list[dict], *, temperature: float | None = None,
             max_tokens: int | None = None, sample_index: int = 0, usage: Usage | None = None) -> LLMResponse:
        temperature = self.cfg.temperature if temperature is None else temperature
        max_tokens = self.cfg.max_tokens if max_tokens is None else max_tokens
        key = self._key(messages, temperature, max_tokens, sample_index)

        hit = self._cache.get(key) if self._cache else None
        if hit is not None:
            resp = LLMResponse(hit["content"], hit["prompt_tokens"], hit["completion_tokens"], 0.0, cached=True)
        else:
            resp = self._call(messages, temperature, max_tokens)
            if self._cache:
                self._cache.put(key, {"content": resp.content, "prompt_tokens": resp.prompt_tokens,
                                      "completion_tokens": resp.completion_tokens})
        self.total.add(resp)
        if usage is not None:
            usage.add(resp)
        return resp

    def _call(self, messages, temperature, max_tokens) -> LLMResponse:
        import openai

        delay = 2.0
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                r = self._client.chat.completions.create(
                    model=self.cfg.model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                    extra_body=self.cfg.extra_body or None,
                )
                u = r.usage
                return LLMResponse(
                    content=r.choices[0].message.content or "",
                    prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                    latency=time.time() - t0,
                )
            except (openai.RateLimitError, openai.APIConnectionError,
                    openai.APITimeoutError, openai.InternalServerError):
                if attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise AssertionError("unreachable")
