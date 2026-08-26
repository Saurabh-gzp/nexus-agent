"""LLMClient — role-based model access with 3-layer resilience.

Layer 1: key rotation      (provider ke andar KeyRing)
Layer 2: model fallback    (role ki fallback chain: medium -> small -> 8b)
Layer 3: provider fallback (mistral -> openai/groq, agar enabled ho)

Plus: per-model rate limiting and usage accounting.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from ..providers.base import ChatResult, ProviderError
from ..providers.registry import ProviderRegistry


class RateLimiter:
    """Per-model pacing shared across threads.

    Reserves the next slot under the lock (instead of sleeping while holding it),
    so N parallel agents queue up correctly rather than all firing at once.
    Adds a safety margin because provider RPS is enforced server-side.
    """

    def __init__(self, margin: float = 1.15) -> None:
        self._next_free: Dict[str, float] = defaultdict(float)
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._global = threading.Lock()
        self.margin = margin

    def _lock_for(self, model: str) -> threading.Lock:
        with self._global:
            return self._locks[model]

    def wait(self, model: str, rps: float) -> float:
        if rps <= 0:
            return 0.0
        gap = (1.0 / rps) * self.margin
        lock = self._lock_for(model)
        with lock:
            now = time.time()
            slot = max(now, self._next_free[model])
            self._next_free[model] = slot + gap
        sleep_for = slot - time.time()
        if sleep_for > 0:
            time.sleep(min(sleep_for, 60.0))
            return sleep_for
        return 0.0

    def penalise(self, model: str, seconds: float) -> None:
        """Push the next slot out after a 429 from this model."""
        lock = self._lock_for(model)
        with lock:
            self._next_free[model] = max(self._next_free[model], time.time() + seconds)


@dataclass
class UsageStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    fallbacks: int = 0
    latency: float = 0.0
    by_model: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_role: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def snapshot(self) -> dict:
        return {
            "calls": self.calls, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens,
            "errors": self.errors, "fallbacks": self.fallbacks,
            "avg_latency": round(self.latency / self.calls, 2) if self.calls else 0,
            "by_model": dict(self.by_model), "by_role": dict(self.by_role),
        }


class LLMClient:
    def __init__(self, config, registry: Optional[ProviderRegistry] = None,
                 notifier: Optional[Callable[[str, str], None]] = None):
        self.tick = None   # ui.tick — live processing indicator (set by AgentContext)
        self.config = config
        self.notify = notifier or (lambda level, msg: None)
        self.registry = registry or ProviderRegistry(config, self.notify)
        self.limiter = RateLimiter()
        self.stats = UsageStats()
        self._large_calls: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def chat(self, role: str, messages: List[dict], tools: Optional[List[dict]] = None,
             model: Optional[str] = None, task_id: str = "global",
             **overrides: Any) -> ChatResult:
        from contextlib import nullcontext
        tk = self.tick(f"thinking · {role}") if self.tick else nullcontext()
        with tk:
            return self._chat_impl(role, messages, tools, model, task_id, **overrides)

    def _chat_impl(
        self,
        role: str,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        model: Optional[str] = None,
        task_id: str = "global",
        **overrides: Any,
    ) -> ChatResult:
        chain = [model] if model else self.config.model_chain(role)
        params = {**self.config.gen_params(role), **overrides}
        last_err: Optional[Exception] = None

        for provider_name in self.registry.order():
            try:
                provider = self.registry.get(provider_name)
            except Exception:
                continue
            for idx, m in enumerate(chain):
                if not m:
                    continue
                # Never skip the role's PRIMARY model — budget only applies to
                # later chain entries (true large / last-resort models).
                # Skipping medium-2508 here was why SYNTHESIZE jumped to
                # medium-latest after one call despite a full key pool.
                if idx > 0 and self._is_large(m) and not self._allow_large(task_id):
                    self.notify("warn", f"Large-model budget exhausted, skipping {m}")
                    continue
                self.limiter.wait(m, self.config.rate_limit(m))
                try:
                    res = provider.chat(m, messages, tools=tools, **params)
                    with self._lock:
                        self.stats.calls += 1
                        self.stats.prompt_tokens += res.prompt_tokens
                        self.stats.completion_tokens += res.completion_tokens
                        self.stats.latency += res.latency
                        self.stats.by_model[res.model] += 1
                        self.stats.by_role[role] += 1
                        if idx > 0:
                            self.stats.fallbacks += 1
                    if idx > 0:
                        self.notify("info", f"[{role}] fallback model in use: {m}")
                    return res
                except ProviderError as e:
                    last_err = e
                    self.stats.errors += 1
                    self.notify("warn", f"[{role}] {provider_name}/{m} failed: {str(e)[:140]}")
                    if e.status == 429:
                        self.limiter.penalise(m, 5.0)   # slow this model down globally
                    if not e.retryable and e.status in (400, 422):
                        continue
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    self.stats.errors += 1
                    self.notify("warn", f"[{role}] {provider_name}/{m} error: {str(e)[:140]}")
            if len(self.registry.providers) > 1:
                self.notify("warn", f"Provider '{provider_name}' exhausted -> trying next provider")
        raise RuntimeError(f"All providers/models failed for role '{role}': {last_err}")

    # ------------------------------------------------------------------
    def stream(self, role: str, messages: List[dict], model: Optional[str] = None,
               **overrides: Any) -> Iterator[str]:
        m = model or self.config.model_for(role)
        params = {**self.config.gen_params(role), **overrides}
        self.limiter.wait(m, self.config.rate_limit(m))
        provider = self.registry.get()
        try:
            yield from provider.stream(m, messages, **params)
            with self._lock:
                self.stats.calls += 1
                self.stats.by_model[m] += 1
                self.stats.by_role[role] += 1
        except Exception as e:  # noqa: BLE001
            self.notify("warn", f"Stream failed ({e}); using non-stream fallback")
            yield self.chat(role, messages, model=model, **overrides).content

    # ------------------------------------------------------------------
    def ask(self, role: str, prompt: str, system: Optional[str] = None, **kw) -> str:
        msgs: List[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.chat(role, msgs, **kw).content

    def embed(self, texts: List[str], role: str = "embed") -> List[List[float]]:
        from contextlib import nullcontext
        tk = self.tick("embedding") if self.tick else nullcontext()
        with tk:
            return self._embed_impl(texts, role)

    def _embed_impl(self, texts: List[str], role: str = "embed") -> List[List[float]]:
        chain = self.config.model_chain(role)
        for provider_name in self.registry.order():
            provider = self.registry.providers.get(provider_name)
            if not provider or not provider.supports_embeddings:
                continue
            for m in chain:
                self.limiter.wait(m, self.config.rate_limit(m))
                try:
                    return provider.embed(m, texts)
                except Exception as e:  # noqa: BLE001
                    self.notify("warn", f"embed {m} failed: {str(e)[:120]}")
        raise RuntimeError("Embeddings unavailable on all providers")

    def moderate(self, texts: List[str]) -> List[dict]:
        m = self.config.model_for("safety")
        for pname in self.registry.order():
            p = self.registry.providers.get(pname)
            if p and p.supports_moderation:
                try:
                    return p.moderate(m, texts)
                except Exception as e:  # noqa: BLE001
                    self.notify("warn", f"moderation failed: {str(e)[:100]}")
        return []

    def ocr(self, document: dict) -> dict:
        m = self.config.model_for("ocr")
        for pname in self.registry.order():
            p = self.registry.providers.get(pname)
            if p and p.supports_ocr:
                return p.ocr(m, document)
        raise RuntimeError("No OCR-capable provider")

    # ------------------------------------------------------------------
    def _is_large(self, model: str) -> bool:
        # Only truly expensive IDs. hard_fallback is often medium-2508 (the
        # intended primary) — treating that as "large" skipped the best model.
        name = (model or "").lower()
        return "large" in name and "medium" not in name

    def _allow_large(self, task_id: str) -> bool:
        cap = int(self.config.get("autonomy.large_model_calls_per_task", 1))
        with self._lock:
            if self._large_calls[task_id] >= cap:
                return False
            self._large_calls[task_id] += 1
            return True

    def reset_task_budget(self, task_id: str) -> None:
        self._large_calls.pop(task_id, None)

    def key_status(self) -> Dict[str, list]:
        return self.registry.status()
