"""Generic OpenAI-compatible provider (OpenAI, Groq, Together, Ollama, LM Studio...).

Just set `type: openai_compatible` + base_url + env_keys in the config, done.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

from .base import BaseProvider, ChatResult, ProviderError
from .httpwatch import json_watchdog


class OpenAICompatibleProvider(BaseProvider):
    name = "openai_compatible"
    supports_tools = True
    supports_embeddings = True

    def __init__(self, cfg: dict, keyring, notifier=None, name: Optional[str] = None):
        super().__init__(cfg, keyring, notifier)
        self.name = name or cfg.get("name", "openai_compatible")
        self.base_url = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.timeout = int(cfg.get("timeout", 180))
        self.static_key = cfg.get("api_key")   # for local servers (ollama)
        self.watchdog_budget_slack = int(cfg.get("watchdog_budget_slack", 90))
        self.watchdog_grace = int(cfg.get("watchdog_grace", 5))

    def _request(self, path: str, payload: dict) -> Dict[str, Any]:
        tried: set = set()
        last: Optional[ProviderError] = None
        rounds = max(2, len(self.keyring) * 2 or 2)
        call_t0 = time.time()
        call_budget = self.timeout + self.watchdog_budget_slack

        for attempt in range(rounds):
            remaining = call_budget - (time.time() - call_t0)
            if remaining <= 0:
                break
            key = self.keyring.acquire(exclude=tried) if len(self.keyring) else None
            token = key.value if key else (self.static_key or "none")
            if key:
                tried.add(key.label)
            req = urllib.request.Request(
                f"{self.base_url}{path}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.time()
            attempt_timeout = int(min(self.timeout, remaining))
            try:
                data = json_watchdog(req, attempt_timeout, self.watchdog_grace)
                if key:
                    self.keyring.report_success(key, int((data.get("usage") or {}).get("total_tokens") or 0))
                data["_key_label"] = key.label if key else "static"
                data["_latency"] = time.time() - t0
                return data
            except TimeoutError as e:
                if key:
                    self.keyring.report_failure(key, None, str(e))
                last = ProviderError(str(e), retryable=True)
                self.notify("warn", f"{self.name}: {e} — watchdog skip")
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "ignore")[:300]
                if key:
                    self.keyring.report_failure(key, e.code, detail)
                last = ProviderError(f"HTTP {e.code}: {detail}", status=e.code)
                if e.code in (400, 404, 422):
                    raise last
                self.notify("warn", f"{self.name}: HTTP {e.code}, rotating key")
            except Exception as e:  # noqa: BLE001
                if key:
                    self.keyring.report_failure(key, None, str(e))
                last = ProviderError(str(e))
                time.sleep(min(6, 1.5 ** attempt))
        raise last or ProviderError(f"{self.name} request failed")

    def chat(self, model: str, messages: List[dict], tools: Optional[List[dict]] = None,
             **params: Any) -> ChatResult:
        payload: Dict[str, Any] = {"model": model, "messages": messages}
        for k in ("temperature", "max_tokens", "top_p", "stop", "response_format"):
            if params.get(k) is not None:
                payload[k] = params[k]
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = params.get("tool_choice", "auto")
        data = self._request("/chat/completions", payload)
        ch = (data.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        usage = data.get("usage") or {}
        return ChatResult(
            content=msg.get("content") or "",
            tool_calls=msg.get("tool_calls") or [],
            model=data.get("model", model), provider=self.name,
            key_label=data.get("_key_label", ""),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=ch.get("finish_reason", ""),
            latency=float(data.get("_latency", 0)), raw=data,
        )

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        data = self._request("/embeddings", {"model": model, "input": texts})
        rows = sorted(data.get("data", []), key=lambda r: r.get("index", 0))
        return [r["embedding"] for r in rows]
