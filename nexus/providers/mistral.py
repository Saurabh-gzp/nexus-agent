"""Mistral provider — pure-stdlib HTTP (Termux friendly, no heavy SDK).

Har call automatically:
  * picks a healthy key from the KeyRing
  * on 429/401/5xx it switches to another key (user is notified)
  * the model fallback chain is handled by the caller (LLMClient)
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

from .base import BaseProvider, ChatResult, ProviderError


class MistralProvider(BaseProvider):
    name = "mistral"
    supports_tools = True
    supports_embeddings = True
    supports_moderation = True
    supports_ocr = True

    def __init__(self, cfg: dict, keyring, notifier=None):
        super().__init__(cfg, keyring, notifier)
        self.base_url = cfg.get("base_url", "https://api.mistral.ai/v1").rstrip("/")
        self.timeout = int(cfg.get("timeout", 180))
        self.max_rotations = int(cfg.get("max_key_rotations_per_call", 6))
        # v1.8.6 watchdog knobs
        self.watchdog_budget_slack = int(cfg.get("watchdog_budget_slack", 90))
        self.watchdog_grace = int(cfg.get("watchdog_grace", 5))

    # ------------------------------------------------------------------
    def _request(self, path: str, payload: dict, timeout: Optional[int] = None) -> Dict[str, Any]:
        """POST with automatic key rotation."""
        tried: set = set()
        last_err: Optional[ProviderError] = None
        # Two full walks of EVERY key, then give up (caller may model-fallback).
        # Pass 1: keys 1..N. Pass 2: same keys again. Never jump to a weaker
        # model after a single 429.
        n_keys = max(len(self.keyring) or 1, 1)
        ring_passes = 2
        rotations = n_keys * ring_passes
        call_t0 = time.time()
        # whole-call wall-clock cap: timeout for one good generation +
        # slack for failing over the remaining keys (×2 passes).
        call_budget = self.timeout + self.watchdog_budget_slack * ring_passes

        for attempt in range(rotations):
            if attempt == n_keys:
                tried.clear()
                self.notify(
                    "warn",
                    f"All {n_keys} key(s) failed on this model — second full pass "
                    "before any model fallback",
                )
            # v1.8.4: honest stop — if no key has been healthy for 90s+, the
            # quota is gone (not a storm). Raise instead of spinning in circles.
            if self.keyring.healthy_count == 0:
                self.keyring.mark_all_down()
                if self.keyring.all_down_for(90):
                    raise ProviderError(
                        "ALL API keys have been rate-limited / quota-exhausted for "
                        "90s+ (nothing is recovering). Pausing honestly instead of "
                        "retrying in circles — add fresh keys or wait.", status=429,
                        retryable=False)
            else:
                self.keyring.mark_healthy()
            key = self.keyring.acquire(exclude=tried)
            if key is None:
                # every key tried/cooling — wait for the soonest instead of dying
                key = self.keyring.acquire_or_wait(exclude=tried if len(tried) < len(self.keyring) else None)
                if key is None:
                    raise ProviderError(
                        "No API keys available for mistral. Add one with `/keys add <key>` "
                        "or set MISTRAL_API_KEY.", retryable=False)
                tried.discard(key.label)
            tried.add(key.label)

            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}{path}",
                data=body,
                headers={
                    "Authorization": f"Bearer {key.value}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "nexus-agent/1.0",
                },
                method="POST",
            )
            t0 = time.time()
            # v1.8.6 WATCHDOG: a single urlopen can hang PAST its socket timeout
            # (live runs #4/#5: one hung request kept the spinner ticking for
            # 12-28 min). The request runs in a worker thread; if it does not
            # finish within the attempt budget we skip that key and move on —
            # the whole call is capped at timeout+90s wall clock.
            remaining = call_budget - (time.time() - call_t0)
            if remaining <= 0:
                break
            attempt_timeout = int(min(self.timeout, remaining))
            box: Dict[str, Any] = {}

            def _do() -> None:
                try:
                    with urllib.request.urlopen(req, timeout=attempt_timeout) as resp:
                        box["data"] = json.loads(resp.read().decode("utf-8"))
                except BaseException as e:  # noqa: BLE001 — capture, handle below
                    box["err"] = e

            th = threading.Thread(target=_do, daemon=True)
            th.start()
            th.join(attempt_timeout + self.watchdog_grace)
            if th.is_alive():
                # hung beyond every budget — watchdog skip, key gets a failure streak
                self.keyring.report_failure(key, None, f"watchdog: hung >{attempt_timeout}s")
                last_err = ProviderError(
                    f"Key {key.label} hung {attempt_timeout}s (watchdog killed it)",
                    retryable=True)
                self.notify("warn", f"{key.label} HUNG >{attempt_timeout}s — watchdog skip")
                time.sleep(0.5)
                continue
            err = box.get("err")
            if err is not None and isinstance(err, urllib.error.HTTPError):
                try:
                    detail = err.read().decode("utf-8")[:400]
                except Exception:
                    detail = str(err)
                retry_after = 0.0
                try:
                    retry_after = float(err.headers.get("Retry-After") or 0)
                except (TypeError, ValueError):
                    retry_after = 0.0
                self.keyring.report_failure(key, err.code, detail, retry_after)
                last_err = ProviderError(f"HTTP {err.code}: {detail}", status=err.code,
                                         retryable=err.code in (408, 409, 429) or err.code >= 500)
                if err.code in (400, 404, 422):  # payload/model problem -> switching won't help
                    raise last_err
                if self.keyring.healthy_count > 0:
                    self.notify("warn", f"Switching key after HTTP {err.code} ({key.label})")
                    continue
                time.sleep(min(8, 1.5 ** attempt))
            elif err is not None and isinstance(err, (urllib.error.URLError, TimeoutError, OSError)):
                self.keyring.report_failure(key, None, str(err))
                last_err = ProviderError(f"Network error: {err}", status=None, retryable=True)
                self.notify("warn", f"Network issue on {key.label}: {err}")
                time.sleep(min(4, 1.5 ** attempt))
            elif err is not None and isinstance(err, json.JSONDecodeError):
                last_err = ProviderError(f"Bad JSON from API: {err}", retryable=True)
            else:
                data = box.get("data")
                if data is None:
                    last_err = ProviderError("Empty response from API", retryable=True)
                    continue
                usage = data.get("usage") or {}
                self.keyring.report_success(key, int(usage.get("total_tokens") or 0))
                data["_key_label"] = key.label
                data["_latency"] = time.time() - t0
                return data

        raise last_err or ProviderError(
            f"Request failed after {rotations} keys within the {call_budget:.0f}s "
            "call budget \u2014 network trouble or all keys exhausted")

    # ------------------------------------------------------------------
    def chat(self, model: str, messages: List[dict], tools: Optional[List[dict]] = None,
             **params: Any) -> ChatResult:
        payload: Dict[str, Any] = {"model": model, "messages": messages}
        for k in ("temperature", "max_tokens", "top_p", "random_seed", "stop",
                  "presence_penalty", "frequency_penalty", "response_format",
                  "parallel_tool_calls", "prompt_mode"):
            if params.get(k) is not None:
                payload[k] = params[k]
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = params.get("tool_choice", "auto")

        data = self._request("/chat/completions", payload)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        content = msg.get("content") or ""
        if isinstance(content, list):   # multimodal chunk list
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        return ChatResult(
            content=content,
            tool_calls=msg.get("tool_calls") or [],
            model=data.get("model", model),
            provider=self.name,
            key_label=data.get("_key_label", ""),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=choice.get("finish_reason", ""),
            latency=float(data.get("_latency", 0.0)),
            raw=data,
        )

    # ------------------------------------------------------------------
    def stream(self, model: str, messages: List[dict], tools: Optional[List[dict]] = None,
               **params: Any) -> Iterator[str]:
        """SSE streaming with key rotation on connect failure."""
        payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        for k in ("temperature", "max_tokens", "top_p"):
            if params.get(k) is not None:
                payload[k] = params[k]
        if tools:
            payload["tools"] = tools

        tried: set = set()
        for _ in range(max(2, len(self.keyring))):
            key = self.keyring.acquire(exclude=tried)
            if key is None:
                break
            tried.add(key.label)
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": f"Bearer {key.value}",
                         "Content-Type": "application/json",
                         "Accept": "text/event-stream"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8").strip()
                        if not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if chunk == "[DONE]":
                            self.keyring.report_success(key)
                            return
                        try:
                            j = json.loads(chunk)
                            delta = (j.get("choices") or [{}])[0].get("delta", {})
                            if piece := delta.get("content"):
                                yield piece
                        except json.JSONDecodeError:
                            continue
                self.keyring.report_success(key)
                return
            except urllib.error.HTTPError as e:
                self.keyring.report_failure(key, e.code, str(e))
                self.notify("warn", f"Stream failed on {key.label} (HTTP {e.code}) -> switching key")
            except Exception as e:  # noqa: BLE001
                self.keyring.report_failure(key, None, str(e))
        # last resort: non-streaming
        yield self.chat(model, messages, tools, **params).content

    # ------------------------------------------------------------------
    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        BATCH = 32
        for i in range(0, len(texts), BATCH):
            batch = [t[:8000] for t in texts[i:i + BATCH]]
            data = self._request("/embeddings", {"model": model, "input": batch})
            rows = sorted(data.get("data", []), key=lambda r: r.get("index", 0))
            out.extend(r["embedding"] for r in rows)
        return out

    def moderate(self, model: str, texts: List[str]) -> List[dict]:
        data = self._request("/moderations", {"model": model, "input": texts})
        return data.get("results", [])

    def ocr(self, model: str, document: dict) -> dict:
        return self._request("/ocr", {"model": model, "document": document},
                             timeout=max(self.timeout, 240))
