"""Multi-API-key manager with automatic failover.

Since this is an autonomous agent, it cannot stall when one key fails.
KeyRing tracks the health of every key and automatically moves to the next healthy one
and switches, notifying the user.

Key states:
    HEALTHY    -> normal use
    COOLING    -> 429/5xx ke baad temporary rest (cooldown_seconds)
    DEAD       -> 401/403/quota-exhausted (hard_fail_cooldown ke baad revive try)
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional


class KeyState(str, Enum):
    HEALTHY = "healthy"
    COOLING = "cooling"
    DEAD = "dead"


@dataclass
class ApiKey:
    value: str
    label: str
    provider: str
    state: KeyState = KeyState.HEALTHY
    cooldown_until: float = 0.0
    success: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    total_tokens: int = 0
    last_error: str = ""
    last_used: float = 0.0

    @property
    def masked(self) -> str:
        v = self.value
        return f"{v[:4]}…{v[-4:]}" if len(v) > 10 else "****"

    def available(self, now: Optional[float] = None) -> bool:
        now = now or time.time()
        if self.state is KeyState.HEALTHY:
            return True
        return now >= self.cooldown_until

    def to_dict(self) -> dict:
        return {
            "label": self.label, "provider": self.provider, "masked": self.masked,
            "state": self.state.value, "success": self.success, "failures": self.failures,
            "tokens": self.total_tokens, "last_error": self.last_error[:120],
            "cooldown_left": max(0, round(self.cooldown_until - time.time())),
        }


class KeyRing:
    """Thread-safe rotating pool of API keys for one provider."""

    def __init__(
        self,
        provider: str,
        keys: List[str],
        cooldown: int = 60,
        hard_cooldown: int = 600,
        notifier: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.provider = provider
        self.cooldown = cooldown
        self.hard_cooldown = hard_cooldown
        self.notify = notifier or (lambda level, msg: None)
        self._lock = threading.RLock()
        self._no_health_since: float = 0.0
        self._idx = 0
        self.keys: List[ApiKey] = [
            ApiKey(value=k.strip(), label=f"{provider}#{i + 1}", provider=provider)
            for i, k in enumerate(keys) if k and k.strip()
        ]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.keys)

    @property
    def healthy_count(self) -> int:
        now = time.time()
        return sum(1 for k in self.keys if k.available(now))

    # v1.8.4: honest stop instead of a silent sleep-loop. If at some moment
    # ZERO keys are healthy, the provider marks it; after `seconds` with no key
    # ever recovering, the provider raises a clear non-retryable error (24/7:
    # better an honest 'quota exhausted' than an agent that looks alive but
    # does nothing — live run sat in nanosleep with 0 sockets for 8+ minutes).
    def mark_all_down(self) -> None:
        with self._lock:
            if self._no_health_since <= 0:
                self._no_health_since = time.time()

    def mark_healthy(self) -> None:
        with self._lock:
            self._no_health_since = 0.0

    def all_down_for(self, seconds: float) -> bool:
        with self._lock:
            return (self._no_health_since > 0
                    and time.time() - self._no_health_since >= seconds)



    def acquire(self, exclude: Optional[set] = None) -> Optional[ApiKey]:
        """Round-robin pick of the next available key. None only if the ring is empty."""
        exclude = exclude or set()
        with self._lock:
            n = len(self.keys)
            if n == 0:
                return None
            now = time.time()
            # revive expired cooldowns
            for k in self.keys:
                if k.state is not KeyState.HEALTHY and now >= k.cooldown_until:
                    prev = k.state
                    k.state = KeyState.HEALTHY
                    k.consecutive_failures = 0
                    if prev is KeyState.DEAD:
                        self.notify("info", f"Key {k.label} revived (cooldown over)")
            for offset in range(n):
                i = (self._idx + offset) % n
                k = self.keys[i]
                if k.label in exclude:
                    continue
                if k.available(now):
                    self._idx = (i + 1) % n
                    k.last_used = now
                    return k
            return None

    def acquire_or_wait(self, exclude: Optional[set] = None,
                        max_wait: float = 45.0) -> Optional[ApiKey]:
        """Never give up while keys exist: if all are cooling, wait for the soonest one.

        An autonomous agent must not die because every key is momentarily rate-limited.
        Returns None only when the ring holds no keys at all.
        """
        k = self.acquire(exclude)
        if k is not None:
            return k
        with self._lock:
            if not self.keys:
                return None
            pool = [x for x in self.keys if x.label not in (exclude or set())] or self.keys
            # prefer a cooling key over a dead one, then the earliest to recover
            soonest = min(pool, key=lambda x: (x.state is KeyState.DEAD, x.cooldown_until))
            wait = max(0.0, soonest.cooldown_until - time.time())
        if wait > max_wait:
            self.notify("warn",
                        f"All keys cooling {int(wait)}s > max_wait {int(max_wait)}s "
                        f"— not forcing {soonest.label} healthy (honest pause)")
            return None
        if wait > 0:
            self.notify("warn", f"All keys cooling — waiting {wait:.0f}s for {soonest.label}")
            time.sleep(wait + 0.25)
        with self._lock:
            if time.time() >= soonest.cooldown_until:
                soonest.state = KeyState.HEALTHY
                soonest.cooldown_until = 0.0
            soonest.last_used = time.time()
        return soonest if soonest.available() else None

    # ------------------------------------------------------------------
    def report_success(self, key: ApiKey, tokens: int = 0) -> None:
        with self._lock:
            key.success += 1
            key.consecutive_failures = 0
            key.total_tokens += tokens
            if key.state is KeyState.COOLING:
                key.state = KeyState.HEALTHY

    def report_failure(self, key: ApiKey, status: Optional[int], error: str,
                       retry_after: float = 0.0) -> None:
        """Mark key unhealthy based on HTTP status. Returns nothing; caller rotates."""
        with self._lock:
            key.failures += 1
            key.consecutive_failures += 1
            key.last_error = error or f"HTTP {status}"
            now = time.time()

            if status in (401, 403):
                key.state = KeyState.DEAD
                key.cooldown_until = now + self.hard_cooldown
                self.notify("error", f"Key {key.label} unauthorized -> disabled {self.hard_cooldown}s")
            elif status == 429:
                # honour Retry-After; otherwise back off progressively (rate limits are short)
                rest = retry_after if retry_after > 0 else min(
                    self.cooldown, 4.0 * key.consecutive_failures)
                key.state = KeyState.COOLING
                key.cooldown_until = now + rest
                self.notify("warn", f"Key {key.label} rate-limited -> cooling {rest:.0f}s")
            elif status and status >= 500:
                key.state = KeyState.COOLING
                key.cooldown_until = now + min(self.cooldown, 30)
                self.notify("warn", f"Key {key.label} server error {status} -> short cooldown")
            else:
                # network / unknown -> escalate only after repeated failures
                if key.consecutive_failures >= 3:
                    key.state = KeyState.COOLING
                    key.cooldown_until = now + self.cooldown
                    self.notify("warn", f"Key {key.label} unstable -> cooling {self.cooldown}s")

    def status(self) -> List[dict]:
        with self._lock:
            return [k.to_dict() for k in self.keys]

    def remove_key(self, value: str) -> bool:
        """Remove a key from the live pool (for KeyManager /key delete)."""
        with self._lock:
            v = (value or "").strip()
            for i, k in enumerate(self.keys):
                if k.value == v:
                    self.keys.pop(i)
                    return True
        return False

    def add_key(self, value: str, label: Optional[str] = None) -> ApiKey:
        with self._lock:
            k = ApiKey(value=value.strip(), provider=self.provider,
                       label=label or f"{self.provider}#{len(self.keys) + 1}")
            self.keys.append(k)
            return k

    # ------------------------------------------------------------------
    @staticmethod
    def discover(provider: str, env_names: List[str], keyfile: Optional[Path] = None) -> List[str]:
        """Collect keys from env vars, numbered env vars and keys.json."""
        found: List[str] = []

        def _add(v: Optional[str]) -> None:
            if v and v.strip() and v.strip() not in found:
                found.append(v.strip())

        for name in env_names or []:
            _add(os.getenv(name))
        # PROVIDER_API_KEY_1..20 convention (multi-key pools: 10+ keys in one run)
        base = f"{provider.upper()}_API_KEY"
        _add(os.getenv(base))
        for i in range(1, 21):
            _add(os.getenv(f"{base}_{i}"))
        # comma separated bulk — accept ALL spellings in use (historic bug: docs
        # said MISTRAL_APIS but the code read MISTRAL_API_KEYS/MISTRALS; a whole
        # 8-key pool silently reduced to 1 key -> all traffic on key#1, 429s)
        for bulk_name in ("MISTRAL_APIS", f"{base}S", f"{provider.upper()}S"):
            bulk = os.getenv(bulk_name)
            if bulk:
                for part in bulk.split(","):
                    _add(part)

        if keyfile and keyfile.exists():
            try:
                data = json.loads(keyfile.read_text(encoding="utf-8"))
                entry = data.get(provider, [])
                if isinstance(entry, str):
                    _add(entry)
                else:
                    for v in entry:
                        _add(v if isinstance(v, str) else v.get("key"))
            except Exception:
                pass
        return found
