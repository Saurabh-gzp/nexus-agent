"""Shared urlopen watchdog — hung sockets must not stall the agent."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def urlopen_watchdog(req: urllib.request.Request, timeout: float,
                     grace: float = 5.0) -> bytes:
    box: Dict[str, Any] = {}

    def _do() -> None:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                box["raw"] = resp.read()
        except BaseException as e:  # noqa: BLE001
            box["err"] = e

    th = threading.Thread(target=_do, daemon=True)
    th.start()
    th.join(float(timeout) + float(grace))
    if th.is_alive():
        raise TimeoutError(f"watchdog: hung >{timeout}s")
    if "err" in box:
        raise box["err"]
    return box.get("raw") or b""


def json_watchdog(req: urllib.request.Request, timeout: float,
                  grace: float = 5.0) -> Dict[str, Any]:
    raw = urlopen_watchdog(req, timeout, grace)
    return json.loads(raw.decode("utf-8")) if raw else {}
