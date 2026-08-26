"""Workspace path-boundary checks (no string-prefix sandbox)."""
from __future__ import annotations

from pathlib import Path


def in_workspace(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False
