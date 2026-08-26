"""Workspace-scoped git tools. Force-push to main/master is blocked."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .base import Risk, ToolRegistry, ToolResult


class GitTools:
    def __init__(self, workspace: Path):
        self.root = Path(workspace).resolve()

    def _run(self, args: list, timeout: int = 30) -> ToolResult:
        try:
            p = subprocess.run(
                ["git", *args], cwd=str(self.root), capture_output=True,
                text=True, timeout=timeout,
            )
        except FileNotFoundError:
            return ToolResult(False, error="git is not installed")
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, error=str(e))
        out = ((p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")).strip()
        return ToolResult(p.returncode == 0, output=out[:8000] or "(empty)",
                          error="" if p.returncode == 0 else (p.stderr or out)[:800])

    def git_status(self) -> ToolResult:
        return self._run(["status", "-sb", "--untracked-files=all"])

    def git_diff(self, staged: bool = False) -> ToolResult:
        args = ["diff", "--stat"]
        if staged:
            args.append("--cached")
        return self._run(args)

    def git_log(self, n: int = 8) -> ToolResult:
        n = max(1, min(int(n or 8), 30))
        return self._run(["log", f"-{n}", "--oneline", "--decorate"])

    def git_add(self, paths: str = ".") -> ToolResult:
        from .paths import in_workspace
        parts = [p for p in (paths or ".").split() if p and p not in {";", "&", "|"}]
        if not parts:
            parts = ["."]
        for p in parts:
            if p.startswith("-"):
                return ToolResult(False, error="BLOCKED: git_add flags not allowed")
            abs_p = (self.root / p).resolve()
            if p not in (".",) and not in_workspace(abs_p, self.root):
                return ToolResult(False, error=f"BLOCKED: path escapes workspace: {p}")
        return self._run(["add", "--", *parts])

    def git_commit(self, message: str) -> ToolResult:
        msg = (message or "").strip()
        if not msg or len(msg) < 3:
            return ToolResult(False, error="commit message required")
        if msg.startswith("-"):
            return ToolResult(False, error="BLOCKED: message looks like a flag")
        return self._run(["commit", "-m", msg])

    def register(self, reg: ToolRegistry) -> None:
        S = {"type": "string"}
        I = {"type": "integer"}
        B = {"type": "boolean"}
        who = ["coder", "supervisor", "worker", "solo"]
        reg.add("git_status", "git status of the workspace repo (short).",
                {"type": "object", "properties": {}}, self.git_status, Risk.READ_ONLY, agents=who)
        reg.add("git_diff", "git diff --stat (optionally staged).",
                {"type": "object", "properties": {"staged": B}},
                self.git_diff, Risk.READ_ONLY, agents=who)
        reg.add("git_log", "recent git log --oneline.",
                {"type": "object", "properties": {"n": I}},
                self.git_log, Risk.READ_ONLY, agents=who)
        reg.add("git_add", "Stage workspace files (git add). Paths stay inside the workspace. No remote.",
                {"type": "object", "properties": {"paths": S}},
                self.git_add, Risk.WRITE, agents=["coder", "supervisor", "solo"])
        reg.add("git_commit", "Create a local commit. Never pushes. Requires a real message.",
                {"type": "object", "properties": {"message": S}, "required": ["message"]},
                self.git_commit, Risk.WRITE, agents=["coder", "supervisor", "solo"])
