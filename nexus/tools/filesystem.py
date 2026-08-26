"""Filesystem tools — sandboxed to workspace root."""
from __future__ import annotations

import fnmatch
import os
import shutil
from pathlib import Path
from typing import List, Optional

from .base import Risk, ToolRegistry, ToolResult
from .paths import in_workspace

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".nexus",
             "dist", "build", ".next", ".cache", "target", ".idea"}
BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".tar", ".gz",
              ".exe", ".so", ".dll", ".jar", ".mp4", ".mp3", ".woff", ".ttf"}


class FileSystemTools:
    def __init__(self, workspace: Path, sandbox: bool = True, max_read_kb: int = 400):
        self.root = Path(workspace).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.sandbox = sandbox
        self.max_read = max_read_kb * 1024
        self.write_scope: Optional[Path] = None   # per-goal project isolation

    def set_write_scope(self, subdir: Optional[str]) -> None:
        """The engine sets this per build-goal — new files must be
        written inside the project folder only (reads allowed anywhere)."""
        if subdir is None:
            self.write_scope = None
        else:
            p = (self.root / subdir).resolve()
            if in_workspace(p, self.root):
                p.mkdir(parents=True, exist_ok=True)
                self.write_scope = p

    # ------------------------------------------------------------------
    def _resolve(self, path: str) -> Path:
        p = Path(os.path.expanduser(str(path)))
        if not p.is_absolute():
            rel = Path(*[seg for seg in p.parts if seg not in (".", "")])
            # Agents often write "workspace/foo" although cwd IS the workspace
            # root — strip accidental duplication of the root folder name.
            while len(rel.parts) > 1 and rel.parts[0] == self.root.name:
                rel = rel.relative_to(rel.parts[0])
            p = self.root / rel
        p = p.resolve()
        if self.sandbox and in_workspace(p, self.root) and not p.exists():
            # Absolute paths with the same doubled prefix ("/ws/workspace/x")
            # that do NOT exist -> point at the de-duplicated location.
            try:
                rel = p.relative_to(self.root)
                parts = rel.parts
                while len(parts) > 1 and parts[0] == self.root.name:
                    parts = parts[1:]
                if parts != rel.parts:
                    cand = (self.root / Path(*parts)).resolve()
                    if in_workspace(cand, self.root):
                        p = cand
            except ValueError:
                pass
        if self.sandbox and not in_workspace(p, self.root):
            raise PermissionError(f"Path outside workspace sandbox: {p}")
        return p

    def _resolve_write(self, path: str) -> Path:
        """With write-scope active, relative paths resolve inside the project folder."""
        if self.write_scope is not None:
            s = str(path)
            sc = str(self.write_scope.relative_to(self.root))
            if (not s.startswith(("/", "~")) and not s.startswith("projects/")
                    and not s.startswith(sc)):
                return self._resolve(str(self.write_scope / s))
        return self._resolve(path)

    def _write_allowed(self, p: Path) -> bool:
        """with write_scope active, new writes go inside the project folder only."""
        if self.write_scope is None:
            return True
        return in_workspace(p, self.write_scope)

    def _rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    # ------------------------------------------------------------------
    def read_file(self, path: str, start_line: int = 1, end_line: int = 0) -> ToolResult:
        try:
            p = self._resolve(path)
            if not p.exists():
                return ToolResult(False, error=f"File not found: {path}")
            if p.is_dir():
                return ToolResult(False, error=f"'{path}' is a directory — use list_dir")
            if p.suffix.lower() in BINARY_EXT:
                return ToolResult(False, error=f"Binary file ({p.suffix}); cannot read as text")
            if p.stat().st_size > self.max_read:
                return ToolResult(False, error=f"File too large ({p.stat().st_size // 1024} KB)")
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            s = max(1, start_line)
            e = end_line if end_line and end_line >= s else len(lines)
            sel = lines[s - 1:e]
            body = "\n".join(f"{s + i:>5}| {ln}" for i, ln in enumerate(sel))
            head = f"# {self._rel(p)} (lines {s}-{min(e, len(lines))} of {len(lines)})\n"
            return ToolResult(True, output=head + body, data={"lines": len(lines)})
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, error=str(e))

    def write_file(self, path: str, content: str, mode: str = "overwrite") -> ToolResult:
        try:
            p = self._resolve_write(path)
            if not self._write_allowed(p):
                return ToolResult(False, error=(
                    f"Blocked: during this goal new files go inside the project folder "
                    f"'{self.write_scope.relative_to(self.root)}/'. Write "
                    f"{self.write_scope / p.name} instead (reads are unrestricted)."))
            p.parent.mkdir(parents=True, exist_ok=True)
            existed = p.exists()
            if mode == "append" and existed:
                with p.open("a", encoding="utf-8") as f:
                    f.write(content)
                action = "Appended to"
            else:
                p.write_text(content, encoding="utf-8")
                action = "Updated" if existed else "Created"
            n = len(content.splitlines())
            return ToolResult(True, output=f"{action} {self._rel(p)} ({n} lines, {len(content)} chars)",
                              data={"path": str(p)})
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, error=str(e))

    def edit_file(self, path: str, old_text: str, new_text: str, count: int = 1) -> ToolResult:
        try:
            p = self._resolve_write(path)
            if not self._write_allowed(p):
                return ToolResult(False, error="Blocked: this file is outside the active project folder.")
            if not p.exists():
                return ToolResult(False, error=f"File not found: {path}")
            src = p.read_text(encoding="utf-8", errors="replace")
            if old_text not in src:
                # whitespace-tolerant retry
                import re
                pat = re.compile(r"\s*".join(re.escape(t) for t in old_text.split()))
                m = pat.search(src)
                if not m:
                    return ToolResult(False, error="old_text not found in file. Read the file first.")
                out = src[:m.start()] + new_text + src[m.end():]
                hits = 1
            else:
                hits = src.count(old_text) if count <= 0 else min(count, src.count(old_text))
                out = src.replace(old_text, new_text, hits)
            p.write_text(out, encoding="utf-8")
            return ToolResult(True, output=f"Edited {self._rel(p)} ({hits} replacement(s))")
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, error=str(e))

    def list_dir(self, path: str = ".", depth: int = 2, show_hidden: bool = False) -> ToolResult:
        try:
            base = self._resolve(path)
            if not base.exists():
                return ToolResult(False, error=f"Not found: {path}")
            lines: List[str] = [f"{self._rel(base) or '.'}/"]

            def walk(d: Path, prefix: str, level: int) -> None:
                if level > depth:
                    return
                try:
                    entries = sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
                except PermissionError:
                    return
                entries = [e for e in entries if show_hidden or not e.name.startswith(".")]
                entries = [e for e in entries if e.name not in SKIP_DIRS]
                for i, e in enumerate(entries[:200]):
                    last = i == len(entries) - 1
                    branch = "└── " if last else "├── "
                    if e.is_dir():
                        lines.append(f"{prefix}{branch}{e.name}/")
                        walk(e, prefix + ("    " if last else "│   "), level + 1)
                    else:
                        try:
                            sz = e.stat().st_size
                            hs = f"{sz}B" if sz < 1024 else (f"{sz // 1024}KB" if sz < 1048576 else f"{sz // 1048576}MB")
                        except OSError:
                            hs = "?"
                        lines.append(f"{prefix}{branch}{e.name} ({hs})")

            walk(base, "", 1)
            return ToolResult(True, output="\n".join(lines[:400]))
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, error=str(e))

    def search_files(self, pattern: str, path: str = ".", regex: bool = False,
                     max_results: int = 60) -> ToolResult:
        """grep-like content search."""
        try:
            import re
            base = self._resolve(path)
            rx = re.compile(pattern if regex else re.escape(pattern), re.IGNORECASE)
            hits: List[str] = []
            for root, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
                for fn in files:
                    fp = Path(root) / fn
                    if fp.suffix.lower() in BINARY_EXT:
                        continue
                    try:
                        if fp.stat().st_size > self.max_read:
                            continue
                        for i, line in enumerate(fp.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                            if rx.search(line):
                                hits.append(f"{self._rel(fp)}:{i}: {line.strip()[:160]}")
                                if len(hits) >= max_results:
                                    raise StopIteration
                    except (StopIteration, OSError):
                        if len(hits) >= max_results:
                            break
                        continue
                if len(hits) >= max_results:
                    break
            return ToolResult(True, output="\n".join(hits) if hits else f"No matches for '{pattern}'",
                              data={"count": len(hits)})
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, error=str(e))

    def find_files(self, glob: str = "*", path: str = ".", max_results: int = 100) -> ToolResult:
        try:
            base = self._resolve(path)
            out: List[str] = []
            for root, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for fn in files:
                    if fnmatch.fnmatch(fn, glob):
                        out.append(self._rel(Path(root) / fn))
                        if len(out) >= max_results:
                            break
                if len(out) >= max_results:
                    break
            return ToolResult(True, output="\n".join(out) if out else f"No files match '{glob}'",
                              data={"files": out})
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, error=str(e))

    def delete_path(self, path: str = "", src: str = "", target: str = "") -> ToolResult:
        """Live bug fix: agent kabhi `src` bhejta hai kabhi `path` — aliases accept."""
        path = path or src or target
        try:
            p = self._resolve(path)
            if not p.exists():
                return ToolResult(False, error=f"Not found: {path}")
            if p == self.root:
                return ToolResult(False, error="Refusing to delete workspace root")
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            return ToolResult(True, output=f"Deleted {self._rel(p)}")
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, error=str(e))

    def move_path(self, src: str, dst: str) -> ToolResult:
        try:
            s, d = self._resolve(src), self._resolve_write(dst)
            if not self._write_allowed(d):
                return ToolResult(False, error="Blocked: destination outside the active project folder.")
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(s), str(d))
            return ToolResult(True, output=f"Moved {self._rel(s)} -> {self._rel(d)}")
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, error=str(e))

    # ------------------------------------------------------------------
    def register(self, reg: ToolRegistry) -> None:
        S = {"type": "string"}
        I = {"type": "integer"}
        reg.add("read_file", "Read a text file with line numbers. Always read before editing.",
                {"type": "object", "properties": {"path": S, "start_line": I, "end_line": I},
                 "required": ["path"]},
                self.read_file, Risk.READ_ONLY)
        reg.add("write_file", "Create or overwrite a file with full content.",
                {"type": "object", "properties": {
                    "path": S, "content": S,
                    "mode": {"type": "string", "enum": ["overwrite", "append"]}},
                 "required": ["path", "content"]},
                self.write_file, Risk.WRITE,
                agents=["supervisor", "coder", "worker", "researcher", "solo"])
        reg.add("edit_file", "Replace exact text inside an existing file (surgical edit).",
                {"type": "object", "properties": {"path": S, "old_text": S, "new_text": S, "count": I},
                 "required": ["path", "old_text", "new_text"]},
                self.edit_file, Risk.WRITE,
                agents=["supervisor", "coder", "worker", "solo"])
        reg.add("list_dir", "Show directory tree of the workspace.",
                {"type": "object", "properties": {"path": S, "depth": I,
                                                  "show_hidden": {"type": "boolean"}}},
                self.list_dir, Risk.READ_ONLY)
        reg.add("search_files", "Search file CONTENTS for text/regex (grep).",
                {"type": "object", "properties": {"pattern": S, "path": S,
                                                  "regex": {"type": "boolean"}, "max_results": I},
                 "required": ["pattern"]},
                self.search_files, Risk.READ_ONLY)
        reg.add("find_files", "Find files by name glob pattern e.g. '*.py'.",
                {"type": "object", "properties": {"glob": S, "path": S, "max_results": I}},
                self.find_files, Risk.READ_ONLY)
        reg.add("delete_path",
                "Delete a file or folder (accepts path= or src=). Requires human approval. "
                "This is the ONLY way to delete — rm/shred/python deletes are hard-blocked.",
                {"type": "object", "properties": {"path": S, "src": S}, "required": ["path"]},
                self.delete_path, Risk.DESTRUCTIVE,
                agents=["supervisor", "coder", "worker", "solo"], approval=True)
        reg.add("move_path", "Move or rename a file/folder.",
                {"type": "object", "properties": {"src": S, "dst": S}, "required": ["src", "dst"]},
                self.move_path, Risk.WRITE, agents=["supervisor", "coder", "worker", "solo"])
