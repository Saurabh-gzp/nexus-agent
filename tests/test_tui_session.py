#!/usr/bin/env python3
"""Real interactive TUI test — spawns nexus.py in a pseudo-terminal (pty),
sends messages like a human user, captures ANSI + rendered screen snapshots.

Usage: python3 tests/test_tui_session.py [fast]
  fast = skip the long autonomous goal, only smoke commands
"""
import fcntl
import os
import pty
import re
import select
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

try:
    import pyte
except ImportError:          # optional dev dependency (real-terminal testing)
    pyte = None

ROOT = str(Path(__file__).resolve().parent.parent)   # repo root (was hardcoded)
ART = os.path.join(ROOT, ".nexus", "tui_artifacts")
PROMPT = "nexus ❯"          # rendered prompt marker
ALLOW = "Allow?"             # approval prompt marker

COLS, ROWS = 96, 44


class Session:
    def __init__(self, cmd, cwd=ROOT, env=None):
        if pyte is None:
            raise SystemExit("pyte missing — install with: pip install pyte")
        os.makedirs(ART, exist_ok=True)
        self.master, slave = pty.openpty()
        fcntl.ioctl(self.master, termios.TIOCSWINSZ,
                    struct.pack("HHHH", ROWS, COLS, 0, 0))
        e = dict(os.environ)
        e.setdefault("TERM", "xterm-256color")
        e.pop("CI", None)
        if env:
            e.update(env)
        self.proc = subprocess.Popen(cmd, stdin=slave, stdout=slave, stderr=slave,
                                     cwd=cwd, env=e, start_new_session=True)
        os.close(slave)
        self.buf = b""
        self.ansi_path = os.path.join(ART, "session.ansi")
        open(self.ansi_path, "wb").close()
        # pyte screen for rendering snapshots
        self.screen = pyte.Screen(COLS, ROWS)
        self.stream = pyte.ByteStream(self.screen)
        self.shots = []

    # ---------------- low level ----------------
    def _pump(self, timeout):
        r, _, _ = select.select([self.master], [], [], timeout)
        if not r:
            return False
        try:
            chunk = os.read(self.master, 65536)
        except OSError:
            return False
        if not chunk:
            return False
        self.buf += chunk
        self.stream.feed(chunk)
        with open(self.ansi_path, "ab") as f:
            f.write(chunk)
        return True

    def wait_for(self, pattern, timeout=120, quiet=False):
        """Read until regex pattern appears in NEW terminal output or timeout."""
        rx_s = re.compile(pattern)
        rx_b = re.compile(pattern.encode())
        mark = len(self.buf)          # only look at output produced from now on
        t0 = time.time()
        while time.time() - t0 < timeout:
            text = "\n".join(self.screen.display)
            if rx_s.search(text):
                return True
            # also search raw buffer produced since we started waiting
            if rx_b.search(self.buf[mark:]):
                return True
            self._pump(0.4)
        if not quiet:
            self.snapshot(f"TIMEOUT-{pattern[:30]}")
        return False

    def send(self, line, newline=True):
        data = line.encode() + (b"\r" if newline else b"")
        os.write(self.master, data)

    def drain(self, secs=1.5):
        end = time.time() + secs
        while time.time() < end:
            self._pump(0.2)

    # ---------------- snapshots ----------------
    def snapshot(self, label):
        self.drain(0.4)
        txt = "\n".join(self.screen.display).rstrip()
        self.shots.append((label, txt))
        n = len(self.shots)
        with open(os.path.join(ART, f"shot_{n:02d}_{re.sub(r'[^a-z0-9]+','_',label.lower())[:40]}.txt"), "w") as f:
            f.write(txt)
        # reset the emulated screen so the next snapshot shows fresh content
        try:
            self.screen.reset()
        except Exception:
            pass
        return txt

    def close(self):
        try:
            self.send("/exit")
            time.sleep(1.0)
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def run(fast=False):
    log = []

    def check(name, ok, extra=""):
        log.append((name, bool(ok), extra))
        print(f"  {'✓' if ok else '✗'} {name} {extra}")

    s = Session(["python3", "nexus.py"])
    print("» launching TUI in pty (96x44) …")

    # ---- 1. banner appears
    got = s.wait_for(r"NEXUS|Autonomous Agent", 60)
    check("banner rendered", got)
    got = s.wait_for(re.escape(PROMPT), 40)
    check("first prompt shown", got)
    s.snapshot("banner")

    # ---- 2. /help
    s.send("/help")
    ok = s.wait_for(r"/skills|COMMANDS", 20)
    check("/help lists commands", ok)
    s.snapshot("help")

    # ---- 3. /keys  (key health table)
    s.send("/keys")
    ok = s.wait_for(r"API KEYS|healthy", 20)
    check("/keys shows key health", ok)
    s.snapshot("keys")

    # ---- 4. /status
    s.send("/status")
    ok = s.wait_for(r"USAGE|BY MODEL", 20)
    check("/status usage table", ok)
    s.snapshot("status")

    # ---- 5. /skills
    s.send("/skills web")
    ok = s.wait_for(r"frontend_ui_ux_design|SKILLS", 20)
    check("/skills search works", ok)
    s.snapshot("skills")

    # ---- 6. /tools (least-privilege visibility)
    s.send("/tools")
    ok = s.wait_for(r"TOOLS \(", 20)
    check("/tools lists tools+agents", ok)
    s.snapshot("tools")

    # ---- 7. trivial chat -> router direct (no supervisor)
    s.send("hello! who are you, tell me in 2 lines")
    ok = s.wait_for(r"ROUTE", 90)
    check("router phase shown", ok)
    ok2 = s.wait_for(r"RESULT|intent=chat", 120)
    check("router direct answer (no plan)", ok2 and "PLAN" not in "\n".join(s.screen.display))
    s.snapshot("chat-direct")
    ok3 = s.wait_for(re.escape(PROMPT), 60)
    check("prompt returned after chat", ok3)

    if not fast:
        # ---- 8. REAL autonomous goal: build + verify todo CLI
        goal = ("Build a todo CLI in todo.py with add/list/done subcommands stored in "
                "todos.json, then run it: add 'buy milk', add 'study', mark 'buy milk' done, "
                "list todos, save output to todo_output.txt")
        s.send(goal)
        ok = s.wait_for(r"PLAN", 180)
        check("supervisor PLAN phase", ok)
        s.snapshot("plan")
        ok = s.wait_for(r"VERIFY|verdict|pass", 400)
        check("critic VERIFY phase", ok)
        s.snapshot("verify")
        ok = s.wait_for(r"SYNTHESIZE", 200)
        check("SYNTHESIZE phase", ok)
        ok = s.wait_for(re.escape(PROMPT), 240)
        check("goal finished, prompt back", ok)
        s.snapshot("goal-done")
        text = "\n".join(s.screen.display) + s.buf.decode(errors="ignore")
        check("final stats line", "tasks" in text and "tokens" in text)
        check("todo.py created", os.path.exists(os.path.join(ROOT, "workspace", "todo.py")))
        if os.path.exists(os.path.join(ROOT, "workspace", "todo.py")):
            check("todo_output exists",
                  os.path.exists(os.path.join(ROOT, "workspace", "todo_output.txt")))

        # ---- 9. approval flow: ask to delete the file (should prompt human)
        s.send("delete todo.py from the workspace permanently")
        ok = s.wait_for(r"APPROVAL REQUIRED|Allow\?", 300)
        check("destructive action asked approval", ok)
        if ok:
            time.sleep(0.5)
            s.send("y")  # allow
            s.snapshot("approval")
            ok = s.wait_for(re.escape(PROMPT), 420)
            check("approval flow completes", ok)
            s.snapshot("after-approval")
            check("todo.py actually deleted after 'y'",
                  not os.path.exists(os.path.join(ROOT, "workspace", "todo.py")))

        # ---- 9b. DENY flow: user says NO — file must survive, no workarounds
        if os.path.exists(os.path.join(ROOT, "workspace", "todos.json")):
            s.send("delete todos.json from the workspace permanently")
            ok = s.wait_for(r"APPROVAL REQUIRED|Allow\?", 300)
            check("second delete also asked approval", ok)
            if ok:
                time.sleep(0.5)
                s.send("n")  # DENY
                ok = s.wait_for(re.escape(PROMPT), 420)
                check("deny flow completes", ok)
                s.snapshot("after-deny")
                check("todos.json SURVIVED after 'n' (no workaround)",
                      os.path.exists(os.path.join(ROOT, "workspace", "todos.json")))

    # ---- 10. memory + sessions
    s.send("/memory")
    ok = s.wait_for(r"MEMORY|FACTS|SESSIONS", 20)
    check("/memory shows memory", ok)
    s.snapshot("memory")
    s.send("/sessions")
    ok = s.wait_for(r"SESSIONS", 20)
    check("/sessions lists sessions", ok)
    s.snapshot("sessions")

    s.close()

    # ---- summary
    n_ok = sum(1 for _, okk, _ in log if okk)
    print(f"\n━━ TUI SESSION: {n_ok}/{len(log)} checks passed ━━")
    for name, okk, extra in log:
        if not okk:
            print(f"   FAILED: {name} {extra}")
    with open(os.path.join(ART, "summary.txt"), "w") as f:
        f.write("\n".join(f"{'PASS' if okk else 'FAIL'}  {name}" for name, okk, _ in log))
    return 0 if n_ok == len(log) else 1


if __name__ == "__main__":
    sys.exit(run(fast=len(sys.argv) > 1 and sys.argv[1] == "fast"))
