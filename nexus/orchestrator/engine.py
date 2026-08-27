"""Orchestrator — the autonomous execution loop.

    PLAN -> DAG -> ASSIGN -> RUN (parallel, bounded) -> COLLECT
         -> VERIFY (critic) -> FAILED? retry/replan -> SUCCESS? save memory -> FINAL

State-machine/DAG based (no free-form agent chatter), with budgets.
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..agents.base import AgentOutcome
from ..agents.specialists import (CoderAgent, CriticAgent, MIRROR_RULE,
                                  ResearcherAgent, RouterAgent,
                                  SupervisorAgent, WorkerAgent)
from ..core.envintents import (classify as env_classify, has_reference,
                               is_resource_lookup, is_session_recap,
                               needs_grounding, resolve as env_resolve,
                               wants_observation, wants_reflection)
from .dag import Task, TaskDAG, TaskStatus

import re as _re

# ---- Router safety net ------------------------------------------------
# The router has NO tools. If the request talks about an ACTION (file
# create/delete, writing code, running something), its direct_answer is
# never accepted — it must go through the supervisor.
ACTION_VERB = _re.compile(
    r"\b(delete|remove|rm|erase|wipe|uninstall|create|write|build|generate|edit|"
    r"modify|update|rename|move|copy|fix|refactor|install|deploy|publish|send|"
    r"email|post|upload|download|scrape|crawl|automate|schedule|convert|"
    r"compress|migrate|save|"
    r"store|record|todo|"
    r"search|research|google|lookup|look\s*up|find|fetch)\b", _re.I)
DIRECT_SAFE_INTENTS = {"chat", "question"}
ACTION_CLAIM = _re.compile(
    r"\b(i\s+(?:have|ve|had|just)\s+\w+ed|"
    r"i\s+(?:deleted|removed|created|wrote|built|saved|fixed|made|ran|edited)\b|"
    r"i'?ve\s+(?:deleted|removed|created|wrote|built|saved|fixed|made|ran|edited))", _re.I)
# Device/system questions — the router used to answer "no access" even
# though system_info/termux-api can actually check them.
IMAGE_PATH_RE = _re.compile(
    r"(?i)(/[^\n\"']+\.(?:png|jpe?g|gif|webp|bmp|heic|heif)"
    r"|[\w./\\ -]+\.(?:png|jpe?g|gif|webp|bmp|heic|heif))")
SEE_IMAGE_Q = _re.compile(
    r"(?i)\b(dikh|dekho|dekh|look|see|show|describe|kya.*(image|photo|pic|png)"
    r"|what.*(image|photo|picture|png)|is (image|photo))\b")
DEVICE_Q = _re.compile(
    r"\b(battery|charging|power|storage|disk|space|memory|ram|cpu|temperature|"
    r"overheat|network|wifi|signal|ip\s?address|internet|connect|device|phone|"
    r"screen|brightness|volume|clipboard|location|gps|sensors?|android|termux|"
    r"kernel|uptime|os version)\b", _re.I)
# LIVE info — the model has no real-time data; web_search must be used.
LIVE_Q = _re.compile(
    r"\b(weather|temperature|forecast|news|headlines?|score|match|"
    r"cricket|ipl|price|rate|stock|share|crypto|bitcoin|currency|dollar|rupee|"
    r"latest|current|today'?s|right now|who won|"
    r"release date|schedule|holiday)\b", _re.I)
# One-word greetings — never feed old memory context to the router
# (live bug: "hy" + an old hosting memory => "hosting follow-up" plan and
#  a 20s pipeline that created goal_statement.md. Context is only used
#  when the goal references it or has 3+ words.)
GREETING_RE = _re.compile(
    r"^\s*(h+e+y+|hy+|hi+|hello+|yo+|sup|hola|"
    r"hii+|good\s?(morning|afternoon|evening|night))"
    r"(\s+(bro|bhai|yaar|there|dude|mate))?\s*[!.,?]*\s*$", _re.I)
FOLLOWUP_REF = _re.compile(
    r"\b(it|this|that|continue|"
    r"again|repeat|same|to)\b", _re.I)
# Identity questions — instant deterministic answer (the LLM sometimes leaked ROUTER)
IDENTITY_Q = _re.compile(
    r"(your name|who are you|about yourself|introduce|"
    r"what can you do|how can you help|kaise help|"
    r"kya kar sakte|kya kr skte|"
    r"^help\s*[?!.]*$)", _re.I)
SESSION_Q = _re.compile(
    r"(kitne\s+session|how many session|list session|"
    r"^sessions?\s*[?!.]*$|session (count|list))", _re.I)
CHECK_FOLLOW = _re.compile(
    r"^(check|dekho|dekh|verify|confirm)(\s+to)?(\s+kr[oe]?)?[\s?!.]*$", _re.I)
DROP_THIS = _re.compile(
    r"(chor|chhod|chodo|chhod do|ise delete|is ko delete|"
    r"delete (this|it|ise)|leave this|drop this)", _re.I)
NEXUS_INTRO = (
    "I'm **Nexus** — your personal autonomous agent, running right on this device.\n"
    "- Write, fix, run and verify code\n"
    "- Web research + live info (weather, news, prices)\n"
    "- Build and manage projects\n"
    "- Automation scripts, data analysis\n"
    "- Device checks (battery, storage, network)\n"
    "- Remember things (memory)\n\n"
    "Just tell me what to do — I'll plan it and get it done.")

# Calculator-style math never goes through the LLM — deterministic Python.
MATH_EXPR = _re.compile(r"^[\s\d+\-*/×÷%^().,]+$")
MATH_HAS_OP = _re.compile(r"[\d)]\s*[+\-*/×÷^]\s*[\d(]")
# v1.8.1: tasks that mention these concern hosting/verification — they need the
# full coder (start_server etc.), never the cheap quick-coder path.
# v1.9.8 REWRITTEN: the old pattern matched bare words like "curl"/"localhost"/"deploy",
# so ANY analysis/automation task that mentioned curl was misread as a HOSTING task and
# failed forever on "no verified start_server call" (live: login-flow analysis burned
# 5m46s + 230k tokens on a hosting check it could never satisfy). Real hosting intent =
# an explicit serving tool/command, or host/serve/deploy DIRECTLY next to a deliverable.
# v1.9.9: localhost:PORT alone is NOT hosting intent (live regression: the
# SSRF-analysis goal "Open http://127.0.0.1:7777..." fired the goal-level
# hosting parachute). A bare localhost:PORT now counts only when a serving
# verb exists anywhere in the same text (serve/host/deploy/publish).
_SERVE_VERB = _re.compile(r"\b(?:serve|serving|host|hosting|deploy|deploying|publish|publishing|restart|restarting|relaunch|launch|launching)\b", _re.I)
_QUICK_BLOCK = _re.compile(
    r"(?is)\b(?:"
    r"http\.server|start_server|"
    r"(?:host|hosting|serve|serving|deploy|deploying|publish|publishing|restart|restarting|relaunch|launch|launching)\s+"
    r"(?:the\s+|this\s+|it\s+|my\s+|a\s+)?"
    r"(?:site|website|page|web\s?app|app|server|portfolio|landing|dashboard|frontend|url|it|them)"
    r")\b")


def env_guard(ctx, goal: str, decision: Dict[str, Any], ui=None) -> Dict[str, Any]:
    """§14 anti-hallucination: an environment question may NOT be answered from
    world knowledge. If the router tried a direct answer for a state-dependent
    request, either hand it to the deterministic resolver or force orchestration.

    Returns the (possibly rewritten) decision. Never invents an answer here.
    """
    d = dict(decision or {})
    if d.get("needs_orchestration"):
        return d                      # already going to the supervisor: fine
    direct = str(d.get("direct_answer") or "").strip()
    if not direct:
        return d
    referential = is_resource_lookup(goal)
    if not (needs_grounding(goal) or referential):
        return d
    try:
        from ..core.envintents import resolve
        hit = resolve(goal, {
            "workspace": ctx.config.workspace, "memory": ctx.memory,
            "server_registry": None,
            "exec": lambda name, args: (lambda r: {
                "ok": r.ok, "output": r.output, "error": r.error})(
                ctx.tools.execute(name, args or {}, "solo")),
            "config": ctx.config},
            project_hint=str(ctx.state.get("project_dir")
                             or ctx.state.get("last_project") or ""))
    except Exception:
        hit = None
    if hit and hit.get("answer"):
        # GROUND IT — replace the hallucinated prose with the real observation
        if ui is not None:
            ui.event("warn", ("resource question answered from world knowledge "
                              "→ replaced with a real recursive search")
                     if referential else
                     "environment question answered from world knowledge "
                     "→ replaced with real tool evidence")
        d["direct_answer"] = hit["answer"]
        d["intent"] = "question"
        d["_grounded"] = True
        return d
    if ui is not None:
        ui.event("warn", "environment question needs live state — routing to "
                         "supervisor instead of answering from memory")
    d["needs_orchestration"] = True
    d["direct_answer"] = ""
    d.setdefault("suggested_agents", [])
    if "worker" not in (d.get("suggested_agents") or []):
        d["suggested_agents"] = ["worker"] + list(d.get("suggested_agents") or [])
    return d


def _is_hosting_intent(text: str) -> bool:
    """True when the text carries REAL hosting intent (task/goal level)."""
    if not text:
        return False
    if _QUICK_BLOCK.search(text):
        return True
    bare = _re.search(r"(?:localhost|127\.0\.0\.1):\d+", text)
    return bool(bare and _SERVE_VERB.search(text))
# v1.8.3: explicit hosting spec the supervisor writes into host-task descriptions
_START_SERVER_SPEC = _re.compile(
    r"start_server\(\s*command\s*=\s*['\"]([^'\"]+)['\"][^)]*?port\s*=\s*(\d+)"
    r"[^)]*?marker\s*=\s*['\"]([^'\"]+)['\"]", _re.S)
_HTML_TITLE = _re.compile(r"<title>\s*([^<]{2,80}?)\s*</title>", _re.S | _re.I)




def rel_dir_serves(cmd: str, body: str) -> bool:
    """F4d helper: for heuristic (spec-less) hosting, accept the already-live
    server when its response is an HTML document (a real site, not a 404 or
    an empty dir listing of a wrong folder)."""
    return "<html" in body.lower() or "<!doctype" in body.lower()

def _hosting_mandatory(task) -> bool:
    """v1.9.9 F4: per-task hosting enforcement keys on the TITLE (the short
    deliverable the planner wrote) or an explicit start_server(...) spec in
    the description -- NOT on the whole description. Live bug: the replanner
    embeds goal context ("python3 -m http.server 8090", "restart the site")
    into every task description, so a pure "Create sample stats.json" task
    was judged a hosting task, failed 3x on hosting it never owed, and burned
    ~460k tokens. A task's description may QUOTE hosting; its title says what
    the task IS. Titles of real hosting tasks start with serve/host/restart/
    deploy/launch (observed live: "Restart the site on port 8090", "Start
    server and verify")."""
    try:
        if _is_hosting_intent(task.title or ""):
            return True
        return bool(_START_SERVER_SPEC.search(task.description or ""))
    except Exception:
        return _is_hosting_intent(task.description or "")

def quick_math(goal: str) -> Optional[str]:
    """Solve pure-arithmetic goals like '8282+282282' locally, without the LLM."""
    expr = goal.strip().rstrip("?.!=").strip()
    if not MATH_EXPR.match(expr) or not MATH_HAS_OP.search(expr):
        return None
    expr = (expr.replace("×", "*").replace("÷", "/").replace("^", "**")
                .replace(",", ""))
    try:
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — sirf digits/operators allow hue
    except (ArithmeticError, SyntaxError, TypeError, ValueError):
        return None
    if isinstance(val, (int, float)):
        pretty = f"{val:,}" if isinstance(val, int) else f"{val:,.6f}".rstrip("0").rstrip(".")
        return f"{goal.strip()} = **{pretty}**\n\n(calculated locally, exact — no AI guessing)"
    return None


_MATH_SUB = _re.compile(r"\d[\d,\s]*(?:[.]\d+)?(?:\s*[+\-*/×÷^%]\s*\d[\d,\s]*(?:[.]\d+)?)+")


def precompute_math(goal: str) -> str:
    """Deterministically evaluate arithmetic embedded in a mixed question.

    Live (v1.9.8): 'what is 17*23 and who wrote Python?' went through a full
    2-task DAG because the router is told to never do arithmetic. Here the
    arithmetic is solved locally FIRST and the goal is rewritten with the exact
    result substituted in — the router then only answers the knowledge parts.
    """
    def _sub(m: "_re.Match") -> str:
        src = m.group(0)
        trailing = src[len(src.rstrip()):]          # keep trailing whitespace
        expr = src.strip().replace(",", "").replace("×", "*").replace("÷", "/").replace("^", "**")
        try:
            val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — digits/operators only
        except Exception:
            return src
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        pretty = f"{val:,}" if isinstance(val, int) else f"{val:,.6f}".rstrip("0").rstrip(".")
        return pretty + trailing

    if not MATH_HAS_OP.search(goal):
        return goal
    return _MATH_SUB.sub(_sub, goal)


def looks_like_noise(goal: str) -> bool:
    """1–3 token garbage / typo / unknown token — not an action request.

    Live: 'ilogy' was treated as an action → 4-task research + replan into
    a leftover portfolio site. That must never happen.
    """
    g = (goal or "").strip()
    if not g or len(g) > 40:
        return False
    words = g.split()
    if len(words) > 3:
        return False
    if words[0].lower() in {
        "search", "research", "google", "find", "fetch", "lookup",
        "build", "make", "create", "write", "fix", "run",
    }:
        return False
    if ACTION_VERB.search(g) or DEVICE_Q.search(g) or LIVE_Q.search(g):
        return False
    if MATH_HAS_OP.search(g) or IDENTITY_Q.search(g) or GREETING_RE.match(g):
        return False
    if SESSION_Q.search(g) or CHECK_FOLLOW.match(g) or DROP_THIS.search(g):
        return False
    if "/" in g or g.startswith("-") or any(w.endswith((".py", ".md", ".html")) for w in words):
        return False
    return True


def router_guard(goal: str, decision: Dict[str, Any]) -> tuple:
    """Deterministic harness rule over the router LLM's decision.

    Returns (decision, overridden). Direct answers are allowed ONLY for
    chat/question intents with no action verb in the goal and no action
    claim in the answer text. Everything else is forced to orchestration.
    """
    d = dict(decision or {})
    intent = str(d.get("intent", "unclear")).lower()
    direct = str(d.get("direct_answer") or "").strip()
    # Casual / short / unclear with no action verb: NEVER force a DAG.
    # Do not invent a canned "not sure what X means" — the router LLM replies.
    if IMAGE_PATH_RE.search(goal or "") and SEE_IMAGE_Q.search(goal or ""):
        d["needs_orchestration"] = True
        d["intent"] = "file_ops"
        d["task_type"] = "vision"
        d["model_hint"] = "worker"
        d["direct_answer"] = ""
        return d, True
    if looks_like_noise(goal) and not ACTION_VERB.search(goal):
        d["needs_orchestration"] = False
        d["intent"] = "chat"
        d["complexity"] = "trivial"
        return d, False
    unsafe = (intent not in DIRECT_SAFE_INTENTS
              or bool(ACTION_VERB.search(goal))
              or bool(DEVICE_Q.search(goal))
              or bool(LIVE_Q.search(goal))
              or bool(ACTION_CLAIM.search(direct))
              or bool(MATH_HAS_OP.search(goal)))
    if unsafe and (direct or not d.get("needs_orchestration")):
        d["needs_orchestration"] = True
        d["direct_answer"] = ""
        return d, True
    return d, False


@dataclass
class RunReport:
    goal: str
    task_id: str
    final: str = ""
    ok: bool = False
    plan: Dict[str, Any] = field(default_factory=dict)
    tasks: List[dict] = field(default_factory=list)
    elapsed: float = 0.0
    tokens: int = 0
    replans: int = 0
    verified: bool = False
    stopped_reason: str = ""
    intent: str = ""                    # v1.10.4: env:*/evidence-followup/chat/dag
    evidence: List[Any] = field(default_factory=list)
    llm_calls: int = 0                  # v1.10.4: performance budget accounting
    mode: str = "dag"                   # L0 deterministic | L2 evidence | L4 dag

class Orchestrator:
    def __init__(self, ctx):
        self.ctx = ctx
        self.config = ctx.config
        self.ui = ctx.ui
        self.router = RouterAgent(ctx)
        self.supervisor = SupervisorAgent(ctx)
        self.critic = CriticAgent(ctx)
        self._agent_cache: Dict[str, Any] = {}
        self.cancelled = False
        try:
            self.ctx.state.pop("cancelled", None)
        except Exception:
            pass

        self.max_parallel = int(self.config.get("autonomy.max_parallel_agents", 3))
        self.max_retries = int(self.config.get("autonomy.max_retries", 2))
        self.overall_timeout = float(self.config.get("autonomy.overall_timeout_seconds", 1500))
        self.max_depth = int(self.config.get("autonomy.max_task_depth", 3))
        self._devstral_slots = int(self.config.get("autonomy.max_devstral_parallel", 2))
        # v1.8.3: verified start_server outputs across the run (hosting truth)
        self._server_evidence: List[str] = []

    # ------------------------------------------------------------------
    def agent_for(self, name: str, quick: bool = False):
        key = f"{name}{'_q' if quick else ''}"
        if key not in self._agent_cache:
            if name == "coder":
                self._agent_cache[key] = CoderAgent(self.ctx, quick=quick)
            elif name == "researcher":
                self._agent_cache[key] = ResearcherAgent(self.ctx)
            elif name == "critic":
                self._agent_cache[key] = CriticAgent(self.ctx)
            elif name == "supervisor":
                self._agent_cache[key] = self.supervisor
            else:
                self._agent_cache[key] = WorkerAgent(self.ctx)
        return self._agent_cache[key]

    # ==================================================================
    # v1.10.4 — deterministic environment answers + evidence ledger
    # ==================================================================
    _WRITE_VERBS = _re.compile(
        r"\b(?:build|create|make|write|edit|modify|update|rename|move|copy|delete|remove|"
        r"add|insert|append|rename_to|replace|patch|jod|jodo|badh|banao|banado|likh|daal|daalo|"
        r"fix|refactor|install|deploy|publish|send|email|post|upload|download|convert|"
        r"compress|migrate|save|generate|start|stop|restart|band\s+kar|"
        r"chalao|host|serve|commit|push)\b"
        r"|\b(?:isko|ise|is\s+folder|workspace)\b[^?!.]{0,30}\b(?:clean|delete|remove|band)\b",
        _re.I)

    def _read_only_goal(self, goal: str) -> bool:
        """A goal that only ASKS about state (no mutation implied)."""
        g = (goal or "").strip()
        if not g or len(g) > 160 or "\n" in g:
            return False
        return not bool(self._WRITE_VERBS.search(g))

    def _has_action_verb(self, goal: str) -> bool:
        return bool(ACTION_VERB.search(goal or "")) or bool(self._WRITE_VERBS.search(goal or ""))

    def _ledger(self):
        led = self.ctx.state.get("ledger")
        if led is None:
            from ..core.ledger import EvidenceLedger
            led = EvidenceLedger()
            self.ctx.state["ledger"] = led
        return led

    def _active_project(self) -> str:
        """Project the conversation is currently about (for 'usme/isme' scope)."""
        return str(self.ctx.state.get("project_dir")
                   or self.ctx.state.get("last_project")
                   or self._ledger().current_project() or "")

    def _env_ctx(self) -> Dict[str, Any]:
        def _sig(tool_name: str):
            t = self.ctx.tools.get(tool_name)
            try:
                import inspect
                ps = inspect.signature(t.handler).parameters
                return {k for k, v in ps.items() if v.kind is not v.VAR_KEYWORD}
            except Exception:
                return set()
        return {"workspace": self.config.workspace,
                "memory": self.ctx.memory, "tool_sig": _sig,
                "server_registry": None,
                "exec": lambda name, args: (
                    (lambda r: {"ok": r.ok, "output": r.output, "error": r.error})(
                        self.ctx.tools.execute(name, args or {}, "solo"))),
                "config": self.config}

    def _deterministic_env_answer(self, goal: str, report: RunReport, t0: float,
                                  task_id: str) -> Optional[RunReport]:
        intent = env_classify(goal)
        if not intent:
            return None
        hit = env_resolve(goal, self._env_ctx(),
                          project_hint=self._active_project())
        if not hit or not hit.get("answer"):
            return None
        self.ui.phase("ENV", f"deterministic {intent} — no agent pipeline")
        if hit.get("ok") is False:
            self.ui.event("warn", f"deterministic {intent} could not be verified from runtime state")
        led = self._ledger()
        led.begin_turn(task_id, goal, intent=f"env:{intent}")
        # remember WHICH project the conversation is now about: "usme auth hai?"
        # must resolve to the folder we just listed, not to the whole workspace.
        hit_proj = str((hit.get("evidence") or [{}])[0][1].get("path", "")
                       ) if hit.get("evidence") else ""
        if intent in ("project_tree", "resource_lookup") and hit_proj:
            led.set_project(hit_proj)
        for op, args, ok, out in hit.get("evidence") or []:
            led.record(op, args, ok, out, agent="harness")
        led.close_turn(hit["answer"])
        if self.ctx.memory:
            try:
                self.ctx.memory.add_message("assistant", hit["answer"], "harness")
            except Exception:
                pass
        ok_hit = hit.get("ok", True)
        report.final = hit["answer"]
        report.ok = bool(ok_hit)
        report.verified = bool(ok_hit)   # deterministic tool output IS the evidence
        report.mode = "L0-deterministic" if ok_hit else "L0-failed"
        report.evidence = hit.get("evidence") or []
        report.intent = f"env:{intent}"
        report.elapsed = time.time() - t0
        return report

    def _deterministic_host(self, goal: str, report: RunReport, t0: float,
                            task_id: str) -> Optional[RunReport]:
        """v1.10.5 §19/§20 — 'host THIS existing folder on port N' is ONE tool call,
        yet it was the single most expensive thing in the harness: live TUI runs of
        `portfolio-site ko port 8133 pe serve karo` cost 2m51s / 75,603 tokens (and
        once 6m07s / 148,439) to arrive at exactly what start_server() already does
        atomically — start detached, wait for the port, fetch it, check the content
        marker. The LLM added latency and a chance to get the port wrong, nothing else.

        Deliberately narrow, and it DEGRADES rather than denies: it fires only when
        (a) the goal is real hosting intent, (b) the port is stated, (c) the folder
        exists under the workspace and already has an index file. Anything else —
        a missing entry file, a site that must be built, no port given — returns
        None and the full L4 pipeline runs, so 'build me a portfolio and host it'
        is untouched.
        """
        g = goal or ""
        # v1.10.5 gate: NOT _is_hosting_intent() — that predicate is deliberately
        # strict for the *quick coder* (it wants http.server/localhost:NNNN shapes),
        # so it returned False for the exact Hinglish phrasing users type:
        # 'portfolio-site ko port 8131 pe serve karo'. Here the safety net is
        # different and tighter in the way that matters: we require a serve verb,
        # an EXPLICIT port, and an existing folder that already has an index file —
        # anything that must be built falls through to the agents.
        if not (_SERVE_VERB.search(g)
                and _re.search(r"\b(?:serve|host|deploy|publish|launch)\w*\s+(?:karo|kar|do|de|please|kar\s+do|do\s+na)\b", g, _re.I)):
            return None
        m = _re.search(r"\bport\s*(\d{4,5})\b|\b(\d{4,5})\s+pe\b", g, _re.I)
        if not m:
            return None
        port = int(m.group(1) or m.group(2))
        root = Path(str(self.config.workspace or ".")).resolve()
        try:
            from ..core.ledger import current_project
            scope = str(current_project() or "")
        except Exception:
            scope = ""
        words = [w.strip("/.,:;'\"()").lower() for w in
                 _re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,}", goal or "")]
        cand = ([scope] if scope else []) + [w for w in words
                                             if w not in ("port", "karo", "server", "site",
                                                          "http", "https", "start", "chal")
                                             and not w.isdigit()]
        target = ""
        for name in cand:
            n = name.split("/")[-1] if "/" in name else name
            probes = [root / name, root / "projects" / name]
            try:
                probes += list((root / "projects").glob(f"*{n}*"))
            except Exception:
                pass
            for pr in probes:
                try:
                    if pr.is_dir() and any((pr / f).exists() for f in
                                          ("index.html", "index.htm")):
                        target = str(pr.resolve().relative_to(root))
                        break
                except Exception:
                    continue
            if target:
                break
        if not target:
            return None
        self.ui.phase("HOST", f"deterministic start_server on :{port} — no agent pipeline")
        # 'harness' is not an allow-listed agent for an EXECUTE-risk tool; 'solo'
        # is the operator/one-shot context that start_server permits.
        res = self.ctx.tools.execute("start_server",
                                    {"port": port, "directory": target}, "solo")
        out = (res.output or res.error or "").strip()
        up = bool(res.ok and out)
        ans = (f"Hosted `{target}` at **http://127.0.0.1:{port}** — proved by the "
               f"tool itself (real HTTP fetch), no agent pipeline needed.\n\n"
               f"```\n{out}\n```" if up else
               f"Hosting on :{port} did NOT verify — tool output below, and I have "
               f"not assumed it worked.\n```\n{out}\n```")
        ans = self._flag_script_mismatch(goal, ans)
        led = self._ledger()
        led.begin_turn(task_id, goal, intent="host:deterministic")
        led.record("start_server", {"port": port, "directory": target}, up, out,
                   agent="harness")
        led.close_turn(ans)
        if self.ctx.memory:
            try:
                self.ctx.memory.add_message("assistant", ans, "harness")
            except Exception:
                pass
        report.final = ans
        report.ok = bool(up)
        report.verified = bool(up)          # only the tool's own verdict says so
        report.mode = "L1-host-deterministic" if up else "L1-host-failed"
        report.intent = "host"
        report.elapsed = time.time() - t0
        return report

    def _evidence_answer(self, goal: str, report: RunReport, t0: float,
                         task_id: str) -> Optional[RunReport]:
        """Ground a referential follow-up in what was actually observed."""
        led = self._ledger()
        src = led.last_with_evidence()
        if not src:
            return None
        # a recap question needs EVERY turn, not the recency window
        block = (led.context_block(max_chars=9000, turns=99)
                 if is_session_recap(goal) else led.context_block(turns=3))
        if not block:
            return None
        self.ui.phase("EVIDENCE", "answering from verified observations in this chat")
        # v1.10.5: a SESSION RECAP has no deterministic resolver — it must take the
        # LLM-over-ledger branch. It used to fall into the else-branch below, where
        # env_resolve() returned None, so the recap escaped L2 (0 LLM) and the
        # router re-classified it: 1 wasted call + a shallower answer.
        if (wants_reflection(goal) or is_session_recap(goal)
                or not self._read_only_goal(goal)):
            prompt = (
                f"USER QUESTION: {goal}\n\n{block}\n\n"
                + ("This is a SESSION RECAP: enumerate every turn listed above, in order, "
                   "with what the user asked and what was actually observed. Do not stop at "
                   "the last two turns and do not invent turns that are not listed.\n\n"
                   if is_session_recap(goal) else "")
                + "Answer ONLY from the OBSERVED evidence above. Rules: never invent a "
                "filename, count or meaning you did not observe; if the evidence does not "
                "cover something, say it is not covered; no generic self-introduction; "
                f"{MIRROR_RULE} 3-8 lines, concrete.")
            try:
                text = (self.ctx.llm.ask("router", prompt) or "").strip()
            except Exception:
                text = ""
            if not text:
                facts = "\n".join(f"- {f}" for f in led.fact_digest(8))
                text = ("Main yahi keh sakta hoon jo actually observe hua:\n"
                        + facts + "\n(iske aage ka matlab confirm karne ke liye mujhe "
                        "aur files padhne honge.)")
        else:
            hit = env_resolve(goal, self._env_ctx())
            if not hit:
                return None
            text = hit["answer"]
        led.begin_turn(task_id, goal, intent="evidence-followup")
        for ev in (src.get("evidence") or [])[-6:]:
            led.record(ev.operation, {"path": ev.target}, ev.ok, ev.observed,
                       source=ev.source, agent="ledger-replay")
        led.close_turn(text)
        report.final = self._flag_script_mismatch(goal, text)
        report.ok = True
        report.verified = True
        report.mode = "L2-grounded"
        report.intent = "evidence-followup"
        report.elapsed = time.time() - t0
        return report

    # ==================================================================
    def handle(self, goal: str, force_orchestration: bool = False) -> RunReport:
        """Main entry: route -> (direct answer | full autonomous run)."""
        t0 = time.time()
        task_id = uuid.uuid4().hex[:8]
        self.cancelled = False
        self.ctx.llm.reset_task_budget(task_id)
        report = RunReport(goal=goal, task_id=task_id)

        # ---- safety: input moderation
        allowed, reason = self.ctx.guard.check_text(goal, "input")
        if allowed is False:
            report.final = f"⚠️ {reason}"
            report.stopped_reason = "moderation"
            report.elapsed = time.time() - t0
            return report
        if allowed is None:
            # v1.10.4 BUG-1 FIX: jailbreak / PII signals now ESCALATE to the
            # user instead of silently passing (before this the classifier's
            # category names didn't exist, so nothing escalated at all).
            ok_user = True
            if self.config.get("safety.approval_mode", "smart") != "never":
                try:
                    ok_user = bool(self.ctx.approve("moderation_escalate",
                                                    {"reason": reason}, "system"))
                except Exception:
                    ok_user = True
            if not ok_user:
                report.final = f"⚠️ Not run — {reason}"
                report.stopped_reason = "moderation escalate declined"
                report.elapsed = time.time() - t0
                return report

        # ---- fast path 0: greeting / identity — instant, deterministic,
        #      always in the user's own script, zero LLM calls
        if not force_orchestration:
            if SESSION_Q.search(goal.strip()) or CHECK_FOLLOW.match(goal.strip()):
                n, listing = self._session_listing()
                reply = f"**{n} session(s)**\n{listing}" if listing else f"**{n} session(s)**"
                self.ctx.state["last_meta"] = "sessions"
                self.ui.phase("CHAT", "sessions")
                report = RunReport(goal=goal, task_id=task_id,
                                   final=reply, ok=True, verified=True,
                                   elapsed=time.time() - t0)
                if self.ctx.memory:
                    self.ctx.memory.add_message("assistant", reply, "nexus")
                return report
            if DROP_THIS.search(goal) and len(goal.split()) <= 12:
                reply = self._drop_last_project()
                self.ui.phase("CHAT", "drop last project")
                report = RunReport(goal=goal, task_id=task_id,
                                   final=reply, ok=True, verified=True,
                                   elapsed=time.time() - t0)
                if self.ctx.memory:
                    self.ctx.memory.add_message("assistant", reply, "nexus")
                return report

        # ---- fast path: pure arithmetic solved exactly, without the LLM
        # (live test: the router claimed 8282+282282 = 601144. Never again.)
        if not force_orchestration:
            ans = quick_math(goal)
            if ans is not None:
                self.ui.phase("CALC", "solved locally — exact, no AI guessing")
                report = RunReport(goal=goal, task_id=task_id, final=ans,
                                   ok=True, verified=True, elapsed=time.time() - t0)
                if self.ctx.memory:
                    self.ctx.memory.add_message("assistant", ans, "calculator")
                return report

        # ==================================================================
        # v1.10.4 L0/L1 — DETERMINISTIC ENVIRONMENT PATH (0 LLM calls)
        # A question whose answer is one read-only tool call must never become
        # router → supervisor → worker → critic → synthesis.
        # Live cost of doing it the old way: "abe apne workspace me dekh"
        # = 15.9s + 16,180 tokens for a `list_dir`.
        # ==================================================================
        # ---- fast path: "host this existing folder on port N" — one tool call
        # (before the read-only L0 pass: hosting is a WRITE-ish action, so L0's
        # _read_only_goal gate would never let it through)
        if not force_orchestration:
            try:
                hh = self._deterministic_host(goal, report, t0, task_id)
            except Exception as e:  # noqa: BLE001
                hh = None
                self.ui.event("warn", f"deterministic host path skipped "
                            f"({type(e).__name__}: {str(e)[:60]}) — using agents")
            if hh is not None:
                return hh

        if not force_orchestration and self._read_only_goal(goal):
            # v1.10.4: a fast path must never be able to kill the REPL — a
            # resolver bug degrades to the normal agent route, not a traceback.
            try:
                hit = self._deterministic_env_answer(goal, report, t0, task_id)
            except Exception as e:  # noqa: BLE001
                hit = None
                self.ui.event("warn", f"deterministic env path skipped "
                            f"({type(e).__name__}: {str(e)[:70]}) — using agents")
            if hit is not None:
                return hit

        # ---- L2 grounded follow-up: user refers to what we already observed
        # ("isse tu kya sikha?", "isme kya important hai?"). Resolve the
        # anaphora against the evidence ledger BEFORE the router can mistake it
        # for a fresh identity/chat question (live bug: it answered
        # "Main Nexus, tera personal agent hoon!…" to a follow-up).
        if (not force_orchestration and (has_reference(goal) or is_session_recap(goal))
                and not env_classify(goal)
                and (wants_observation(goal) or is_session_recap(goal))):
            try:
                grounded = self._evidence_answer(goal, report, t0, task_id)
            except Exception as e:  # noqa: BLE001
                grounded = None
                self.ui.event("warn", f"evidence path skipped ({type(e).__name__}) — using agents")
            if grounded is not None:
                return grounded

        # ---- memory context
        mem_ctx = ""
        if self.ctx.memory:
            mem_ctx = self.ctx.memory.build_context(
                goal, int(self.config.get("memory.recent_window", 12)),
                int(self.config.get("memory.semantic_top_k", 5)))
            self.ctx.memory.add_message("user", goal)

        # ---- ROUTE
        self.ui.phase("ROUTE", "classifying request")
        tok0 = self.ctx.llm.stats.snapshot().get("total_tokens", 0)
        self._ledger().begin_turn(task_id, goal, intent="routed")
        # v1.9.8: solve any embedded arithmetic locally, exactly, BEFORE routing
        # (router is forbidden from doing math, so mixed math+trivia questions
        #  no longer need a DAG just because one part is arithmetic)
        route_goal = precompute_math(goal)
        if route_goal != goal:
            self.ui.event("ok", f"arithmetic solved locally: {goal!r} → {route_goal!r}")
        # greeting / very short input => do NOT give the router memory context
        if GREETING_RE.match(goal) or len(goal.split()) < 3:
            rctx = ""
        elif FOLLOWUP_REF.search(goal) and mem_ctx:
            rctx = mem_ctx[-1200:]
        else:
            rctx = ""
        # v1.10.4 §3/§4: a referential follow-up gets the OBSERVATION ledger,
        # not only chat memory — that is what lets "isse kya seekha" resolve.
        if has_reference(goal):
            block = self._ledger().context_block(turns=3)
            if block:
                rctx = (rctx + "\n\n" + block)[-2400:]
        decision = self.router.route(route_goal, rctx)
        # v1.10.4 §1/§2/§6/§14: an environment-grounded question may NEVER be
        # answered from world knowledge. Live bug: "bro workspace me kya hai"
        # was intent=question + orchestrate=False and Nexus recited the
        # Termux app description instead of listing the filesystem.
        decision = env_guard(self.ctx, route_goal, decision, self.ui)
        # v1.9.8: guard against the PRECOMPUTED goal too — the original goal still
        # contains raw arithmetic ('17*23') and MATH_HAS_OP would force a full DAG
        # even though the math is already solved exactly at this point.
        decision, overridden = router_guard(route_goal, decision)
        if overridden:
            self.ui.event("warn", "router override → supervisor "
                          "(action requests cannot be answered without doing them)")
        self.ui.route_info(decision)

        if not force_orchestration and not decision.get("needs_orchestration"):
            answer = str(decision.get("direct_answer") or "").strip()
            if not answer:
                answer = self._live_chat(goal)
            answer = self._flag_script_mismatch(goal, answer)
            report.final, report.ok, report.verified = answer, True, True
            report.elapsed = time.time() - t0
            report.mode = "L2-grounded" if decision.get("_grounded") else "L0-chat"
            report.llm_calls = 1 if answer else 0
            report.tokens = self.ctx.llm.stats.snapshot().get("total_tokens", 0) - tok0
            self._ledger().close_turn(answer)
            if self.ctx.memory:
                self.ctx.memory.add_message("assistant", answer, "router")
            return report

        # ---- PLAN
        self.ui.phase("PLAN", "supervisor building task DAG")
        # Planning uses only user preferences — old task summaries
        # (semantic memory) are NOT injected into plan context, to keep the
        # supervisor from planning stale files (pollution caught in live tests).
        plan_ctx = ""
        if self.ctx.memory:
            prefs = self.ctx.memory.recall("preference", 8)
            if prefs:
                plan_ctx = "### User preferences\n" + "\n".join(
                    f"- {p['key']}: {p['value']}" for p in prefs)
        # Router hint → planner steer: the 8B decider says WHAT class of work this
        # is; the supervisor turns that into tasks with capability-fit models.
        if decision.get("task_type") in ("device", "web", "code") or decision.get("model_hint"):
            plan_ctx += ("\n\n### Router classification (authoritative)\n"
                         f"- task_type: {decision.get('task_type', 'general')}\n"
                         f"- model_hint: {decision.get('model_hint', '')}\n"
                         "- Assign tasks to the agent class that matches model_hint, "
                         "following the MODEL CAPABILITY TABLE below.")
        # v1.9.8 PROJECT MEMORY: existing-project state steers the plan
        existing_ctx = self._existing_project_context(goal)
        if existing_ctx:
            plan_ctx += ("\n\n" + existing_ctx[:2600]
                         + "\nPlan follow-up tasks that EDIT this project â "
                           "never recreate it.")
            self.ui.event("ok", "project memory: continuing existing project "
                                "(context injected into the plan)")
        rt_plan = self._runtime_state_block(goal)
        if rt_plan:
            plan_ctx = (plan_ctx + "\n\n" + rt_plan)[:4600]
        plan = self.supervisor.plan(goal, plan_ctx)
        report.plan = plan
        dag = TaskDAG.from_plan(plan)
        if dag.dangling():
            self.ui.event("warn", "dangling DAG deps — rebuilding fallback plan")
            plan = self.supervisor._fallback_plan(goal, "dangling dependencies")
            report.plan = plan
            dag = TaskDAG.from_plan(plan)
        self._apply_project_scope(goal, plan, dag)
        self._reinforce_assignment(dag)     # v1.6: harness-level capability fix
        self.ui.show_plan(plan, dag)

        # ---- EXECUTE loop with replanning
        max_replans = self.max_retries
        while True:
            self._execute_dag(dag, task_id, t0)
            failed = dag.failed()
            if time.time() - t0 > self.overall_timeout:
                report.stopped_reason = "overall timeout"
                break
            if not failed or report.replans >= max_replans or self.cancelled:
                break
            report.replans += 1
            note = "\n".join(f"- {t.title}: {t.error or t.verdict}" for t in failed)
            self.ui.phase("REPLAN", f"{len(failed)} task(s) failed — attempt {report.replans}")
            # NEVER inject old session/RAG memory here — live: 'ilogy' replan
            # became a portfolio website from leftover workspace docs.
            if looks_like_noise(goal):
                self.ui.event("warn", "short/unclear goal — stop instead of inventing a new project")
                break
            # v1.10.2 BUG-W2: the replan LLM call sits BETWEEN task-boundary
            # deadline checks — with a rate-limited key it retried/backoff-slept
            # forever and the run hung far past overall_timeout (live W1 x2).
            if t0 and time.time() - t0 > self.overall_timeout:
                self.ui.event("warn", "overall timeout reached — not replanning further")
                break
            replan_ctx = (
                f"USER GOAL (do not change, do not invent a different project): {goal}\n"
                "If this goal is a single unknown word, ask the user — do NOT build a website."
            )
            plan = self.supervisor.plan(goal, replan_ctx, failure_note=note)
            report.plan = plan
            new_dag = TaskDAG.from_plan(plan)
            # carry over successful work as context
            done_ctx = "\n".join(f"[{t.id}] {t.title}: {t.output[:400]}" for t in dag.done())
            for t in new_dag.tasks.values():
                if done_ctx:
                    t.description += f"\n\nAlready completed earlier:\n{done_ctx[:1500]}"
            dag = new_dag
            self.ui.show_plan(plan, dag)

        # v1.8.7: GOAL-LEVEL parachute — if the USER GOAL asked to host and
        # nothing verified (timeout killed t4, or no host-task ran), the
        # harness still hosts the newest index.html. Live parity: 900s cap
        # cut t3, t4 never started, user was told to run http.server themselves.
        if self._goal_needs_host(goal, dag) and not self._server_evidence:
            host_task = next((t for t in reversed(dag.order())
                              if _is_hosting_intent(f"{t.title} {t.description}")),
                             None)
            if host_task is None and dag.order():
                host_task = dag.order()[-1]
            if host_task is not None:
                self.ui.event("warn", "goal-level hosting parachute (DAG ended unhosted)")
                if self._host_parachute(host_task):
                    host_task.status = TaskStatus.DONE
                    host_task.verdict = "pass"
                    host_task.score = max(host_task.score, 100.0)

        # ---- SYNTHESIZE
        results = [t.to_dict() for t in dag.order()]
        report.tasks = results
        # v1.10.4 §13 SYNTHESIS POLICY: a single small deterministic result does
        # not need a medium-2508 synthesis round (that call alone cost more than
        # the whole task that produced it). Pass the tool output through verbatim
        # when it is short, factual, and nothing else needs combining.
        done = [t for t in dag.order() if t.status is TaskStatus.DONE]
        cheap = (len(dag.order()) == 1 and len(done) == 1 and not dag.failed()
                 and len(done[0].output or "") <= 2600
                 and not self._goal_needs_host(goal, dag)
                 and not any(k in (done[0].output or "").lower()
                             for k in ("error", "blocked", "not found")))
        if cheap:
            self.ui.phase("SYNTHESIZE", "single deterministic result — evidence "
                                        "passed through (no LLM round)")
            report.mode = "L1-agent-raw"
            report.final = (done[0].output or "").strip()
            # verified/ok are still decided below from the real verdict+score — the
            # cheap path skips the SYNTHESIS LLM round, never the honesty check.
            final = report.final
        else:
            # ---- SYNTHESIZE (expensive path)
            self.ui.phase("SYNTHESIZE", "supervisor combining results")
            # v1.8.3: hosting truth is FACT, not opinion — the synthesizer gets
            # the verified evidence (or the explicit warning) so it can never
            # fabricate "HTTP 200 / live at / marker found" (a live run did that).
            facts_parts: List[str] = []
            wf = self._workspace_facts()
            if wf:
                facts_parts.append(wf)
            if self._goal_needs_host(goal, dag):
                if self._server_evidence:
                    facts_parts.append(
                        "VERIFIED HOSTING EVIDENCE (real start_server tool output, "
                        "quote it as proof): " + self._server_evidence[-1])
                else:
                    facts_parts.append(
                        "HOSTING REALITY: NO verified hosting happened in this run — "
                        "no start_server tool output shows HTTP 200 + marker. Your "
                        "final answer MUST say hosting was NOT verified. NEVER write "
                        "'HTTP 200', 'live at', 'marker found' or 'hosted' unless the "
                        "RESULTS contain that actual evidence. NEVER tell the user to "
                        "run python -m http.server / npm start / flask themselves — "
                        "that is a hosting-guide, which is FORBIDDEN.")
            facts_parts.append(MIRROR_RULE.strip() + " The user's request above is the "
                               "language contract: if it is Roman Hinglish, answer in Roman "
                               "Hinglish — never Devanagari, never English.")
            facts = "\n\n".join(facts_parts)
            try:
                final = self.supervisor.synthesize(goal, results, plan, facts=facts)
            except Exception as e:  # noqa: BLE001
                final = self._manual_summary(dag, str(e))
            report.final = self._sanitize_final(final, self._goal_needs_host(goal, dag))

        # ---- final safety + memory
        ok_out, reason = self.ctx.guard.check_text(final, "output")
        if ok_out is False:
            final = f"⚠️ Output withheld: {reason}"
            report.final = final
        report.mode = report.mode or "L4-dag"
        self._ledger().close_turn(report.final or "")
        done_n = len(dag.done())
        report.ok = (done_n > 0 and not dag.failed()
                     and all((t.verdict or "pass") == "pass" for t in dag.done()))
        report.verified = (bool(dag.done()) and not dag.failed()
                           and all((t.verdict or "pass") == "pass" and t.score >= 60
                                   for t in dag.done()))
        # hosting-required run with zero real start_server evidence is never 'verified'
        if self._goal_needs_host(goal, dag) and not self._server_evidence:
            report.verified = False
            report.stopped_reason = "; ".join(
                x for x in [report.stopped_reason or "", "hosting not verified"] if x)
        elif self._goal_needs_host(goal, dag) and self._server_evidence:
            ev = " \n".join(self._server_evidence).lower()
            slug = ""
            try:
                slug = str(self.ctx.state.get("project_dir") or "").rsplit("/", 1)[-1].lower()
            except Exception:
                slug = ""
            # v1.10.4 BUG-2 FIX (part 2): accept when the evidence names this
            # project by slug, by its workspace-relative path, or by an on-disk
            # match of the served dir — substring-on-slug alone was too brittle.
            served_dir = str(self.ctx.state.get("project_dir") or "").lower()
            matches = bool(slug and (
                slug in ev or slug.replace("-", " ") in ev
                or (served_dir and served_dir in ev)
                or f"serving: {served_dir}" in ev))
            if not matches and served_dir:
                # last resort: the running registry really points at this folder
                try:
                    reg = self.config.workspace / ".nexus" / "servers.json"
                    if reg.exists():
                        import json as _json
                        for meta in (_json.loads(reg.read_text() or "{}")).values():
                            if served_dir in str(meta).lower():
                                matches = True
                                break
                except Exception:
                    pass
            # stale server on :8000 serving another project is NOT this goal
            if slug and not matches:
                report.verified = False
                report.stopped_reason = "; ".join(
                    x for x in [report.stopped_reason or "",
                                "hosting evidence does not match this project"] if x)
            else:
                report.verified = True
        report.elapsed = time.time() - t0
        report.tokens = sum(t.tokens for t in dag.tasks.values())

        if self.ctx.memory:
            self.ctx.memory.add_message("assistant", final, "supervisor")
            if self.config.get("memory.save_task_summaries", True):
                self.ctx.memory.save_task(task_id, goal[:120], "supervisor",
                                          "done" if report.ok else "partial",
                                          final, 100.0 if report.ok else 50.0,
                                          {"tasks": len(dag), "replans": report.replans})
        self._clear_project_scope()
        return report


    # ==================================================================
    def _execute_dag(self, dag: TaskDAG, task_id: str, t0: float) -> None:
        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            while not dag.all_settled() and not self.cancelled:
                if time.time() - t0 > self.overall_timeout:
                    for t in dag.tasks.values():
                        if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                            t.status = TaskStatus.FAILED
                            t.error = "overall timeout"
                    break
                batch = dag.ready(self.max_parallel)
                if not batch:
                    if dag.pending_count() == 0:
                        break
                    time.sleep(0.4)
                    continue

                futures = {}
                for t in batch:
                    t.status = TaskStatus.RUNNING
                    t.started = time.time()
                    self.ui.task_start(t)
                    futures[pool.submit(self._run_task, t, dag, task_id, t0)] = t

                for fut in as_completed(futures):
                    t = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:  # noqa: BLE001
                        t.status = TaskStatus.FAILED
                        t.error = str(e)[:300]
                    t.finished = time.time()
                    self.ui.task_end(t)
                    self._checkpoint(dag, task_id, "")
                    # v1.8: enforce the overall deadline BETWEEN futures too —
                    # a long task must not let the run sail past the cap
                    if time.time() - t0 > self.overall_timeout:
                        for tt in dag.tasks.values():
                            if tt.status is TaskStatus.PENDING:
                                tt.status = TaskStatus.FAILED
                                tt.error = "overall timeout"
                        break

    # ------------------------------------------------------------------
    @staticmethod
    def _goal_needs_host(goal: str, dag: Optional[TaskDAG] = None) -> bool:
        """v1.8.7: hosting is required if the USER GOAL says so, not only if
        some task description happened to mention http (timeout can skip t4)."""
        if _is_hosting_intent(goal or ""):
            return True
        if dag is not None:
            return any(_is_hosting_intent(f"{t.title} {t.description}")
                       for t in dag.tasks.values())
        return False

    def _workspace_facts(self) -> str:
        """Authoritative file list so the synthesizer cannot deny existing files
        (live: claimed test_contact.py was never created — it was)."""
        try:
            ws = Path(self.config.workspace)
            if not ws.exists():
                return ""
            keep = {".html", ".css", ".js", ".py", ".md", ".json", ".txt"}
            skip = {".nexus", "__pycache__", ".git", "node_modules"}
            files: List[str] = []
            for p in sorted(ws.rglob("*")):
                if not p.is_file():
                    continue
                if any(part in skip for part in p.parts):
                    continue
                if p.suffix.lower() not in keep:
                    continue
                files.append(str(p.relative_to(ws)))
                if len(files) >= 48:
                    break
            if not files:
                return ""
            return ("WORKSPACE FILES THAT EXIST (authoritative — NEVER say these "
                    "are missing):\n" + "\n".join(f"- {f}" for f in files))
        except Exception:
            return ""

    _DIY_SERVER = _re.compile(
        r"(?im)^[ \t]*([•*\-]\s*)?(to host:.*|python3?\s+-m\s+http\.server[^\n]*|"
        r"then visit http://localhost[^\n]*|"
        r"run (the )?(start_server|server) command[^\n]*|"
        r"you must run the server[^\n]*)\n?")

    def _sanitize_final(self, text: str, hosting_required: bool) -> str:
        """Strip DIY hosting-guides even if the LLM ignores FACTS."""
        out = text or ""
        if hosting_required and not self._server_evidence:
            out = self._DIY_SERVER.sub("", out)
            if not _re.search(r"not verified", out, _re.I):
                out = (out.rstrip() +
                       "\n\nHosting was NOT verified in this run "
                       "(no start_server evidence).")
        return out

    def _host_parachute(self, task: Task) -> bool:
        """v1.8.3: HARNESS-EXECUTED hosting. If a coder task that must host ends
        without a verified start_server call, the engine starts the server itself:
        explicit spec (start_server(command=..., port=..., marker=...)) if the
        plan wrote one, else the newest projects/*/index.html with its <title> as
        the marker. The agent can simply never fail to host when files exist."""
        try:
            text = f"{task.title}\n{task.description}\n{task.acceptance or ''}"
            m = _START_SERVER_SPEC.search(text)
            if m:
                cmd, port, marker = m.group(1), int(m.group(2)), m.group(3)
                # v1.8.5: a spec pointing at a MISSING dir can never serve (live
                # run #4: --directory projects/varanasi-hub didn't exist -> 404).
                dm = _re.search(r"--directory\s+(\S+)", cmd)
                if dm and not (Path(self.ctx.config.workspace) / dm.group(1)).exists():
                    m = None
            if m is None:                       # invalid/absent spec -> heuristic
                ws = Path(self.ctx.config.workspace)
                pdir = ""
                try:
                    pdir = str(self.ctx.state.get("project_dir") or "")
                except Exception:
                    pdir = ""
                scoped = (ws / pdir) if pdir else (ws / "projects")
                if not scoped.exists():
                    scoped = ws / "projects"
                if not scoped.exists():
                    return False
                cands = sorted(scoped.rglob("index.html"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
                if not cands:
                    return False
                idx = cands[0]
                rel = idx.parent.relative_to(Path(self.ctx.config.workspace))
                title_m = _HTML_TITLE.search(
                    idx.read_text(encoding="utf-8", errors="replace"))
                marker = (title_m.group(1).strip() if title_m else "index")[:80]
                # v1.9.9 F4c: the goal usually names its port ("on port 8090",
                # ":8090") -- use it instead of a hardcoded 8000 (live: goal
                # said 8090, parachute tried 8000, which a stale server held).
                port = 8000
                pm = _re.search(r"port\s*[=: ]\s*(\d{4,5})\b|:(\d{4,5})\b", text)
                if pm:
                    cand = int(pm.group(1) or pm.group(2))
                    if 1024 <= cand <= 65535:
                        port = cand
                cmd = f"python3 -m http.server {port} --directory {rel}"
            # v1.9.9 F4d: a harness-tracked server ALREADY serving this port
            # with the marker present in its body counts as hosted (live:
            # "restart the site on 8090" while 8090 already served that site ->
            # parachute died on ALREADY IN USE instead of accepting reality).
            try:
                # v1.10.3 F9: if ANYTHING on the target port already serves the
                # expected marker, hosting is DONE — accept it. (Registry-tracked
                # OR not: live W3, the coder's process served the right site on
                # 8093 but start_server refused ALREADY IN USE and the parachute
                # burned 3 retries for a fact a single GET could prove.)
                import urllib.request as _ur
                with _ur.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as _resp:
                    _body = _resp.read().decode("utf-8", "replace")
                if (not marker or marker in _body) and (
                        "<html" in _body.lower() or "<!doctype" in _body.lower()):
                    out = (f"[already-hosted] port {port} verified live, marker "
                           f"{marker!r} present in response.")
                    self._server_evidence.append(out)
                    self.ui.event("ok", f"{task.id}: hosting already live on {port}")
                    task.output = (task.output or "") + "\n\n" + out
                    return True
            except Exception:
                pass  # nothing live on the port -> real start_server attempt below
            r = self.ctx.shell.start_server(command=cmd, port=port, marker=marker,
                                            name="nexus-parachute")
            if not r.ok and "marker" in (r.error or "") and not m:
                # title guessed wrong? retry with a marker-free verification
                r = self.ctx.shell.start_server(command=cmd, port=port, marker="",
                                                name="nexus-parachute")
            if not r.ok:
                self.ui.event("warn", f"{task.id}: harness hosting failed: {(r.error or '')[:120]}")
                return False
            out = str(r.output or r.error or "")
            self._server_evidence.append(out[:400])
            self.ui.event("ok", f"{task.id}: harness hosted + verified ({port})")
            task.output = (task.output or "") + "\n\n[HARNESS-EXECUTED HOSTING]\n" + out[:800]
            return True
        except Exception as e:  # noqa: BLE001
            self.ui.event("warn", f"{task.id}: harness hosting error: {e}")
            return False

    # ------------------------------------------------------------------
    def _deadline_or_cancel(self, t0: float) -> bool:
        """v1.8.1: cancelled once the overall cap is crossed — checked on every
        agent STEP so a single long task can't sail past the deadline."""
        if t0 and time.time() - t0 > self.overall_timeout:
            self.cancelled = True
            return True
        return False

    _DEVANAGARI = _re.compile(r"[\u0900-\u097F]")

    def _flag_script_mismatch(self, goal: str, text: str) -> str:
        """v1.10.5: prompt instructions cannot fully stop a small model from
        answering Roman Hinglish in Devanagari. Rather than silently shipping a
        script the user did not ask for, detect it deterministically and say so —
        a visible, honest flag beats a silent mismatch (and beats a second LLM
        call to re-ask)."""
        try:
            g, t = str(goal or ""), str(text or "")
            if not t or self._DEVANAGARI.search(g):        # user already Devanagari
                return t
            if not self._DEVANAGARI.search(t):             # matched scripts
                return t
            latin = sum(1 for c in t if c.isascii() and c.isalpha())
            deva = len(self._DEVANAGARI.findall(t))
            if deva < 8 or latin > deva * 3:               # stray quotes/paths only
                return t
            return (t + "\n\n(script note: answer aaya Devanagari mein, tumhara "
                    "sawal Roman/Hinglish tha — chat model ne script badal di. Bolo to "
                    "Roman mein repeat kar doon.)")
        except Exception:
            return text

    def _runtime_state_block(self, goal: str = "") -> str:
        """LIVE runtime facts for agent prompts (v1.10.4).

        Plan context deliberately excludes old task summaries (semantic memory
        polluted plans), but that also excluded *runtime* state — so a follow-up
        like 'server band kar do' had to guess. Live TUI: the worker stopped the
        orphan :8131 left by an EARLIER session and reported success while the
        user's actual :8132 server kept serving. Wrong-target is not a style
        problem. These lines come from what a real tool printed in this
        conversation, so the agent can only act on ports it can see.
        """
        try:
            led = self._ledger()
            facts: List[str] = []
            for t in list(getattr(led, "_turns", []) or []):
                for ev in (t.get("evidence") or []):
                    op = str(getattr(ev, "operation", "") or "")
                    if not _re.search(r"server|port", op, _re.I):
                        continue
                    body = str(getattr(ev, "observed", "") or "")
                    for line in body.splitlines():
                        ln = line.strip()
                        if (ln.startswith("serving:") or ln.startswith("stopped:")
                                or _re.search(r":\d{2,5}\s+(UP|DOWN|FREE|LISTENING)", ln, _re.I)):
                            facts.append(f"{op} → {ln}"[:200])
            proj = ""
            try:
                from ..core.ledger import current_project
                proj = current_project() or ""
            except Exception:
                proj = ""
            if proj:
                facts.append(f"current project scope: {proj}")
            if not facts:
                return ""
            uniq = list(dict.fromkeys(facts))[-8:]
            block = ("### RUNTIME STATE (printed by the real tools in THIS conversation, "
                     "not recalled from memory)\n" + "\n".join(f"- {f}" for f in uniq))
            if _re.search(r"server|port\b|band|start|restart|host", goal or "", _re.I):
                block += ("\nRULE for server operations: act only on ports shown ABOVE — never "
                          "stop/start a port that is not listed there, never re-report a port from "
                          "an earlier session, and never claim an action without calling the tool "
                          "that proves it. If the user says 'the server' and exactly one port is "
                          "UP, that is the one. Quote the tool output line as your evidence.")
            return block
        except Exception:
            return ""

    def _run_task(self, task: Task, dag: TaskDAG, task_id: str, t0: float = None) -> None:
        agent_name = task.agent
        task_budget = float(self.config.get("autonomy.task_timeout_seconds", 180)) * \
            (1 + self.max_retries * 0.6)          # retries get a shrinking allowance
        t_task = time.time()
        context = self._dep_context(task, dag)
        rt = self._runtime_state_block()
        if rt:
            context += "\n\n" + rt
        # v1.9.8 PROJECT MEMORY: builders see the existing project state
        existing_ctx = str(self.ctx.state.get("existing_project_ctx") or "")
        if existing_ctx and task.agent in ("coder", "worker"):
            context += ("\n\n" + existing_ctx[:2000]
                        + "\nYou are EXTENDING this existing project â read its files "
                          "first (list_dir + read_file), then edit in place.")
        if task.skill:
            context += (f"\n\nRECOMMENDED SKILL: `{task.skill}` — call "
                        f"load_skill('{task.skill}') first and follow it.")
        if task.acceptance:
            context += f"\n\nACCEPTANCE CRITERION (must be met): {task.acceptance}"

        for attempt in range(self.max_retries + 1):
            if self.cancelled:
                task.status = TaskStatus.SKIPPED
                return
            # v1.10.4 BUG-3 FIX: this guard used to sit *after* the `return`
            # inside the cancelled-branch — i.e. it could never run, so a task
            # with max_retries=2 could burn 3 full step budgets with no
            # per-task cap (live: 10m43s / 112k tokens on one goal).
            if attempt > 0 and time.time() - t_task > task_budget:
                self.ui.event("warn", f"{task.id}: time budget spent — not marking done")
                task.status = TaskStatus.FAILED
                task.error = task.error or "task time budget exceeded"
                task.score = max(task.score, 50.0)
                task.verdict = task.verdict or "fail"
                return
            task.attempts = attempt + 1
            # v1.8.1: the quick (cheap) coder must never handle hosting/verification
            # — live TUI run: host task (short desc) went to codestral-2508 which
            # returned an EMPTY response (0 tool calls), burning an attempt+critic round.
            quick = (agent_name == "coder" and attempt == 0
                     and len(task.description) < 400
                     and not _is_hosting_intent(task.description))
            agent = self.agent_for(agent_name, quick=quick)

            # v1.8.3: a planned model that stayed silent gets EXCLUDED on retries —
            # attempt 0 may use task.model, retries always use the role chain
            # (live: codestral-2508 returned zero tool calls on 3 attempts; only
            #  devstral-2512 in the chain actually works on this account)
            outcome: AgentOutcome = agent.run(
                f"{task.title}\n\n{task.description}", context,
                on_step=lambda s, tt=task: (self._deadline_or_cancel(t0) or
                                            self.ui.task_step(tt, s)), task_id=task_id,
                model=(task.model if attempt == 0 else None))

            # v1.8.1: hard deadline — no critic round after the cap is gone
            # (live: the last task of the last plan ran 325s past the 900s cap
            #  because only the NEXT future would have triggered the check)
            if t0 and time.time() - t0 > self.overall_timeout:
                task.status = TaskStatus.FAILED
                task.error = "overall timeout"
                return
            task.steps += len(outcome.steps)
            task.tokens += outcome.tokens
            task.output = outcome.output or task.output

            # v1.10.4 §12: classify this outcome so the critic can be skipped
            # for clean read-only work (and never for anything that mutated).
            _WRITE_TOOLS = {"write_file", "edit_file", "delete_path", "move_path",
                            "run_shell", "run_python", "install_package", "start_server",
                            "stop_server", "git_add", "git_commit", "sqlite_exec",
                            "make_pptx", "make_pdf", "make_docx", "remember",
                            "index_knowledge", "browser_fill", "browser_click"}
            tool_steps = [s for s in outcome.steps if s.kind == "tool"]
            task.touched = any(s.tool in _WRITE_TOOLS for s in tool_steps)
            task.clean_reads = (bool(tool_steps) and not task.touched
                                and all(s.ok for s in tool_steps))

            # v1.8.1: deterministic insurance — a coding task that made ZERO tool
            # calls (live: host task answered empty / 'I cannot use tools') is failed
            # fast WITHOUT a 30-60s critic round; retry immediately with the fix note.
            if agent_name == "coder" and not any(s.kind == "tool" for s in outcome.steps):
                # v1.8.3: hosting tasks go to the parachute even when the model
                # returned nothing at all (live: codestral-2508 answered with zero
                # tool calls on 3 attempts; the host never happened)
                if (_hosting_mandatory(task) and self._host_parachute(task)):
                    task.status = TaskStatus.DONE
                    task.verdict = "pass"
                    task.score = 100.0
                    return
                if attempt < self.max_retries:
                    self.ui.event("retry", f"{task.id} made no tool calls — retry {attempt + 2}")
                    context += ("\n\nPREVIOUS ATTEMPT MADE NO TOOL CALLS. You MUST actually call "
                                "tools (list_dir/read_file/run_shell/start_server...) — a text-only "
                                "answer to a coding/hosting task is always WRONG.")
                    continue
                task.status = TaskStatus.FAILED
                task.error = "agent produced no tool calls (text-only or empty response)"
                return

            # record real start_server successes (hosting truth for the final answer)
            for s in outcome.steps:
                if s.kind == "tool" and s.tool == "start_server" and s.ok:
                    self._server_evidence.append(str(s.content)[:400])

            # v1.8.3: HARD HOSTING REQUIREMENT — a coder task that involves hosting
            # (host/server/verify/http/marker...) must show a VERIFIED start_server
            # call, else the harness executes the hosting itself (parachute).
            # (live: t4 took 4 unrelated steps, critic said 'partial 70' and the
            #  score>=70 shortcut marked it DONE with NO server at all.)
            if (agent_name == "coder" and _hosting_mandatory(task)
                    and not any(s.kind == "tool" and s.ok and s.tool == "start_server"
                                for s in outcome.steps)):
                self.ui.event("warn", f"{task.id}: hosting not executed — harness takes over")
                if self._host_parachute(task):
                    task.status = TaskStatus.DONE
                    task.verdict = "pass"
                    task.score = 100.0
                    return
                if attempt < self.max_retries:
                    context += ("\n\nPREVIOUS ATTEMPT DID NOT HOST. Hosting is MANDATORY: call "
                                "start_server(command='python3 -m http.server <port> --directory "
                                "projects/<slug>', port=<port>, marker='<exact title text>') and "
                                "report its verified output — or the task FAILS.")
                    continue
                task.status = TaskStatus.FAILED
                task.error = "hosting not executed — no verified start_server call"
                return

            if not outcome.ok and not outcome.output:
                task.error = outcome.error or "agent produced no output"
                if attempt < self.max_retries:
                    self.ui.event("retry", f"{task.id} failed ({task.error[:60]}) — retry {attempt + 2}")
                    context += f"\n\nPREVIOUS ATTEMPT FAILED: {task.error}. Try a different approach."
                    continue
                task.status = TaskStatus.FAILED
                return

            # ---- VERIFY
            if self._should_verify(task):
                self.ui.phase("VERIFY", f"critic checking {task.id}", quiet=True)
                failed_ops = [f"{s.tool}: {s.content[:120]}" for s in outcome.steps
                              if s.kind == "tool" and not s.ok]
                verdict = self.critic.verify(task.title, task.acceptance or "Task completed correctly",
                                             task.output, task_id=task_id,
                                             tool_failures=failed_ops)
                # Deterministic insurance: a task whose tool calls ERRORS can never be
                # certified 100-pass out of the box — the critic must justify it.
                # (Live bug: storage task with 5 failed `du` runs still scored 100.0)
                # v1.9.8: BUT a conscious 'pass' with NO issues and NO fix note means
                # the critic re-verified the affected data itself — accept it. This
                # stops the old behaviour of infinitely downgrading sandbox-blocked
                # but otherwise complete tasks and burning retries on nothing.
                if failed_ops and verdict.get("verdict") == "pass":
                    justified = (not (verdict.get("issues") or [])
                                 and not str(verdict.get("fix_instructions") or "").strip())
                    if not justified:
                        verdict["verdict"] = "partial"
                        verdict["score"] = min(float(verdict.get("score", 70)), 79.0)
                        verdict.setdefault("issues", []).insert(
                            0, f"{len(failed_ops)} tool call(s) failed during the task; "
                               f"verify the data is complete despite them: {failed_ops[0][:80]}")
                task.score = float(verdict.get("score", 0))
                task.verdict = verdict.get("verdict", "")
                self.ui.verdict(task, verdict)

                # v1.8.3: ONLY a real "pass" verdict completes immediately. A
                # "partial" (even score>=70) retries with the critic's fix note —
                # live run: 'partial 70' on a hosting task that never hosted became
                # DONE through the old score>=70 shortcut.
                if verdict.get("verdict") == "pass":
                    task.status = TaskStatus.DONE
                    return
                if verdict.get("verdict") == "partial" and task.score >= 85 and attempt >= self.max_retries:
                    # v1.10.1 F5: borderline-accept now requires >=85. The critic
                    # caps partial scores at 79, so a partial at retries-exhausted
                    # falls through to an honest FAILED below. Live bug (W1): a
                    # verify task that never produced its required screenshots
                    # was accepted as DONE at partial 70 — the acceptance was
                    # unmet but the run reported the site "fully implemented".
                    task.status = TaskStatus.DONE
                    task.verdict = "partial"
                    return
                # a retry only helps if the critic said WHAT to fix
                fix = (verdict.get("fix_instructions") or
                       "; ".join(str(i) for i in verdict.get("issues", []) if i))
                actionable = bool(fix.strip()) and "not parseable" not in fix
                if attempt < self.max_retries and actionable:
                    # v1.9.8 ANTI-LOOP: if the failures were sandbox/environment
                    # blocks, hammering the same path again can never succeed —
                    # order a full approach change instead of a blind retry.
                    blocked = any(_re.search(
                        r"outside (the )?workspace|sandbox|BLOCKED|permission",
                        f, _re.I) for f in failed_ops)
                    if blocked:
                        context += (
                            f"\n\nCRITIC REJECTED ATTEMPT {attempt + 1}. Fix these: {fix}\n"
                            f"CRITICAL APPROACH CHANGE REQUIRED: your previous attempt(s) hit "
                            f"SANDBOX BLOCKS. You can NEVER read/write/delete anything outside "
                            f"the workspace directory ({self.config.workspace}). STOP retrying "
                            f"those paths. Redo the work with workspace-RELATIVE paths only "
                            f"(e.g. data.json or projects/<slug>/data.json INSIDE the workspace)."
                        )
                    else:
                        context += (f"\n\nCRITIC REJECTED THE PREVIOUS ATTEMPT. Fix these: {fix}\n"
                                    "IMPORTANT: do NOT repeat the same steps — change what you "
                                    "did so the fix actually lands.")
                    self.ui.event("retry", f"{task.id} rejected by critic — retry {attempt + 2}")
                    continue
                if not actionable:
                    self.ui.event("warn", f"{task.id}: critic gave no actionable feedback — accepting")
                    task.status = TaskStatus.DONE if task.output else TaskStatus.FAILED
                    task.score = max(task.score, 55.0)
                    return
                # last resort: hard verification with the large model
                hard = self.critic.hard_verify(task.title, task.acceptance, task.output, task_id)
                task.score = float(hard.get("score", task.score))
                task.verdict = hard.get("verdict", task.verdict)
                # DONE only when the hard model also says pass/partial (≥60).
                # Live bug #5: t1 was still marked "done" after 3 critic failures.
                hard_v = hard.get("verdict")
                if hard_v == "pass" or (hard_v == "partial" and task.score >= 60):
                    task.status = TaskStatus.DONE
                else:
                    task.status = TaskStatus.FAILED
                    task.error = "; ".join(hard.get("issues", []))[:300] or \
                        f"critic score {task.score:.0f} — task genuinely incomplete"
                return
            else:
                task.status = TaskStatus.DONE
                task.score = 75.0
                return
        task.status = TaskStatus.FAILED

    # ------------------------------------------------------------------
    def _should_verify(self, task: Task) -> bool:
        """§12 CRITIC POLICY. The critic earns its 3-4s + ~5k tokens on work
        that can actually be wrong in ways a tool cannot detect. It adds nothing
        to a deterministic read-only query whose own success IS the evidence.

        Skip when ALL of:
          • the task is not code/research/hosting (those always verify)
          • it made at least one tool call and NONE of its calls failed
          • there is no explicit acceptance criterion the critic must check
        """
        if self.config.get("autonomy.verify_all", True) is False:
            return False
        risky = (task.agent in ("coder", "researcher") or bool(task.acceptance)
                 or getattr(task, "touched", False) or _hosting_mandatory(task))
        if risky:
            return True
        # pure read-only state query: every tool call succeeded, nothing was
        # written, so there is no judgement left to make — the output IS proof.
        return not getattr(task, "clean_reads", False)

    @staticmethod
    def _dep_context(task: Task, dag: TaskDAG) -> str:
        parts = []
        for d in task.depends_on:
            dep = dag.get(d)
            if dep and dep.output:
                parts.append(f"### Result of dependency '{dep.title}' ({dep.id})\n{dep.output[:2000]}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    SLUG_OK = _re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
    DELETE_ONLY = _re.compile(r"\b(delete|remove|clean|clear|wipe|purge|"
                              r"empty)\b", _re.I)
    CREATE_Y = _re.compile(r"\b(create|build|make|generate|new|add|"
                           r"write|setup|install)\b", _re.I)

    # v1.6: capability enforcement — the supervisor plans, the harness guarantees.
    # If the 8B worker is assigned design/code/UI/website work (its model is not
    # capable of it — this was the live complaint), it is moved to coder models.
    CODEY = _re.compile(
        r"\b(design|mockup|wireframe|ui|ux|frontend|website|web ?page|html|css|"
        r"javascript|react|vue|script that|program that|app that|api|bot|debug|"
        r"fix (the )?(bug|error)|component|landing page|portfolio)\b", _re.I)

    def _reinforce_assignment(self, dag: TaskDAG) -> None:
        for t in dag.order():
            if t.agent != "worker":
                continue
            text = f"{t.title} {t.description} {t.acceptance}"
            if self.CODEY.search(text):
                t.agent = "coder"
                t.model = ""
                self.ui.event("assign",
                              f"{t.id} reassigned worker → coder "
                              f"(design/code work needs coder models)")

    _PROJ_STOP = {"the", "a", "an", "me", "my", "in", "to", "of", "and", "is",
                  "it", "this", "that", "karo", "kar", "kr", "karna", "kar de",
                  "mein", "hi", "ek", "do", "aur", "bhi", "wala", "same",
                  "add", "update", "fix", "project", "please", "existing",
                  "into", "for", "with", "use", "using", "make", "create",
                  "build", "only", "now"}

    def _match_existing_project(self, goal: str) -> str:
        """Deterministically match a goal to an EXISTING projects/<slug>/ dir.
        Score = overlap between the goal's content words and the slug's words.
        >= 2 word overlap counts as the same project (user follow-up). Ties
        prefer the SHORTEST slug (fewest stray words)."""
        try:
            ws = Path(self.config.workspace) / "projects"
            if not ws.exists():
                return ""
            gwords = set(_re.sub(r"[^a-z0-9\s]", " ", (goal or "").lower()).split())
            gwords -= self._PROJ_STOP
            best, best_score = "", 0
            for d in sorted(ws.iterdir()):
                if not d.is_dir():
                    continue
                slug_words = set(d.name.lower().replace("-", " ").split())
                overlap = len(slug_words & gwords)
                if f"projects/{d.name}" in (goal or "").lower():
                    overlap += 10
                if overlap > best_score or (overlap == best_score
                                            and best and len(d.name) < len(best)):
                    best, best_score = d.name, overlap
            return best if best_score >= 2 else ""
        except Exception:
            return ""

    def _existing_project_context(self, goal: str) -> str:
        """v1.9.8 PROJECT MEMORY (user-reported kami: a follow-up task in the SAME
        project made the agent forget everything it had built — it re-planned from
        scratch / duplicated folders). If the goal references an existing
        projects/<slug>/ folder, inject its REAL state (files, TASKS.md, recent
        activity) into the planning context so the team builds ON the project."""
        try:
            ws = Path(self.config.workspace) / "projects"
            if not ws.exists():
                return ""
            best = self._match_existing_project(goal)
            best_score = 2 if best else 0
            if not best:
                return ""
            pdir = ws / best
            files = sorted(p for p in pdir.rglob("*")
                           if p.is_file() and "__pycache__" not in p.parts)[:25]
            tree = "\n".join(f"  {f.relative_to(self.config.workspace)}"
                             for f in files) or "  (empty)"
            tasks_md = ""
            tm = pdir / "TASKS.md"
            if tm.exists():
                tasks_md = tm.read_text(encoding="utf-8", errors="replace")[:1200]
            readme = ""
            rm = pdir / "README.md"
            if rm.exists():
                readme = rm.read_text(encoding="utf-8", errors="replace")[:400]
            recent = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)[:3]
            recent_s = ", ".join(f.name for f in recent) or "-"
            block = (
                f"### EXISTING PROJECT DETECTED: projects/{best}/ (score {best_score})\n"
                f"THIS PROJECT ALREADY EXISTS with real work inside it. Files:\n{tree}\n"
                + (f"TASKS.md (previous plan/progress):\n{tasks_md}\n" if tasks_md else "")
                + (f"README.md excerpt: {readme}\n" if readme else "")
                + f"Most recently modified: {recent_s}\n"
                "RULES FOR FOLLOW-UP WORK:\n"
                f"- Do NOT create a new/parallel project folder â build inside "
                f"projects/{best}/.\n"
                "- FIRST task: read the existing key files (TASKS.md, README, main "
                "source files) to learn what was already built.\n"
                "- EDIT/EXTEND the existing code in place. Never rewrite from scratch "
                "what already exists and works.\n"
                "- Reuse existing naming, structure and conventions.")
            self.ctx.state["existing_project_ctx"] = block
            return block
        except Exception:
            return ""

    def _apply_project_scope(self, goal: str, plan: Dict[str, Any], dag) -> None:
        """Every build-goal gets its own project folder so the workspace
        never gets cluttered (user feedback: 'a new project would mix
        files together'). The supervisor may set a 'project' slug in the
        plan; otherwise one is auto-derived from the goal if the plan
        creates files."""
        # Live bug: "workspace clean" also spawned projects/workspace-clean/.
        # Isolation is pointless for delete/clean-only goals — skip it.
        if self.DELETE_ONLY.search(goal) and not self.CREATE_Y.search(goal):
            plan.pop("project", None)
            self._clear_project_scope()
            return
        # v1.8.5: if the GOAL itself names projects/<slug>/, that slug wins —
        # live run #4: goal said varanasi-hub, scope made complete-varanasi-digital
        # and the acceptance check kept failing forever.
        gm = _re.search(r"projects/([A-Za-z0-9_-]+)", goal)
        slug = (gm.group(1).strip().lower() if gm else
                str(plan.get("project") or "").strip().lower().replace(" ", "-"))
        # v1.9.8 HARD PROJECT MEMORY: if the goal is follow-up work on an existing
        # project, the EXISTING folder wins over any slug the plan invented.
        # (Live: 'calculator app me power add karo' -> supervisor slugged
        #  calculator-app-ek and built a PARALLEL copy instead of editing the
        #  original — user-reported kami. Prompts alone were not enough.)
        if not gm:
            existing = self._match_existing_project(goal)
            if existing and slug.replace("projects/", "") != existing:
                if slug:
                    self.ui.event("warn", f"project memory: plan wanted projects/{slug} "
                                          f"— forcing EXISTING projects/{existing} "
                                          "(follow-up on previous work)")
                slug = existing
        creates = any(t.agent in ("worker", "coder") for t in dag.order())
        if not slug and creates and self.CREATE_Y.search(goal):
            words = [w for w in _re.sub(r"[^a-z0-9\s]", " ", goal.lower()).split()
                     if w not in {"a", "an", "the", "me", "my", "for", "with",
                                  "and", "to", "of", "use", "it", "using",
                                  "best", "make", "create", "build"}][:3]
            if words:
                slug = "-".join(words)[:40]
        if not slug or not self.SLUG_OK.match(slug):
            self._clear_project_scope()
            return
        pdir = slug if slug.startswith("projects/") else f"projects/{slug}"
        self.ctx.state["project_dir"] = pdir
        self.ctx.state["last_project"] = pdir
        try:
            if self.ctx.memory:
                self.ctx.memory.remember("preference", "last_project", pdir, 0.8)
        except Exception:
            pass
        if hasattr(self.ctx.fs, "set_write_scope"):
            self.ctx.fs.set_write_scope(pdir)
        note = (f"\n\n[PROJECT FOLDER] All NEW files MUST be created inside "
                f"`{pdir}/` (create it first with list_dir/write_file). "
                f"Reference existing files by their full path. Final report "
                f"must state this folder name.")
        for t in dag.order():
            if t.agent in ("worker", "coder"):
                t.description = str(t.description) + note
                t.acceptance = (str(t.acceptance) +
                                f" New files are inside {pdir}/.").strip(" .") + "."
        self.ui.event("ok", f"project folder: {pdir}/ — files isolated")

    def _clear_project_scope(self) -> None:
        self.ctx.state.pop("project_dir", None)
        if hasattr(self.ctx.fs, "set_write_scope"):
            self.ctx.fs.set_write_scope(None)

    @staticmethod
    def _manual_summary(dag: TaskDAG, err: str) -> str:
        lines = [f"(Synthesis model unavailable: {err[:100]})", "", "## Task results"]
        for t in dag.order():
            lines.append(f"\n### [{t.status.value}] {t.title}\n{t.output[:1200]}")
        return "\n".join(lines)

    def _session_listing(self) -> tuple:
        mem = self.ctx.memory
        if not mem:
            return 0, ""
        rows = []
        try:
            rows = mem.list_sessions(50) or []
        except Exception:
            rows = []
        n = len(rows)
        try:
            n = int((mem.stats() or {}).get("sessions") or n)
        except Exception:
            pass
        lines = []
        for i, s in enumerate(rows[:50], 1):
            if isinstance(s, dict):
                goal = (s.get("goal") or s.get("title") or "")[:50]
                sid = str(s.get("id") or "")[:10]
                msgs = s.get("msgs", "")
                lines.append(f"{i}. {sid}  msgs={msgs}  {goal}")
            else:
                lines.append(f"{i}. {s}")
        return n, "\n".join(lines)

    def _drop_last_project(self) -> str:
        pdir = str(self.ctx.state.get("last_project")
                   or self.ctx.state.get("project_dir") or "").strip()
        if not pdir and self.ctx.memory:
            try:
                for f in self.ctx.memory.recall("preference", 8):
                    if f.get("key") == "last_project" and f.get("value"):
                        pdir = str(f["value"]).strip()
                        break
            except Exception:
                pass
        if not pdir:
            return ("No current project in this chat to drop. "
                    "Name the folder if you want a specific one deleted.")
        self._clear_project_scope()
        self.ctx.state.pop("last_project", None)
        return (f"Dropped **{pdir}** from this session scope "
                f"(files on disk were not deleted).")

    def _checkpoint(self, dag: TaskDAG, task_id: str, goal: str = "") -> None:
        try:
            import json as _json
            d = Path(self.config.data_dir) / "checkpoints"
            d.mkdir(parents=True, exist_ok=True)
            payload = {"task_id": task_id, "goal": goal, "ts": time.time(),
                       "tasks": [t.to_dict() for t in dag.order()]}
            (d / f"{task_id}.json").write_text(_json.dumps(payload)[:200000], encoding="utf-8")
        except Exception:
            pass

    def _live_chat(self, goal: str) -> str:
        """Real Nexus reply — no canned templates."""
        sys = (
            "You are Nexus, a personal autonomous agent on the user's device. "
            "This is casual chat or slang. Reply as a smart friend. "
            "Match the user's language and script (Roman Hindi stays Roman). "
            "1-3 short lines. Never mention router/supervisor/tools/pipeline. "
            "Never use the phrase 'typo, a search, or a project name'."
        )
        try:
            return (self.ctx.llm.ask("router", goal, system=sys) or "").strip()
        except Exception:
            return ""

    def cancel(self) -> None:
        self.cancelled = True
        try:
            self.ctx.state["cancelled"] = True   # agents check this every step
        except Exception:
            pass
