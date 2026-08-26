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
from ..agents.specialists import (CoderAgent, CriticAgent, ResearcherAgent,
                                  RouterAgent, SupervisorAgent, WorkerAgent)
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
    r"store|record|todo)\b", _re.I)
DIRECT_SAFE_INTENTS = {"chat", "question"}
ACTION_CLAIM = _re.compile(
    r"\b(deleted|removed|created|wrote|built|saved|i\s?have|i've|done)\b", _re.I)
# Device/system questions — the router used to answer "no access" even
# though system_info/termux-api can actually check them.
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
    r"hii+|good\s?(morning|afternoon|evening|night|night))\s*[!.,?]*\s*$", _re.I)
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
GREETING_REPLIES = [
    "Hey! 😄 I'm Nexus — what are we working on today?",
    "Hi! Nexus here 💪 Code, research, files, automation — what do you need?",
    "Hello! How's it going? Give me a task and watch it get done.",
    "Yo! Ready when you are — drop a task and let's go ⚡",
]

# Calculator-style math never goes through the LLM — deterministic Python.
MATH_EXPR = _re.compile(r"^[\s\d+\-*/×÷%^().,]+$")
MATH_HAS_OP = _re.compile(r"[\d)]\s*[+\-*/×÷^]\s*[\d(]")
# v1.8.1: tasks that mention these concern hosting/verification — they need the
# full coder (start_server etc.), never the cheap quick-coder path.
_QUICK_BLOCK = _re.compile(
    r"\b(host|hosting|http\.server|start_server|localhost|deploy|"
    r"live (site|url)|curl\b)\b",
    _re.IGNORECASE)
# v1.8.3: explicit hosting spec the supervisor writes into host-task descriptions
_START_SERVER_SPEC = _re.compile(
    r"start_server\(\s*command\s*=\s*['\"]([^'\"]+)['\"][^)]*?port\s*=\s*(\d+)"
    r"[^)]*?marker\s*=\s*['\"]([^'\"]+)['\"]", _re.S)
_HTML_TITLE = _re.compile(r"<title>\s*([^<]{2,80}?)\s*</title>", _re.S | _re.I)


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


def router_guard(goal: str, decision: Dict[str, Any]) -> tuple:
    """Deterministic harness rule over the router LLM's decision.

    Returns (decision, overridden). Direct answers are allowed ONLY for
    chat/question intents with no action verb in the goal and no action
    claim in the answer text. Everything else is forced to orchestration.
    """
    d = dict(decision or {})
    intent = str(d.get("intent", "unclear")).lower()
    direct = str(d.get("direct_answer") or "").strip()
    unsafe = (intent not in DIRECT_SAFE_INTENTS
              or bool(ACTION_VERB.search(goal))
              or bool(DEVICE_Q.search(goal))          # device/system question
              or bool(LIVE_Q.search(goal))            # live info — needs web
              or bool(ACTION_CLAIM.search(direct))
              or bool(MATH_HAS_OP.search(goal)))      # arithmetic — the LLM gets it wrong
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
    def handle(self, goal: str, force_orchestration: bool = False) -> RunReport:
        """Main entry: route -> (direct answer | full autonomous run)."""
        t0 = time.time()
        task_id = uuid.uuid4().hex[:8]
        self.cancelled = False
        self.ctx.llm.reset_task_budget(task_id)
        report = RunReport(goal=goal, task_id=task_id)

        # ---- safety: input moderation
        allowed, reason = self.ctx.guard.check_text(goal, "input")
        if not allowed:
            report.final = f"⚠️ {reason}"
            report.stopped_reason = "moderation"
            report.elapsed = time.time() - t0
            return report

        # ---- fast path 0: greeting / identity — instant, deterministic,
        #      always in the user's own script, zero LLM calls
        import random as _rnd
        if not force_orchestration:
            if (IDENTITY_Q.search(goal) and len(goal.split()) <= 12
                    and not ACTION_VERB.search(goal)):
                self.ui.phase("CHAT", "hello! 👋")
                report = RunReport(goal=goal, task_id=task_id,
                                   final=NEXUS_INTRO, ok=True, verified=True,
                                   elapsed=time.time() - t0)
                if self.ctx.memory:
                    self.ctx.memory.add_message("assistant", NEXUS_INTRO, "nexus")
                return report
            if GREETING_RE.match(goal):
                reply = _rnd.choice(GREETING_REPLIES)
                self.ui.phase("CHAT", "hello! 👋")
                report = RunReport(goal=goal, task_id=task_id,
                                   final=reply, ok=True, verified=True,
                                   elapsed=time.time() - t0)
                if self.ctx.memory:
                    self.ctx.memory.add_message("assistant", reply, "nexus")
                return report
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
        # greeting / very short input => do NOT give the router memory context
        if GREETING_RE.match(goal) or len(goal.split()) < 3:
            rctx = ""
        elif FOLLOWUP_REF.search(goal) and mem_ctx:
            rctx = mem_ctx[-1200:]
        else:
            rctx = ""
        decision = self.router.route(goal, rctx)
        decision, overridden = router_guard(goal, decision)
        if overridden:
            self.ui.event("warn", "router override → supervisor "
                          "(action requests cannot be answered without doing them)")
        self.ui.route_info(decision)

        if (not force_orchestration and not decision.get("needs_orchestration")
                and decision.get("direct_answer")):
            answer = decision["direct_answer"]
            report.final, report.ok, report.verified = answer, True, True
            report.elapsed = time.time() - t0
            report.tokens = self.ctx.llm.stats.snapshot().get("total_tokens", 0) - tok0
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
            plan = self.supervisor.plan(goal, mem_ctx, failure_note=note)
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
                              if _QUICK_BLOCK.search(f"{t.title} {t.description}")),
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
        self.ui.phase("SYNTHESIZE", "supervisor combining results")
        results = [t.to_dict() for t in dag.order()]
        report.tasks = results
        # v1.8.3: hosting truth is FACT, not opinion — the synthesizer gets the
        # verified evidence (or the explicit warning) so it can never fabricate
        # "HTTP 200 / live at / marker found" (live run did exactly that).
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
        facts = "\n\n".join(facts_parts)
        try:
            final = self.supervisor.synthesize(goal, results, plan, facts=facts)
        except Exception as e:  # noqa: BLE001
            final = self._manual_summary(dag, str(e))
        report.final = self._sanitize_final(final, self._goal_needs_host(goal, dag))

        # ---- final safety + memory
        ok_out, reason = self.ctx.guard.check_text(final, "output")
        if not ok_out:
            final = f"⚠️ Output withheld: {reason}"
            report.final = final
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
            ev = " ".join(self._server_evidence).lower()
            slug = ""
            try:
                slug = str(self.ctx.state.get("project_dir") or "").rsplit("/", 1)[-1].lower()
            except Exception:
                slug = ""
            # stale server on :8000 serving another project is NOT this goal
            if slug and slug not in ev and slug.replace("-", " ") not in ev:
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
        if _QUICK_BLOCK.search(goal or ""):
            return True
        if dag is not None:
            return any(_QUICK_BLOCK.search(f"{t.title} {t.description}")
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
                port = 8000
                cmd = f"python3 -m http.server {port} --directory {rel}"
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

    def _run_task(self, task: Task, dag: TaskDAG, task_id: str, t0: float = None) -> None:
        agent_name = task.agent
        task_budget = float(self.config.get("autonomy.task_timeout_seconds", 180)) * \
            (1 + self.max_retries * 0.6)          # retries get a shrinking allowance
        t_task = time.time()
        context = self._dep_context(task, dag)
        if task.skill:
            context += (f"\n\nRECOMMENDED SKILL: `{task.skill}` — call "
                        f"load_skill('{task.skill}') first and follow it.")
        if task.acceptance:
            context += f"\n\nACCEPTANCE CRITERION (must be met): {task.acceptance}"

        for attempt in range(self.max_retries + 1):
            if self.cancelled:
                task.status = TaskStatus.SKIPPED
                return
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
                     and not _QUICK_BLOCK.search(task.description))
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

            # v1.8.1: deterministic insurance — a coding task that made ZERO tool
            # calls (live: host task answered empty / 'I cannot use tools') is failed
            # fast WITHOUT a 30-60s critic round; retry immediately with the fix note.
            if agent_name == "coder" and not any(s.kind == "tool" for s in outcome.steps):
                # v1.8.3: hosting tasks go to the parachute even when the model
                # returned nothing at all (live: codestral-2508 answered with zero
                # tool calls on 3 attempts; the host never happened)
                if _QUICK_BLOCK.search(task.description) and self._host_parachute(task):
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
            if (agent_name == "coder" and _QUICK_BLOCK.search(task.description)
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
                if failed_ops and verdict.get("verdict") == "pass":
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
                if verdict.get("verdict") == "partial" and task.score >= 60 and attempt >= self.max_retries:
                    # borderline-accept: work is mostly right, but FINAL must
                    # still show 'partial' — no fake DONE (live bug #5)
                    task.status = TaskStatus.DONE
                    task.verdict = "partial"
                    return
                # a retry only helps if the critic said WHAT to fix
                fix = (verdict.get("fix_instructions") or
                       "; ".join(str(i) for i in verdict.get("issues", []) if i))
                actionable = bool(fix.strip()) and "not parseable" not in fix
                if attempt < self.max_retries and actionable:
                    context += f"\n\nCRITIC REJECTED THE PREVIOUS ATTEMPT. Fix these: {fix}"
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
        if self.config.get("autonomy.verify_all", True) is False:
            return False
        return task.agent in ("coder", "researcher") or bool(task.acceptance)

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

    def cancel(self) -> None:
        self.cancelled = True
        try:
            self.ctx.state["cancelled"] = True   # agents check this every step
        except Exception:
            pass
