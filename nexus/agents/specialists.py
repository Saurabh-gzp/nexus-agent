"""Specialized sub-agents: Router, Supervisor, Worker, Researcher, Coder, Critic.

Least-privilege tool sets:
    Router     -> no tools
    Researcher -> web + read-only fs + notes
    Worker     -> read/write fs + python + web
    Coder      -> full workspace + shell
    Critic     -> read-only + shell (tests)
    Supervisor -> planning/task tools (no destructive shell)
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ..core.jsonutil import extract_field, extract_json
from .base import AgentOutcome, BaseAgent

READ_ONLY = ["read_file", "list_dir", "search_files", "find_files", "load_skill",
             "search_knowledge", "system_info"]
WEB = ["web_search", "web_fetch", "http_request"]
WRITE = ["write_file", "edit_file", "move_path"]
EXEC = ["run_shell", "run_python", "install_package", "start_server"]
GIT = ["git_status", "git_diff", "git_log", "git_add", "git_commit"]
OFFICE = ["make_pptx", "make_pdf", "make_docx"]
DBMS = ["sqlite_exec", "sqlite_schema"]
MEM = ["remember", "index_knowledge"]


# ======================================================================
class RouterAgent(BaseAgent):
    """Fast triage — ministral-3b. Classify, estimate, route."""
    role_key = "router"
    agent_name = "router"
    allowed_tools = []
    use_skills = False
    max_steps = 1
    system_prompt = (
        "You are the ROUTER of an autonomous agent system. You are fast and cheap.\n"
        "Classify the user request and decide how it should be handled.\n"
        "Return STRICT JSON only, no prose, no markdown fences:\n"
        "{\n"
        '  "intent": "chat|question|research|code|automation|data|planning|file_ops|unclear",\n'
        '  "complexity": "trivial|simple|moderate|complex",\n'
        '  "needs_orchestration": true|false,\n'
        '  "suggested_agents": ["researcher"|"worker"|"coder"|"critic"],\n'
        '  "direct_answer": "answer text ONLY if trivial chat/greeting/simple fact, else empty",\n'
        '  "task_type": "device|web|code|data|general",\n'
        '  "model_hint": "which MODEL class fits best: coder|researcher|worker|supervisor",\n'
        '  "estimated_subtasks": 1-8,\n'
        '  "priority": "low|normal|high",\n'
        '  "duplicate_of_recent": true|false,\n'
        '  "reason": "one short sentence"\n'
        "}\n"
        "Rules: greetings/small talk/simple definitions => trivial + direct_answer filled + "
        "needs_orchestration false. Anything needing files, code, web, or multiple steps => "
        "needs_orchestration true.\n"
        "CRITICAL: You have NO tools — you cannot perform actions. If the request asks to "
        "create/delete/modify/run/save ANYTHING, you MUST set needs_orchestration=true and "
        "direct_answer=\"\" — NEVER claim you performed an action. direct_answer is ONLY for "
        "greetings, small talk, and simple factual questions.\n"
        "NEVER answer arithmetic yourself (you WILL get it wrong) — math always goes to "
        "needs_orchestration=true. NEVER say you lack access to the user's device/system "
        "(battery, storage, wifi, files) — the system has shell tools; route it with "
        "needs_orchestration=true and suggested_agents=[\"worker\",\"coder\"].\n"
        "LIVE INFO (weather, news, scores, prices — anything 'current/latest/today') — "
        "you have NO live data: set needs_orchestration=true with suggested_agents "
        "[\"researcher\"]. NEVER deflect users to other websites/apps.\n"
        "MODEL ROUTING (you classify capability):\n"
        "  * device/system questions (storage, battery, wifi, memory, phone) => task_type=device, "
        "model_hint=worker, suggested_agents=[\"worker\"] — the system has a device_info tool "
        "that knows the correct commands.\n"
        "  * code/bug-fix/website/UI requests => task_type=code, model_hint=coder, "
        "suggested_agents=[\"coder\"]\n"
        "  * live info / research => task_type=web, model_hint=researcher, "
        "suggested_agents=[\"researcher\"]\n"
        "  * the supervisor still plans in detail; your hints steer the plan.\n\n"
        "WHEN you fill direct_answer, you are speaking AS 'Nexus', the user's personal agent:\n"
        "* Reply in the EXACT SAME language AND script the user used. User writes Roman "
        "If the user writes in another language, reply in that language "
        "English => English. NEVER switch scripts on the user.\n"
        "* Warm, human, brief (1-3 lines). Like a smart friend, not a support bot.\n"
        "* Your name is 'Nexus'. NEVER mention router/supervisor/sub-agents/pipeline/"
        "classification or ANY internal detail — the user only knows Nexus.\n"
        "* If asked 'who are you': you are Nexus, an autonomous personal agent that can "
        "code, research the web, manage files and run tasks on this device.\n"
        "* Never apologise excessively, never say 'as an AI'."
    )

    def route(self, request: str, recent_context: str = "") -> Dict[str, Any]:
        prompt = request if not recent_context else f"Recent context:\n{recent_context}\n\nRequest:\n{request}"
        try:
            raw = self.llm.ask(self.role_key, prompt, system=self.system_prompt,
                               response_format={"type": "json_object"})
        except Exception:
            try:
                raw = self.llm.ask(self.role_key, prompt, system=self.system_prompt)
            except Exception as e:  # noqa: BLE001
                return self._fallback(request, str(e))
        return self._parse(raw, request)

    def _parse(self, raw: str, request: str) -> Dict[str, Any]:
        d = extract_json(raw, ["intent"])
        if d:
            d.setdefault("intent", "unclear")
            d.setdefault("complexity", "moderate")
            d.setdefault("needs_orchestration", True)
            d.setdefault("suggested_agents", ["worker"])
            d.setdefault("direct_answer", "")
            d.setdefault("task_type", "general")
            d.setdefault("model_hint", "")
            d.setdefault("reason", "")
            return d
        return self._fallback(request, "unparseable router output")

    @staticmethod
    def _fallback(request: str, reason: str) -> Dict[str, Any]:
        simple = len(request.split()) <= 6 and not re.search(
            r"\b(build|create|make|fix|write|code|research|automate|deploy|analy)", request, re.I)
        return {"intent": "chat" if simple else "unclear",
                "complexity": "trivial" if simple else "moderate",
                "needs_orchestration": not simple, "suggested_agents": ["worker"],
                "direct_answer": "", "estimated_subtasks": 1 if simple else 3,
                "priority": "normal", "duplicate_of_recent": False,
                "reason": f"router fallback: {reason[:80]}"}


# ======================================================================
class SupervisorAgent(BaseAgent):
    """Planner/orchestrator brain — mistral-medium. Plans DAG, replans on failure."""
    role_key = "supervisor"
    agent_name = "supervisor"
    allowed_tools = READ_ONLY + WEB + WRITE + MEM
    max_steps = 10
    system_prompt = (
        "You are the SUPERVISOR of an autonomous multi-agent system. You plan and coordinate; "
        "you do not do heavy execution yourself.\n"
        "Break the user's goal into a minimal DAG of concrete, verifiable subtasks and assign "
        "each to the right specialist."
    )

    PLAN_SYSTEM = (
        "You are the SUPERVISOR planner. Convert the goal into an executable task DAG.\n"
        "RULE: Plan ONLY for the CURRENT GOAL. Any CONTEXT notes are background from "
        "previous sessions — NEVER add tasks or filenames from them. Task titles must "
        "name the exact files of the current goal.\n"
        "RULE: If the goal creates a new app/script/website/report-set, include "
        "\"project\": \"short-kebab-slug\" (e.g. \"calculator-app\") so files stay isolated "
        "in projects/<slug>/. For goals operating on EXISTING files (delete/fix/read), "
        "omit the project field.\n"
        "RULE: DELETE/CLEANUP goals (delete/remove/clean/empty the workspace or folders): "
        "ONE task — coder with description 'List the target paths (list_dir), then call "
        "delete_path(path=...) for EVERY file/folder until the target is empty. The tool "
        "itself shows the user an approval prompt — never ask for permission in text.' "
        "NEVER plan 'confirm with user' tasks for deletions — approval is built into "
        "delete_path.\n"
        "RULE: NEVER create files, logs or documents just to ask the user something or " 
        "to 'clarify' — that is absurd over-engineering. If the goal is vague/greeting/"" "
        "unclear, output ONE task: worker with description 'Reply directly to the user — "
        "friendly, in their language/script, ask what exactly they want. NO tools, NO "
        "files.' acceptance: 'a short friendly clarifying question'.\n"
        "RULE: When a project slug is set, acceptance criteria must name paths INSIDE the "
        "project folder (projects/<slug>/file.py) — never bare root filenames.\n""RULE: If an earlier task produces research/design docs, the dependent build task "
        "MUST read those files and its acceptance must say it applies their concrete "
        "recommendations (name at least 3).\n"
        "Return STRICT JSON only:\n"
        "{\n"
        '  "goal_restated": "one sentence",\n'
        '  "strategy": "2-3 sentences on the approach",\n'
        '  "tasks": [\n'
        '    {"id":"t1","title":"short imperative title",\n'
        '     "description":"precise instructions incl. filenames/commands the agent must use",\n'
        '     "agent":"researcher|worker|coder|critic",\n'
        '     "model":"OPTIONAL exact model for THIS task (e.g. codestral-2508), else empty",\n'
        '     "depends_on":[],\n'
        '     "skill":"optional skill_id from the catalog",\n'
        '     "acceptance":"objective, checkable success criterion",\n'
        '     "parallel_safe":true}\n'
        "  ],\n"
        '  "final_deliverable": "what the user receives at the end"\n'
        "}\n"
        "MODEL CAPABILITY TABLE — assign every task to the agent whose MODEL best fits it:\n"
        "- coder = codestral-2508 (quick/small edits) / devstral-2512 (repo-scale work): "
        "writing or FIXING code, debugging, running builds/tests, DESIGN DOCS + MOCKUPS, WEBSITE CODING + UI/UX "
        "implementation, any \"bug fix\" request. codestral for small/single-file edits; "
        "devstral for multi-file/repository tasks.\n"
        "- researcher = mistral-small-2603: web research, live info (weather, news, prices), "
        "documents, citations.\n"
        "- worker = mistral-small-2603 (ministral-14b-2512 fallback): "
        "data shaping, summaries, formatting, comparisons, DEVICE/SYSTEM queries, simple file ops. NEVER code, never design/UI/mockups, never website work — those always go to coder.\n"
        "- critic = mistral-medium-latest: verification ONLY, and only if the goal explicitly "
        "demands verification.\n"
        "Rules:\n"
        "- 2 to 8 tasks. Fewer, meatier tasks beat many tiny ones.\n"
        "- Tasks writing to the SAME file must NOT be parallel_safe together; use depends_on.\n"
        "- CODE + BUG FIXES → coder. WEBSITE CODING / UI DESIGN → coder (frontend skill). "
        "Generic shell-only or data work → worker.\n"
        "- DEVICE/SYSTEM queries (storage, battery, wifi, memory, phone info): exactly ONE "
        "worker task that calls `device_info` + `system_info` and summarises what they return — INCLUDING any explicit 'unavailable' notes (e.g. signal strength when Termux:API is missing is a complete, acceptable answer). "
        "Do NOT plan command experiments, do NOT use coder, do NOT invent shell paths — "
        "Termux has no /sdcard; correct paths are ~/storage/* and /data/data/com.termux/...\n"
        "- DEVICE acceptance criterion: \"the report accurately reflects device_info/output; no fabricated values; explicit unavailability is acceptable and must be noted\".\n"
        "- Every task instruction MUST name the exact tools, self-contained, e.g. \"call device_info(detail=\'storage,battery,network\') and report what it returns\".\n"
        "- LIVE info (weather/news/prices/scores) → researcher with web_search, never coder. Researcher: plain 2-3 keyword queries; if 0 results retry simpler phrasing (up to 3 attempts); for named products (e.g. 'Claude AI skills') fetch first-party pages.\n"
        "- HOSTING tasks → coder uses the start_server tool (ONE call: start detached + wait for port + fetch + verify marker). NEVER run_shell for servers — not even with '&'/nohup: the tool's capture pipes close and the server accepts TCP but answers EMPTY replies (reproduced live). Never claim \"hosted\" without a verified HTTP 200 + content marker.\n"
        "- WEBSITE/build goals: NEVER plan a task that just 'writes a hosting guide' — hosting is EXECUTED. The hosting task's acceptance MUST be: 'start_server output (or the one-shot shell) shows HTTP 200 + expected content marker; the working URL is reported'. A guide file is not hosting and must FAIL. Plain `python3 -m http.server` in run_shell is hard-blocked by the harness — the plan must say 'use start_server(command=..., port=..., marker=...)'." +
        "\n"
        "- HOSTING MARKER DISCIPLINE: the hosting marker string MUST be a literal you also told "
        "the implement task to embed in the page (exactly the <title> text). Never invent a marker "
        "that may not exist in the built files — tell the implement task the exact string to put in <title>.\n"
        "- REPLAN REUSE (critical): when replanning after failures, REUSE all completed work: keep the "
        "SAME projects/<slug>/ directory (creating a second folder beside the first, e.g. portfolio-site "
        "vs portfolio-website, is a FAIL), do NOT re-plan research/implementation tasks that already "
        "passed — only re-do failed tasks. Read the 'Already completed earlier' context the engine injects.\n"
        "- HOSTING ESCALATION (critical): if a hosting/verify task fails after the coder's attempts, YOU "
        "(supervisor) must run start_server(command=..., port=..., marker=...) YOURSELF on the accepted "
        "implementation files, verify HTTP 200 + marker, and report the verified URL. The run NEVER ends "
        "by telling the user to run a server command themselves — the critic's HOSTING CHECK fails that.\n"
"\n"
        "- Do NOT plan a separate 'design doc'/'mockup' task unless the goal explicitly asks for design documents: the coder implements directly from the research task's file, and the research file itself must contain ≥3 concrete recommendations (named in the acceptance) that the code demonstrably applies. Simple site = at most 2 coder tasks (implement → host+verify), never design-doc + implement + host-guide + verify (that is 4 and costs 1000+s).\n"
        "- Plan tasks MUST name the deliverable file paths inside projects/<slug>/ and the exact tool for each step (e.g. 'call start_server(...)' for hosting).\n"
        "- SLUG CONSISTENCY (critical): if the GOAL mentions a project path like "
        "projects/<slug>/, the plan MUST use that EXACT slug everywhere (files, "
        "acceptance, hosting command). Never invent a different slug — live run: "
        "goal said projects/varanasi-hub, plan used complete-varanasi-digital, and the "
        "acceptance check failed the task forever.\n"

        "- Set \"model\" only when one exact model must run that task. IMPORTANT: any task that "
        "must CALL TOOLS (shell, files, start_server...) gets devstral-2512 — codestral-2508 "
        "is TEXT-ONLY on some accounts (it answered with zero tool calls in live runs). "
        "Otherwise leave model empty and the role chain decides.\n"
        "- Every task needs an objective acceptance criterion (file exists, tests pass, etc.).\n"
        "- Reference a skill_id when a matching playbook exists."
    )

    def plan(self, goal: str, context: str = "", failure_note: str = "") -> Dict[str, Any]:
        skills = self.ctx.skills.catalog() if self.ctx.skills else ""
        kb = self.ctx.rag.context_for(goal, max_chars=2500) if self.ctx.rag else ""
        blocks = [f"GOAL:\n{goal}"]
        if context:
            blocks.append(f"CONTEXT:\n{context[:2500]}")
        if skills:
            blocks.append(f"SKILL CATALOG (use skill ids):\n{skills}")
        if kb:
            blocks.append(kb[:2000])
        if failure_note:
            blocks.append(f"PREVIOUS ATTEMPT FAILED — fix the plan:\n{failure_note[:1500]}")
        blocks.append(f"Workspace: {self.config.workspace}")

        prompt = "\n\n".join(blocks)
        # v1.8.5: retry planning ONCE before the fallback — a flaky call must not
        # collapse a 4-phase goal into a single mega-task (live run #4).
        plan: Optional[Dict[str, Any]] = None
        for attempt in range(2):
            raw = ""
            try:
                raw = self.llm.ask(self.role_key, prompt, system=self.PLAN_SYSTEM,
                                   response_format={"type": "json_object"})
            except Exception:
                try:
                    raw = self.llm.ask(self.role_key, prompt, system=self.PLAN_SYSTEM)
                except Exception as e:  # noqa: BLE001
                    if attempt == 0:
                        continue
                    return self._fallback_plan(goal, str(e))
            plan = extract_json(raw, ["tasks"])
            if not plan and attempt == 0:
                continue
            break
        if not plan:
            return self._fallback_plan(goal, "no valid JSON in plan output")
        return self._sanitize(plan, goal)

    # Only these models may be pinned per-task by the supervisor; anything
    # else is dropped so a stray LLM answer can never inject an unknown model.
    PINNABLE_MODELS = {
        "codestral-2508", "devstral-2512",
        "mistral-small-2603", "mistral-medium-latest",
        "mistral-medium-2508", "mistral-medium-2604",
        "ministral-8b-2512", "ministral-14b-2512", "ministral-3b-2512",
    }

    def _sanitize(self, plan: Dict[str, Any], goal: str) -> Dict[str, Any]:
        tasks = plan.get("tasks") or []
        valid_agents = {"researcher", "worker", "coder", "critic"}
        ids = set()
        clean: List[dict] = []
        maxt = int(self.config.get("autonomy.max_subagents", 5)) + 3
        for i, t in enumerate(tasks[:maxt]):
            tid = str(t.get("id") or f"t{i + 1}")
            while tid in ids:
                tid += "x"
            ids.add(tid)
            agent = str(t.get("agent", "worker")).lower()
            model = str(t.get("model") or "").strip().lower()
            clean.append({
                "id": tid,
                "title": str(t.get("title") or f"Task {i + 1}")[:120],
                "description": str(t.get("description") or t.get("title") or goal),
                "agent": agent if agent in valid_agents else "worker",
                "model": model if model in self.PINNABLE_MODELS else "",
                "depends_on": [d for d in (t.get("depends_on") or []) if isinstance(d, str)],
                "skill": t.get("skill") or "",
                "acceptance": str(t.get("acceptance") or "Task output is complete and correct"),
                "parallel_safe": bool(t.get("parallel_safe", True)),
            })
        for t in clean:                       # drop dangling deps
            t["depends_on"] = [d for d in t["depends_on"] if d in ids and d != t["id"]]
        if not clean:
            return self._fallback_plan(goal, "empty task list")
        plan["tasks"] = clean
        plan.setdefault("goal_restated", goal[:200])
        plan.setdefault("strategy", "Execute the task graph and verify results.")
        plan.setdefault("final_deliverable", "Completed goal with artifacts in the workspace.")
        return plan

    def _fallback_plan(self, goal: str, reason: str) -> Dict[str, Any]:
        """v1.8.5: deterministic SPLIT fallback, never one mega-task. Mirrors the
        normal plan shape: research (if the goal asks) -> implement -> verify.
        Live run #4: an old one-worker fallback crammed a 4-phase goal into one
        coder task and died at the task time budget."""
        tasks: List[dict] = []
        import re as _re2
        if _re2.search(r"research|search the internet|web search|find and compare|"
                       r"explore|report", goal, _re2.I):
            tasks.append({
                "id": "t1", "title": "Research the topic on the web",
                "description": (goal + "\n\nFocus: research only. Search the web, fetch "
                               "pages, and write a sourced report to "
                               "projects/<slug>/research.md with >=3 concrete, named "
                               "recommendations (cite each)."),
                "agent": "researcher", "depends_on": [], "skill": "research/deep_research",
                "acceptance": ("projects/<slug>/research.md exists with >=3 concrete "
                               "recommendations and citations"),
                "parallel_safe": True})
        main_id = "t2" if tasks else "t1"
        tasks.append({
            "id": main_id, "title": goal[:80],
            "description": (goal + "\n\nFocus: implement everything the goal demands, "
                           "into projects/<slug>/. Read prior task files first."),
            "agent": "coder", "depends_on": [tasks[0]["id"]] if tasks else [],
            "skill": "", "acceptance": "All deliverables the goal lists exist and work",
            "parallel_safe": True})
        if _re2.search(r"test|host|verify|serve", goal, _re2.I):
            vid = f"t{len(tasks) + 1}"
            tasks.append({
                "id": vid, "title": "Verify, test and host per the goal",
                "description": (goal + "\n\nFocus: verification. Run the tests, host the "
                               "result with start_server(command=..., port=..., marker=...) "
                               "and verify HTTP 200 + marker."),
                "agent": "coder", "depends_on": [main_id], "skill": "",
                "acceptance": ("Tests pass and/or start_server shows HTTP 200 + the "
                               "goal's marker, with the verified URL reported"),
                "parallel_safe": True})
        return {
            "goal_restated": goal[:200],
            "strategy": f"Detached fallback plan ({reason[:80]}).",
            "tasks": tasks,
            "final_deliverable": "Completed goal with artifacts in the workspace",
            "_fallback": True,
        }

    def synthesize(self, goal: str, results: List[dict], plan: Dict[str, Any],
                   facts: str = "") -> str:
        lines = [f"GOAL: {goal}", f"DELIVERABLE: {plan.get('final_deliverable', '')}", "", "RESULTS:"]
        if facts:
            lines.append(f"\nFACTS (authoritative — never contradict them):\n{facts}")
        for r in results:
            lines.append(f"\n### [{r['status'].upper()}] {r['title']} (agent: {r['agent']})\n"
                         f"{str(r.get('output', ''))[:2500]}")
        prompt = "\n".join(lines)
        return self.llm.ask(
            self.role_key, prompt,
            system=("You are 'Nexus', the user's personal autonomous agent, writing the "
                    "FINAL answer after your team finished the work.\n"
                    "VOICE: warm, direct, human — like a smart teammate. Reply in the "
                    "EXACT SAME language and script the user used (never switch to another "
                    "script or language unless the user did). "
                    "NEVER mention agents/router/supervisor/critic/tasks/DAG or internal "
                    "machinery — everything was done by Nexus.\n"
                    "Structure: 1) what was accomplished 2) key outputs/artifacts (file "
                    "paths) 3) how to use them 4) what is incomplete + next step (if any).\n"
                    "Be concrete and concise. Never invent results that are not in the "
                    "data. Plain text/markdown only — no JSON dump.\n"
                    "HONESTY RULE: if FACTS says hosting was NOT verified, your "
                    "answer must say so explicitly — never claim HTTP 200, 'live at', "
                    "'marker found' or 'hosted' as if proven. If FACTS gives verified "
                    "hosting evidence, quote it as the proof. NEVER tell the user to "
                    "run python -m http.server / npm start / flask themselves "
                    "(that is a hosting guide = forbidden). If FACTS lists "
                    "WORKSPACE FILES, those files EXIST — never say they were not "
                    "created."))


# ======================================================================
class WorkerAgent(BaseAgent):
    role_key = "worker"
    agent_name = "worker"
    allowed_tools = READ_ONLY + WRITE + WEB + ["run_python", "run_shell",
                                               "delete_path", "move_path"] + OFFICE + DBMS
    max_steps = 10
    system_prompt = (
        "You are a GENERAL WORKER agent. You execute one concrete subtask end-to-end: "
        "data extraction, summarising, formatting, comparisons, simple file work and light "
        "API/tool calls. Be efficient — use the fewest tool calls that fully complete the task. "
        "Save substantial output to files in the workspace and report the paths.\n"
        "DELETE/CLEANUP TASKS: deleting files has exactly ONE path — call "
        "delete_path(path=...). It automatically shows the user an approval prompt "
        "(yes/always/no) and proceeds on approval. Do NOT ask the user for deletion "
        "permission in your text output, do NOT use rm/shred in run_shell (hard-blocked) — "
        "just call delete_path for every file/folder and report the results.\n"
        "DEVICE QUESTIONS: you run on the user's device. NEVER say you cannot check "
        "battery/storage/network/system — and NEVER guess shell commands. Call "
        "device_info(detail='storage,battery,network,memory') FIRST — it does ALL the "
        "probing itself (Python-first, which-guarded) and reports values OR explicit "
        "'unavailable + why' notes. YOUR ENVIRONMENT BLOCK lists what exists on this "
        "device; NEVER run termux-*/adb/dumpsys/svc/telephony commands unless that "
        "block says they exist — blind runs cost the user real time and tokens. "
        "If you truly lack the exact command: ONE web_search('how to X in termux') "
        "and follow the first working example. Report 'unavailable' honestly — that is "
        "a complete answer. Termux has no /sdcard. Compute ALL arithmetic with "
        "run_python, never in your head.\n"
        "HOSTING: use the start_server tool ONLY (start + verify in one call) — NEVER "
        "run_shell for servers, foreground OR '&'/nohup detached (detached one leaves a "
        "listener that answers EMPTY replies), and never claim a server is up without a "
        "verified HTTP 200 + content marker.\n"
        "Never start long-running servers inside run_python (it blocks until timeout) — "
        "write a small start script (bash) and report how to run it."
    )


class ResearcherAgent(BaseAgent):
    role_key = "researcher"
    agent_name = "researcher"
    allowed_tools = READ_ONLY + WEB + ["write_file", "index_knowledge", "run_python"]
    max_steps = 12
    system_prompt = (
        "You are the RESEARCH & DOCUMENT agent. Gather evidence from the web and local documents, "
        "compare multiple sources, and produce sourced findings.\n"
        "Method: search → fetch 2-4 promising pages → cross-check → synthesise.\n"
        "Always cite URLs inline as [n] with a Sources list. Flag contradictions and unknowns; "
        "never fabricate facts, dates, numbers or links. Save long reports to a .md file.\n"
        "SEARCH DISCIPLINE (v1.7 — the tool auto-rotates 7 engines + caches):\n"
        "- Plain 2-4 keyword queries ONLY. NEVER site:/filetype:/inurl:/quoted-operator "
        "queries — they return nothing. OK: 'claude ai frontend design'. NOT OK: "
        "'site:claude.com frontend best practices'.\n"
        "- 'No results' → simplify to 2-3 keywords, retry ONCE, then move on. A failed "
        "query is not failure — never loop it, never abandon research because of it.\n"
        "- Never re-run the same query verbatim (cache serves the same answer).\n"
        "- After 2+ good hits, FETCH the pages (web_fetch) and extract evidence — snippets "
        "alone are not citations. For named products ('Claude AI skills', MCP) fetch the "
        "first-party page.\n"
        "- WRITE the report file as soon as you have 2+ confirmed sources — note thin "
        "areas honestly instead of burning steps on more searches. A report with honest "
        "gaps passes; a task that never writes its file fails."
    )


class CoderAgent(BaseAgent):
    role_key = "coder_repo"
    agent_name = "coder"
    allowed_tools = READ_ONLY + WRITE + EXEC + GIT + ["delete_path"] + OFFICE + DBMS
    max_steps = 14
    system_prompt = (
        "You are the CODING agent. You inspect, write and fix real code in the workspace.\n"
        "Loop: explore (list_dir/read_file) → plan briefly → implement (write_file/edit_file) → "
        "run it (run_shell/run_python) → read the error → fix → repeat until it actually works.\n"
        "Rules:\n"
        "- ALWAYS read a file before editing it.\n"
        "- If research/design docs from earlier tasks exist, READ them and build exactly "
        "what they specify.\n"
        "- If you loaded a skill, its rules are MANDATORY — follow its checklist literally.\n"
        "- Write complete, runnable code — no TODO stubs or '...' placeholders.\n"
        "- Test what you write; a task is done only when execution succeeds.\n"
        "- Always invoke the interpreter as `python3` (never bare `python`). "
        "Run project tests with cwd=projects/<slug> so imports resolve.\n"
        "- Keep changes minimal and focused; match the project's existing style.\n"
        "- Prefer stdlib; install deps only when necessary (Termux may lack build tools).\n"
        "- On Termux you can use termux-api commands (termux-battery-status, "
        "termux-wifi-connectioninfo...) via run_shell for device data.\n"
        "- Long-running servers: start_server tool ONLY (start+verify ONE call, server "
        "stays up). run_shell for servers is hard-blocked in BOTH forms: foreground "
        "hangs till timeout; '&'/nohup detached leaves a listener answering EMPTY "
        "replies (capture pipes close). Verify REAL content markers, and NEVER claim "
        "hosting without a verified HTTP 200; never pkill a server you started.\n"
        "- REPORT ONLY REAL TOOL OUTPUT. NEVER write 'example output', 'sample', or "
        "invented values in place of actual command results — if a command failed or "
        "was unavailable, SAY SO. Fabricated output is the worst failure mode.\n"
        "- DEVICE REPORTS (Android/Termux): in `df -h` output, rows like "
        "/dev/block/dm-* , /system, /vendor, /product are READ-ONLY SYSTEM "
        "partitions — they are ALWAYS ~100% full and are NOT the user's storage. "
        "User storage = the /data and /storage/emulated rows ONLY. Never sum "
        "system partitions into 'your storage is full' — that is a false alarm."
    )

    def __init__(self, ctx, quick: bool = False):
        super().__init__(ctx)
        if quick:
            self.role_key = "coder_quick"


class CriticAgent(BaseAgent):
    """Verification agent — checks other agents' work objectively with real evidence."""
    role_key = "critic"
    agent_name = "critic"
    allowed_tools = READ_ONLY + ["run_shell", "run_python"]
    max_steps = 6
    use_skills = False
    system_prompt = (
        "You are the CRITIC / VERIFICATION agent. You are skeptical and evidence-driven.\n\n"
        "YOU HAVE WORKING TOOLS — USE THEM. You can and must:\n"
        "  • read_file / list_dir / find_files  → confirm the files really exist with real content\n"
        "  • run_shell                          → run commands, tests, `python script.py`\n"
        "  • run_python                         → execute code and check the output\n"
        "Never claim you are 'unable to execute' or that you 'lack tool access' — you have it. "
        "If a tool call errors, report THAT specific error as the issue.\n"
        "FABRICATION CHECK: if the result contains 'example output', 'sample values', "
        "'replace with actual', or ellipsis-instead-of-real-output for a command that "
        "WAS run (or should have been), the verdict is FAIL — invented numbers are "
        "worse than an honest failure.\n"
        "TOOL-FAILURE CHECK: if the prompt lists tool-call failures from the task, "
        "the verdict cannot be 'pass' unless you yourself re-verified the affected data "
        "with a working tool call; failed commands that were retried successfully and "
        "produce complete data may still score high but expect the task to be partial.\n"
        "UNAVAILABILITY CHECK: if a value is reported as 'unavailable' WITH a reason "
        "(missing app/permission, e.g. Termux:API not installed), that IS an honest, "
        "complete answer for that field — do NOT fail or retry-loop for it. "
        "Only fail when a value the tools CAN provide was omitted.\n"
        "HOSTING CHECK (critical): if the agent claims a site is 'live/hosted/serving at "
        "http://localhost:PORT', YOU MUST PROVE IT — run_shell `curl -s -o /dev/null -w "
        "'%{http_code}' http://127.0.0.1:PORT/` (or `ss -ltn | grep PORT` if curl is missing). "
        "A 'hosted' claim without a verified running server = FAIL. A hosting guide file "
        "or instructions to the user to run a server is NOT hosting."
        "- A FINAL answer that says 'run this command yourself' (python -m http.server / npm start) for "
        "hosting = FAIL — the agent must host via start_server or the run is incomplete.\n"
        "- A 'marker is present' claim MUST be proven by grepping the actual html file (run_shell grep); "
        "a sole claim of HTTP 200 without a tool output proving it = FAIL.\n"
        "If the task used "
        "start_server and its output shows HTTP 200 + marker, that output is proof — "
        "read it in the agent's claimed result.\n"
        "DEVICE-REPORT CHECK: for storage/device summaries, verify the conclusion "
        "matches the CORRECT rows — /data and /storage/emulated are user storage; "
        "/dev/block/dm-*, /system, /vendor rows are read-only system partitions "
        "(always ~100% full, normal). If the agent calls system partitions "
        "'your storage is full', the verdict is FAIL.\n\n"
        "Procedure (max 4 tool calls, be efficient):\n"
        "1. Check the artifacts named in the result actually exist and contain what was claimed.\n"
        "1b. If the task names exact files/paths (e.g. 'todo.py'), they must exist at that exact "
        "location — INSIDE the active project folder when one is in effect (the verify prompt "
        "names it), else the workspace root. Files hidden in OTHER unexpected subfolders or "
        "doubled paths (e.g. 'workspace/workspace/...') => verdict 'partial' with an issue.\n"
        "1c. If the task loaded a SKILL, verify the skill's checklist items are actually "
        "present in the produced code/files (grep for :root, @media, :focus-visible, design "
        "tokens, etc. as the skill demands). A frontend that ignores its design skill = "
        "'partial' at best. Thin/lazy output (tiny CSS, no states, no responsiveness) "
        "must FAIL, not pass.\n"
        "2. If it is code, RUN it and compare the real output to the requirement.\n"
        "3. Judge the acceptance criterion literally — nothing more, nothing less.\n\n"
        "Be fair: the criterion is the bar, not your idea of perfection. Style preferences, "
        "missing extras nobody asked for, and cosmetic nits are NOT failures.\n"
        "Fail only for: missing/empty artifacts, code that errors, wrong results, "
        "fabricated claims, or an unmet stated requirement.\n\n"
        "Your FINAL message must be ONLY this JSON object, nothing before or after:\n"
        '{"verdict":"pass","score":95,"issues":[],"missing":[],"fix_instructions":""}\n'
        "verdict: pass | partial | fail — score 0-100 — fix_instructions: precise, actionable."
    )

    def verify(self, task_title: str, acceptance: str, output: str,
               task_id: str = "global",
               tool_failures: Optional[List[str]] = None) -> Dict[str, Any]:
        prompt = (f"TASK: {task_title}\n\nACCEPTANCE CRITERION: {acceptance}\n\n"
                  f"AGENT'S CLAIMED RESULT:\n{output[:5000]}")
        if tool_failures:
            prompt += ("\n\nTOOL-CALL FAILURES DURING THE TASK (real errors, not optional):\n"
                       + "\n".join(f"- {f}" for f in tool_failures[:8])
                       + "\nIF ANY FAILURE COULD AFFECT THE RESULT, the verdict is 'partial' "
                         "at best — the result cannot be 'pass' unless you independently "
                         "confirmed the affected data with your own working tool call.")
        prompt += ("\n\nVerify it now using your tools (read the files, run the code), "
                   "then reply with ONLY the JSON verdict.")
        try:
            pdir = str(self.ctx.state.get("project_dir", "") or "")
        except Exception:
            pdir = ""
        if pdir:
            prompt += (f"\n\nNOTE: project scope is ACTIVE for this goal — expected file "
                       f"location is INSIDE {pdir}/ (not the workspace root). A file at "
                       f"{pdir}/name.py counts as 'name.py at the exact expected location'. "
                       "Do NOT demand root-level placement.")
        res = self.run(prompt, task_id=task_id)
        return self._parse(res.output, res)

    def hard_verify(self, task_title: str, acceptance: str, output: str,
                    task_id: str = "global") -> Dict[str, Any]:
        """Escalate to the large model for a difficult final check (rate-limited)."""
        prompt = (f"FINAL HARD VERIFICATION\nTASK: {task_title}\nACCEPTANCE: {acceptance}\n\n"
                  f"RESULT:\n{output[:6000]}\n\nOutput only the JSON verdict.")
        try:
            raw = self.llm.ask("hard_fallback", prompt, system=self.system_prompt, task_id=task_id)
            return self._parse(raw, None)
        except Exception as e:  # noqa: BLE001
            # verification itself failed → 'fail'. NEVER return 'partial':
            # the engine may accept it and mark unverified work 'done'
            # (live bug #5).
            return {"verdict": "fail", "score": 50,
                    "issues": [f"hard verify unavailable: {e}"],
                    "missing": [], "fix_instructions": ""}

    @staticmethod
    def _parse(text: str, res: Optional[AgentOutcome]) -> Dict[str, Any]:
        text = text or ""
        d = extract_json(text, ["verdict"])
        if d and "verdict" in d:
            v = str(d.get("verdict", "")).lower().strip()
            if v not in ("pass", "fail", "partial"):
                v = "partial"
            try:
                score = float(d.get("score", 70 if v == "pass" else 40))
            except (TypeError, ValueError):
                score = 70 if v == "pass" else 40
            issues = d.get("issues") or []
            if isinstance(issues, str):
                issues = [issues]
            missing = d.get("missing") or []
            if isinstance(missing, str):
                missing = [missing]
            return {"verdict": v, "score": max(0.0, min(100.0, score)),
                    "issues": [str(i)[:200] for i in issues][:6],
                    "missing": [str(m)[:200] for m in missing][:6],
                    "fix_instructions": str(d.get("fix_instructions", ""))[:600],
                    "raw": text[:1200]}

        # --- no JSON: infer from prose + tool evidence -------------------
        field_v = (extract_field(text, "verdict") or "").lower()
        low = text.lower()
        if field_v in ("pass", "fail", "partial"):
            v = field_v
        elif re.search(r"\b(all (checks|criteria) (pass|met)|verified|works? correctly|"
                       r"criterion (is )?met|looks correct)\b", low):
            v = "pass"
        elif re.search(r"\b(does not exist|missing|error|failed|incorrect|not met|"
                       r"empty file|traceback)\b", low):
            v = "fail"
        else:
            v = "partial"
        # tool evidence: if the critic actually ran things successfully, lean positive
        if res is not None:
            tool_steps = [s for s in res.steps if s.kind == "tool"]
            if tool_steps and all(s.ok for s in tool_steps) and v == "partial":
                v = "pass"
        try:
            score = float(extract_field(text, "score") or (85 if v == "pass" else
                                                           25 if v == "fail" else 55))
        except (TypeError, ValueError):
            score = 85 if v == "pass" else 55
        return {"verdict": v, "score": score,
                "issues": [] if v == "pass" else ["verdict inferred from prose (no JSON)"],
                "missing": [], "fix_instructions": "", "raw": text[:1200]}


AGENT_CLASSES = {
    "router": RouterAgent,
    "supervisor": SupervisorAgent,
    "worker": WorkerAgent,
    "researcher": ResearcherAgent,
    "coder": CoderAgent,
    "critic": CriticAgent,
}
