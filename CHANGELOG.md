# Changelog

## [1.8.8] — office + sqlite tools, task/OOP/DBMS skills, start_server on coder whitelist

New tools: `make_pptx`, `make_pdf`, `make_docx`, `sqlite_exec`, `sqlite_schema`.
Coder/worker can actually *see* `start_server` (it was registered but missing
from `allowed_tools`). Skills: office docs, sqlite DBMS, Python OOP, task
management. Web-automation skill no longer teaches blocked `nohup http.server`.

## [1.8.7] — goal-level hosting parachute + no DIY host-guide + honest files

**Trigger (parity TUI, 8 keys, 4-phase Varanasi Digital Hub):** 913.6s,
`unverified`, t3 hit the 900s overall cap so t4 never started. Final box
honestly said hosting NOT verified, but then told the user to run
`python3 -m http.server 8000` themselves, and denied that `test_contact.py`
existed (it did; 8/8 pass from the project dir). Fixes:

1. **Goal-level parachute** after the DAG ends: if the USER GOAL asked to
   host and there is still no `start_server` evidence, the harness hosts the
   newest `projects/*/index.html` even when the host-task never ran.
2. **DIY host-guide banned twice** — FACTS + deterministic `_sanitize_final`
   strip `python -m http.server` / "run the server yourself" lines.
3. **Workspace file FACTS** — synthesizer is given the real file list and
   must not claim existing files were never created.
4. **`python` → `python3`** rewrite in `run_shell` (bare `python` is often
   missing; live tests looked failed).
5. **`--version`** reads `app.version` (was stuck at 1.4.2).
6. Default **`overall_timeout_seconds` 900 → 1500** so a 4-phase research
   + build + test + host run can finish.

## [1.8.3] — hosting can never be faked: verified-by-construction

**Trigger (8-key live run, 4-phase 'Varanasi Digital Hub'):** completed 4/4 tasks
in 578.8s with 0 replans (vs 1191.7s + 2 replans before) — the 8-key pool worked,
BUT post-run audit found the hosting task had been accepted with **no server at
all**: the coder took 4 unrelated steps, the critic gave 'partial 70', and the
old `score >= 70 -> DONE` shortcut let it pass — and the final answer box then
fabricated 'HTTP/1.0 200 OK ... marker found'. Fixes:

1. **Only a real `pass` completes instantly.** Any `partial` (even 70-99) retries
   with the critic's fix note; accepted only at max retries (score>=60).
2. **Hosting is verified-by-construction:** a coder task mentioning
   host/server/verify/http/marker must contain a VERIFIED start_server tool call,
   else the harness executes hosting itself — `_host_parachute()` parses the
   plan's `start_server(command=..., port=..., marker=...)` spec, or serves the
   newest `projects/*/index.html` with its `<title>` as marker. Task becomes
   DONE only with real evidence. (E2E: 41s, ok+verified, HTTP 200 live.)
3. **The synthesizer gets FACTS** — verified start_server evidence to quote, or
   an explicit "HOSTING REALITY: not verified" block plus an HONESTY RULE
   (never claim HTTP 200 / live / marker without evidence). Live-run fabrication
   is now impossible; verified report=false when no evidence exists.
4. **Silent-model exclusion:** if attempt 0's planned model returns zero tool
   calls (live: codestral-2508 did this 3x on this account), retries always use
   the role chain (devstral-2512) — and the supervisor is told tool-requiring
   tasks get devstral-2512 (codestral-2508 = text-only on some accounts).

Verified: 147/147 tests; the 8-key run itself: 5 rate-limit events (only the
shared key#1), 0 stalls, 0 replans, honest final answer.

## [1.8.2] — rotation kap bhi nahi: jitne keys, utni tries
 — rotation kap bhi nahi: jitne keys, utni tries

`max_key_rotations_per_call` ab **0 = no cap**: har call me ring ke HAR key ko try
karta hai (pehle max(6, len) tha — ab sirf len). Sab cooling ho to ring ≤45s wait
karke soonest key se retry karti hai — agent pause karta hai, kabhi DIE nahi karta.
README me naya section: keys ki minimum/ideal count ka hisaab (peak ~1.5-3 rps →
5-6 minimum, 10 = safe) + `MISTRAL_APIS="k1,k2,..."` bulk form + "alag accounts
best" note (per-org quotas stack nahi karte same org me).

## [1.8.1] — multi-key pools: run 10+ Mistral keys in one job
 — multi-key pools: run 10+ Mistral keys in one job

`KeyRing.discover` now scans `MISTRAL_API_KEY_1.._20` (earlier: .._9) and still
supports the bulk form `MISTRAL_APIS="key1,key2,..."`. A 10-key pool absorbs the
429 storms seen in the live TUI run (dozens of `rate-limited -> cooling` events);
the ring rotates per call (`max_key_rotations_per_call`), cools a key 4s after a
429 and waits for the soonest healthy key instead of failing.

## [1.8.0] — hosting is EXECUTED, never a guide; the run never ends unhosted
 — hosting is EXECUTED, never a guide; the run never ends unhosted

**Trigger (live A/B):** the user's own TUI run ended with the agent writing a
`hosting_guide.md` and claiming "site is live at localhost:8000" — it had only
run `python3 -m http.server 8000` in the foreground, which burned 120s of
timeout and never served anything. Our reproduction run then showed the same
anti-pattern surviving in new forms; every one is fixed at the harness level:

1. **run_shell server choke-point (unconditional):** `python3 -m http.server`,
   flask/uvicorn/gunicorn/hugo/jekyll, npm/pnpm/yarn run dev|start|serve,
   vite/next, node/bun/server, `php -S`, `rails s`, `serve -s` are hard-blocked
   in run_shell — BOTH foreground (hangs till timeout) AND `&`/nohup detached.
   Root cause of the detached form (reproduced 2026-08-26): the tool's capture
   pipes stay open so the call blocks, and after it returns the pipe read-ends
   close → every request handler dies on BrokenPipeError → the server accepts
   TCP but answers **EMPTY replies**. Only `start_server` (detached + log file +
   port-wait + fetch + marker-verify in ONE call) is allowed.
2. **Supervisor plan rules:** no "hosting guide" task ever (hosting is
   executed; acceptance = verified HTTP 200 + marker); hosting marker must be a
   literal the implement task was told to embed in `<title>`; HOSTING
   ESCALATION — if the coder can't host, the supervisor runs `start_server`
   itself and verifies; the run NEVER ends by telling the user to "run this
   command yourself"; REPLAN REUSE — replans keep the SAME `projects/<slug>/`
   dir and only redo failed tasks (live run re-researched 3× and built a second
   folder `portfolio-site` beside `portfolio-website`).
3. **Critic HOSTING CHECK extended:** a final answer handing a server command
   to the user = FAIL; any "marker is present" claim must be proven by grepping
   the actual html file (live: final summary claimed the marker while the
   critic had shown it missing from both index.html files).
4. **Engine discipline:** `_QUICK_BLOCK` — the cheap quick-coder is never used
   for host/server/verify/port/http tasks (live: short description put the host
   task on codestral-2508, which returned an EMPTY response → wasted attempt);
   a coder task that finishes with ZERO tool calls is failed fast WITHOUT a
   30-60s critic round and retried with a "you MUST call tools" note; the
   overall deadline is enforced on every agent step (a long task can't sail
   past the cap — the live run's last task burned 325s beyond the 900s cap because
   only a later future would have triggered the check).
5. **Test hygiene:** the server-block test's own `&`-detached server leaked out
   of pytest and occupied :8000 with an EMPTY directory — the live agent's
   "server" was actually answering from it and the whole hosting phase was
   polluted. Test now kills what it spawns and the whole suite leaves nothing
   listening.

Verified: 142/142 tests; live TUI run (Mistral keys) analysed end-to-end and
the failure chain in this section matches it step for step.

## [1.7.0] — multi-engine web search: never "No results" again
 — multi-engine web search: never "No results" again

**Root cause found (live probe):** DuckDuckGo HTML/Lite rate-limits the IP after
~2-3 rapid queries (returns an "anomaly" page with 0 results). The researcher's
3rd-5th queries hit the same blocked DDG chain → "all engines empty", task
failed, retry loops. Fixes:

1. **Search stack rebuilt — 7 engines, health-aware rotation, cache, merge:**
   DuckDuckGo HTML → DuckDuckGo Lite → **Bing (independent — works while DDG is
   blocked)** → SearXNG public instances ×4 → Mojeek → Wikipedia API → DDG
   Instant-Answer. Live-probed 2026-08-26: DDG ×2 OK then anomaly; Bing still
   returns 5-7 results during the block; Wikipedia API always works;
   Mojeek/Brave/Ecosia/Startpage 403/429 (bot-protected — kept for other IPs).
2. **Engine demotion (120s):** a failing engine is pushed to the back, so one
   blocked engine can never zero the search. Top-2 (ddg ↔ bing) rotate per call
   so neither gets hammered.
3. **Query cache (600s):** identical queries are served from cache — zero
   engine hits, zero latency, no repeat-query burn.
4. **Result merging + dedup:** engines top up each other until the requested
   count; URLs normalized (www, trailing slash, utm stripped, bing ck/a and
   ddg uddg redirects unwrapped).
5. **web_fetch hardened:** mobile UA → desktop UA retry → r.jina.ai proxy
   fallback; JSON endpoints auto-formatted. `engine:` param lets any single
   engine be pinned.
6. **deep_research skill updated** (mandatory §0): plain 2-4 keyword queries
   only (NEVER site:/filetype:/inurl:/quoted operators — they return nothing),
   simplify-once-then-move-on, never re-run the same query, fetch pages after
   2+ hits, write the report after 2+ confirmed sources, and the researcher
   prompt enforces the same rules.
7. **Live tests (previously failing queries now all pass):**
   - "Claude AI portfolio website frontend best practices" → 0.9s, Anthropic's
     own blog as top hit
   - "Claude AI design system accessibility WCAG 2026", "Claude AI responsive
     design performance...", "how to check network type in termux",
     "claude ai skills" → all 0.1-2.6s with results
   - scrapes: claude.com blog (509 KB), bbc.com, HN → all clean text
   - End-to-end research task → 58s, real report written to
     projects/search-web-claude/frontend_design_best_practices.md with 4 real
     cited sources (claude.com blog, platform.claude.com cookbook,
     claude.com/plugins, practitioner blog)
   - 139/139 offline tests (new: parser unwrap, cache, demotion, dedup-merge)

# Changelog

## [1.6.0] — sutra-style harness discipline: no more wrong commands

Live audit of "network status check" (409.6s / 186,094 tokens / 17 failed
commands / 3 critic retries) and a portfolio build revealed the root cause:
the agents were GUESSING commands (`termux-am`, `termux-telephony-*`, `adb
shell`, `dumpsys`, `svc wifi`) and the 8B worker was assigned design/code work.
All fixed with harness-level rules (model proposes, harness disposes):

1. **`device_info` rewritten sutra-style** — pure-Python probes first
   (`socket` for network, `shutil.disk_usage`, `/proc/meminfo`, sysfs battery),
   `shutil.which()` guard before EVERY external command, and anything missing is
   reported as `unavailable` **with a fix hint** — never guessed, never retried.
   Latency: 27–31s → **0.1s**.
2. **`availability()` env facts** — the system prompt now tells the agent
   exactly what exists on the device (`termux-battery-status=no`, `getprop=yes`,
   ...). Blind `termux-*/adb/dumpsys` guesses die before they happen.
3. **Consecutive-failure BRAKE in the agent loop** — 3 failed tool calls in a
   row force a HALT message: device question → re-run device_info only; unknown
   command → ONE web_search; else finalize honestly. Plus a wrap-up nudge before
   the step budget ends. (This is what turned 17 failed runs into 0.)
4. **Capability enforcement in the engine** — deterministic harness rule: any
   worker task matching design/code/UI/website/API/bug-fix keywords is
   reassigned to coder. Supervisor prompt hardened too: design docs/mockups/
   wireframes → coder, never worker; worker scope = data/device/summaries only.
5. **`start_server` tool** — one-shot hosting: launches the server DETACHED,
   waits for the port, fetches the URL, verifies content markers, reports the
   verified URL. The "hosted" claim is now proven (live: 200 + `<title>` match);
   the server stays up across calls. Tolerates fuzzy LLM kwargs.
6. **web_search multi-engine fallback** — DuckDuckGo HTML → DDG lite → Bing →
   Mojeek → instant-answer API, so one blocked engine no longer zeroes
   research ("No results for 'Claude AI frontend design...'" fixed — that query
   now returns Anthropic's own page).
7. **Critic**: a value reported `unavailable` WITH a reason is a complete,
   honest answer — no more retry-loops demanding signal strength that the
   device cannot provide. Tool-failure insurance (≤79) still applies.
8. Coder/worker prompts: hosting = start_server OR one-shot nohup+curl with
   marker verification; never claim hosting without HTTP 200 + content.

**Verified live (self-driven tasks, no test files):**
- `network status check` → **34s · 7,534 tokens · 0 failed commands · pass 100**
  (was 409.6s · 186k tokens · 17 failures · 3 retry loops)
- `storage info` → 19.5s · 8,101 tokens (was 251.5s · 144,732 tokens)
- Portfolio build + host → files in `projects/portfolio2/`, server on :8091
  verified 200 + title, still alive after the run
- Plan audit: portfolio = researcher(small) → coder(devstral) → coder(codestral);
  worker assigned NOTHING code/design. Storage = 1 worker task only.
- 134/134 offline tests.

# Changelog

## [1.5.0] — capability-aware autonomous agent + live-audit fixes

**Model inventory (live-verified against the API, 2026-08-26):**
- 56 models exist on the org key; 13 chat models tested live — all respond except
  `mistral-large-2512` (hangs >180s on this key — moved to second fallback) and
  `labs-leanstral-1-5-1` / `glm-5-2` (HTTP 403 — not enabled for the org).
- Embeddings verify OK (mistral-embed-2312 dim 1024, codestral-embed dim 1536);
  moderation OK; OCR OK (data-URI PDFs).
- Config rate_limits now MATCH the real organization limits
  (admin.mistral.ai → Limits): embed 1.0 rps (was 4.0 — caused 429s),
  small 0.83, 8b 3.13, 3b 12.5, codestral 2.08, devstral 0.83, large 0.07.

**Capability-aware routing & planning (the big one):**
- Router is now the **ministral-8b-2512 decider**: it classifies the request and
  emits `task_type` (device|web|code|data|general) + `model_hint`; the
  supervisor's plan is steered by that hint instead of planning blind.
- Supervisor plan prompt now carries a **MODEL CAPABILITY TABLE** and assigns
  every task to the agent whose model fits: coder=devstral/codestral (code,
  bug fixes, website/UI), researcher=mistral-small (web/live info),
  worker=ministral-8b/14b (data, summaries, device queries), critic=medium.
- New optional per-task `"model"` pin (validated against a whitelist) — the
  supervisor can force `codestral-2508` for a small code task etc. The UI plan
  table now shows the model that will run each task (frontend = simple plan,
  detail stays in the backend).
- Worker fallback chain now includes ministral-14b-2512 (verified live).

**Token/time-waste fix (live bug: "storage info" burned 251s / 144,732 tok):**
- New `device_info` tool — one-shot, CORRECT device report (storage via
  Termux paths `~/storage/*`, `df -h /data /storage/emulated/0`, battery,
  network, memory). No more guessing `/sdcard/*` on Termux (does not exist)
  and 5 failed `du` runs per query.
- Worker prompt: never run a command blind — use device_info or web-search the
  exact command; unknown paths must not be fired as guesses.
- Plan rules: device/system queries = exactly ONE worker task calling
  device_info; never coder; never command experiments. Live queries →
  researcher, never coder.
- Critic + engine: any task whose tool calls errored can no longer score
  100-pass without justification — verdict is capped at 'partial' (79) unless
  the critic re-verifies the affected data itself. (Live bug: storage task
  with 5 failed `du` runs still scored 100.0.)

**Bugs fixed:**
1. `MemoryStore.resolve_session` — an all-digit session-id prefix (e.g.
   `123456`) was misread as a session NUMBER and returned None
   (`/resume <prefix>` broken for ~3.7% of sessions). Exact-id is now checked
   first and out-of-range numbers fall through to prefix matching.
2. `tests/test_live.py` — asserted artifacts at the workspace root, but
   v1.2+ project isolation puts them in `projects/<slug>/` → 2 false failures.
   Now checks both locations; also reports the real client's token stats.
3. `tests/test_tui_session.py` — SyntaxError on line 24: the "English-only"
   commit merged two lines into `ROOT = ... ART = ...`. Fixed; ROOT is now
   derived from `__file__` instead of being hardcoded.

**Tests:** 133/133 offline (was 132 pass + 1 flaky); live suite 26/26.

# Changelog

## [1.4.2] — "workspace clean" disaster fix + full tool audit

**Root causes fixed (live pty-verified):**
1. `delete_path` now accepts `path`/`src`/`target` aliases — the agent's `src:`
   parameter no longer raises a TypeError after approval.
2. Added `delete_path`/`run_shell`/`move_path` to the worker agent's allowed
   tools — previously the worker had NO way to delete, so it just asked the
   user for a "YES" in plain text.
3. DELETE/clean-only goals no longer create a `projects/<slug>/` folder
   (engine `_apply_project_scope` skips). "workspace clean" no longer
   pollutes the scope.
4. Approval 'a' (always) is now ACTION-level — one 'a' covers the whole
   batch for that action (live proof: 1 prompt → 7 delete_path calls
   proceed silently).
5. Critic retry-exhaustion no longer marks a task 'done': hard-verify
   fail/unavailable → honestly FAILED. Borderline acceptance only at
   score ≥ 60 + verdict 'partial'.
6. Worker/supervisor prompts: deletions must use delete_path (the only
   path), planning 'confirm with user' tasks for deletions is forbidden.
   The critic is now project-scope aware — root-location false-conflict
   retries eliminated (build: 55s → 21s).
7. Tool errors are visible in the UI (`↳ reason` under `✕`) — no more
   silent failures.

**Verified:** tool suite 22/22 by direct execution (fs/shell/python/web/
skills/memory/RAG + BLOCKED rm/python deletes), live pty run of
"clean the workspace, delete everything" → 1 approval, 7 deletes,
critic 100.0 pass, workspace EMPTY, no manual `rm -rf`.
Build-goal regression: `projects/<slug>/` isolation intact. Tests: 128 pass.
Setup: one-command `setup.sh` (deps + old-key purge + self-test + launch
instructions); `--update` mode keeps keys. Entire codebase now English-only.

## 1.9.0
- Quota-based roles (medium-2508 / small-2603, not 8B primary)
- start_server path vs URL; Termux chat fast-paths; `/` menu default on
- Duplicate failed tool-call diagnosis; git_status/diff/log tools

## 1.9.1
- start_server: argv-only python -m http.server; reject rm/curl/traversal
- SSRF block + redirect revalidation on web_fetch/http_request
- Path sandbox uses relative_to (not startswith)
- Destructive SQL classified as delete_files (approval)
- partial/timeout no longer count as success; no invented project on fix-only goals
- Dangling DAG deps trigger fallback replan

## 1.9.2
- OpenAI-compatible urlopen watchdog (shared httpwatch)
- acquire_or_wait never force-healthy before Retry-After
- Memory/RAG SQLite reads under the same lock

## 1.9.3
- git_add / git_commit (local only, path-sandboxed)
- DAG checkpoint JSON after every task
