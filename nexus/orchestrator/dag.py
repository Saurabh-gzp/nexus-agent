"""Task DAG with dependency-aware, parallel-safe scheduling."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass
class Task:
    id: str
    title: str
    description: str
    agent: str = "worker"
    model: str = ""              # v1.5: capability-pinned model ("" = role chain decides)
    depends_on: List[str] = field(default_factory=list)
    skill: str = ""
    acceptance: str = ""
    parallel_safe: bool = True
    status: TaskStatus = TaskStatus.PENDING
    output: str = ""
    error: str = ""
    attempts: int = 0
    score: float = 0.0
    verdict: str = ""
    started: float = 0.0
    finished: float = 0.0
    steps: int = 0
    tokens: int = 0
    depth: int = 0

    @property
    def elapsed(self) -> float:
        if not self.started:
            return 0.0
        return (self.finished or time.time()) - self.started

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "agent": self.agent,
                "status": self.status.value, "score": self.score, "attempts": self.attempts,
                "elapsed": round(self.elapsed, 1), "tokens": self.tokens,
                "output": self.output[:4000], "error": self.error[:500],
                "acceptance": self.acceptance, "verdict": self.verdict}


class TaskDAG:
    def __init__(self, tasks: Optional[List[Task]] = None):
        self.tasks: Dict[str, Task] = {}
        for t in tasks or []:
            self.add(t)

    @classmethod
    def from_plan(cls, plan: dict) -> "TaskDAG":
        dag = cls()
        for t in plan.get("tasks", []):
            dag.add(Task(
                id=t["id"], title=t.get("title", t["id"]),
                description=t.get("description", ""), agent=t.get("agent", "worker"),
                model=str(t.get("model") or ""),
                depends_on=list(t.get("depends_on", [])), skill=t.get("skill", ""),
                acceptance=t.get("acceptance", ""),
                parallel_safe=bool(t.get("parallel_safe", True)),
            ))
        dag.break_cycles()
        return dag

    def add(self, task: Task) -> None:
        self.tasks[task.id] = task

    def __len__(self) -> int:
        return len(self.tasks)

    def get(self, tid: str) -> Optional[Task]:
        return self.tasks.get(tid)

    # ------------------------------------------------------------------
    def break_cycles(self) -> List[str]:
        """Remove edges that create cycles (safety for LLM-generated plans)."""
        removed: List[str] = []
        state: Dict[str, int] = {}

        def visit(tid: str) -> None:
            state[tid] = 1
            t = self.tasks[tid]
            for dep in list(t.depends_on):
                if dep not in self.tasks:
                    continue
                s = state.get(dep, 0)
                if s == 1:
                    t.depends_on.remove(dep)
                    removed.append(f"{tid}->{dep}")
                elif s == 0:
                    visit(dep)
            state[tid] = 2

        for tid in list(self.tasks):
            if state.get(tid, 0) == 0:
                visit(tid)
        return removed

    def dangling(self) -> List[str]:
        return [f"{t.id}->{d}" for t in self.tasks.values()
                for d in t.depends_on if d not in self.tasks]

    def ready(self, max_n: int = 3) -> List[Task]:
        """Tasks whose deps are all done; respects parallel_safe."""
        out: List[Task] = []
        for t in self.tasks.values():
            if t.status is not TaskStatus.PENDING:
                continue
            deps = [self.tasks[d] for d in t.depends_on if d in self.tasks]
            if any(d.status in (TaskStatus.FAILED, TaskStatus.SKIPPED) for d in deps):
                t.status = TaskStatus.BLOCKED
                t.error = "upstream task failed"
                continue
            if all(d.status is TaskStatus.DONE for d in deps):
                out.append(t)
        if not out:
            return []
        # if the first ready task isn't parallel-safe, run it alone
        if not out[0].parallel_safe:
            return [out[0]]
        safe = [t for t in out if t.parallel_safe]
        return (safe or out[:1])[:max_n]

    def pending_count(self) -> int:
        return sum(1 for t in self.tasks.values()
                   if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING))

    def all_settled(self) -> bool:
        return all(t.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.SKIPPED,
                                TaskStatus.BLOCKED) for t in self.tasks.values())

    def failed(self) -> List[Task]:
        return [t for t in self.tasks.values()
                if t.status in (TaskStatus.FAILED, TaskStatus.BLOCKED)]

    def done(self) -> List[Task]:
        return [t for t in self.tasks.values() if t.status is TaskStatus.DONE]

    def order(self) -> List[Task]:
        """Topological-ish order for reporting."""
        out, seen = [], set()

        def walk(t: Task) -> None:
            if t.id in seen:
                return
            seen.add(t.id)
            for d in t.depends_on:
                if d in self.tasks:
                    walk(self.tasks[d])
            out.append(t)

        for t in self.tasks.values():
            walk(t)
        return out

    def summary(self) -> Dict[str, int]:
        s: Dict[str, int] = {}
        for t in self.tasks.values():
            s[t.status.value] = s.get(t.status.value, 0) + 1
        return s
