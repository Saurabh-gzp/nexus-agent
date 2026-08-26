"""SafetyGuard — moderation of input/output + action risk policy."""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

HIGH_RISK_CATEGORIES = {
    "sexual/minors", "child_sexual_exploitation", "violence_and_threats",
    "self_harm", "dangerous_and_criminal_content", "weapons",
}
THRESHOLD = 0.75


class SafetyGuard:
    def __init__(self, config, llm, notifier: Optional[Callable[[str, str], None]] = None):
        self.config = config
        self.llm = llm
        self.notify = notifier or (lambda l, m: None)
        self.enabled = bool(config.get("safety.moderation_enabled", True))
        self.approval_actions = set(config.get("safety.human_approval_required", []) or [])

    # ------------------------------------------------------------------
    def check_text(self, text: str, where: str = "input") -> Tuple[bool, str]:
        """Returns (allowed, reason)."""
        if not self.enabled or not text.strip():
            return True, ""
        if where == "input" and not self.config.get("safety.moderate_user_input", True):
            return True, ""
        if where == "output" and not self.config.get("safety.moderate_final_output", True):
            return True, ""
        try:
            results = self.llm.moderate([text[:4000]])
        except Exception as e:  # noqa: BLE001
            self.notify("warn", f"moderation unavailable: {str(e)[:80]}")
            return True, ""
        for r in results or []:
            scores: Dict[str, float] = r.get("category_scores") or {}
            cats: Dict[str, bool] = r.get("categories") or {}
            flagged = [c for c, v in cats.items()
                       if v and scores.get(c, 1.0) >= THRESHOLD and
                       any(h in c for h in HIGH_RISK_CATEGORIES)]
            if flagged:
                return False, f"Blocked by safety policy ({where}): {', '.join(flagged)}"
        return True, ""

    # ------------------------------------------------------------------
    def classify_action(self, tool_name: str, args: dict) -> Optional[str]:
        """Map a tool call onto a human_approval_required action name."""
        blob = f"{tool_name} {str(args)[:400]}".lower()
        if tool_name == "delete_path":
            return "delete_files"
        if tool_name == "move_path":
            dst = str(args.get("dst", "")).lower()
            if "trash" in dst or ".deleted" in dst:
                return "delete_files"          # move-to-trash == deletion
        if tool_name in ("run_shell", "install_package"):
            cmd = str(args.get("command", "") or args.get("package", "")).lower()
            if re.search(r"\b(rm|rmdir|unlink|shred|truncate|srm|wipe|trash-put)\b", cmd) \
                    or re.search(r"(-delete\b|[^a-z]rm[^a-z])", cmd) \
                    or re.search(r"os\.remove|os\.unlink|shutil\.rmtree", cmd):
                return "delete_files"
            if re.search(r"\b(git\s+push|vercel|netlify|heroku|docker\s+push|kubectl\s+apply|"
                         r"fly\s+deploy|npm\s+publish|pip\s+upload|twine)\b", cmd):
                return "deploy_production"
            if re.search(r"\b(curl|wget)\b.*\b(pay|stripe|razorpay|checkout|billing)\b", cmd):
                return "financial_action"
        if tool_name == "run_python":
            code = str(args.get("code", "")).lower()
            if re.search(r"(os\.remove|os\.unlink|os\.rmdir|shutil\.rmtree|"
                         r"\.unlink\s*\(|send2trash|os\.renames?\s*\(.+,\s*[\"'']/dev/null)", code) \
                    or re.search(r"\brm\s+-[a-z]*\b", code):
                return "delete_files"
        if tool_name == "sqlite_exec":
            sql = str(args.get("sql", "")).lower()
            if re.search(r"\b(drop|delete|alter|truncate|attach|detach)\b", sql):
                return "delete_files"
        if tool_name == "http_request":
            url = str(args.get("url", "")).lower()
            method = str(args.get("method", "GET")).upper()
            if method in ("POST", "PUT", "PATCH", "DELETE"):
                if re.search(r"(mail|smtp|sendgrid|mailgun|resend)", url):
                    return "send_email"
                if re.search(r"(twitter|x\.com|facebook|instagram|linkedin|reddit|telegram|discord)", url):
                    return "publish_content"
                if re.search(r"(stripe|razorpay|paypal|billing|payment)", url):
                    return "financial_action"
                if re.search(r"(account|user|profile|settings|password)", url):
                    return "account_change"
        return None

    def needs_approval(self, tool_name: str, args: dict) -> Tuple[bool, str]:
        action = self.classify_action(tool_name, args)
        if action and action in self.approval_actions:
            return True, action
        return False, ""
