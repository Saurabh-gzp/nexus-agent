"""Web tools: multi-engine search (health-aware rotation, cache, merge) + page fetch.

Search stack (all free/open, no API key — live-probed 2026-08-26):
  1. DuckDuckGo HTML (POST)   — best quality; rate-limits after ~2-3 rapid hits
  2. DuckDuckGo Lite (POST)   — same family; used when HTML is cooling
  3. Bing (HTML)              — INDEPENDENT of DuckDuckGo; most reliable fallback
  4. SearXNG public instances — meta-search; many are bot-limited, some work
  5. Mojeek                   — independent small index (some IPs are 403'd)
  6. Wikipedia API            — always works; encyclopedic/fact queries
  7. DDG Instant-Answer API   — last resort (abstract + related topics)

Key behaviours (v1.7 — fixes the live "No results after 2 searches" bug):
  * Health-aware rotation: engines that recently FAILED get demoted ~120s, so a
    blocked DuckDuckGo can never zero the whole search.
  * Query cache (600s): the same query is served from cache — a researcher
    retrying a query costs zero engine hits and zero latency.
  * Result merging: if the first engine yields fewer than requested, the next
    healthy engine's results are appended (dedup by normalized URL) until the
    quota is met or the engine list is exhausted.
  * Every engine failure is recorded and reported — the model is told which
    engines were tried so it can simplify the query instead of blind-retrying.
"""
from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .base import Risk, ToolRegistry, ToolResult
from .ssrf import SafeRedirect, url_blocked

UA_MOBILE = ("Mozilla/5.0 (Linux; Android 13; Mobile) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/122.0 Mobile Safari/537.36")
UA_DESKTOP = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
UA = UA_MOBILE                       # default UA for scrapes (kept for compat)

# searx instances tried in order (public, free; each may be under bot-pressure)
SEARX_INSTANCES = [
    "searx.be", "search.inetol.net", "priv.au", "searx.tiekoetter.com",
]


def _get(url: str, timeout: int = 25, data: Optional[bytes] = None,
         ua: str = UA_MOBILE, headers: Optional[dict] = None) -> str:
    h = {"User-Agent": ua,
         "Accept": ("text/html,application/xhtml+xml,application/json;q=0.9,"
                    "*/*;q=0.8"),
         "Accept-Language": "en-US,en;q=0.9"}
    if data is not None:
        h["Content-Type"] = "application/x-www-form-urlencoded"
    if headers:
        h.update(headers)
    why = url_blocked(url)
    if why:
        raise PermissionError(why)
    req = urllib.request.Request(url, data=data, headers=h)
    opener = urllib.request.build_opener(SafeRedirect)
    with opener.open(req, timeout=timeout) as r:
        raw = r.read()
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "ignore")


def html_to_text(src: str, max_chars: int = 12000) -> str:
    src = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header|form)[^>]*>.*?</\1>", " ", src)
    src = re.sub(r"(?is)<!--.*?-->", " ", src)
    src = re.sub(r"(?i)<br\s*/?>", "\n", src)
    src = re.sub(r"(?i)</(p|div|section|article|li|tr|h[1-6])>", "\n", src)
    src = re.sub(r"(?i)<li[^>]*>", "\n- ", src)
    for i in range(1, 7):
        src = re.sub(rf"(?i)<h{i}[^>]*>", f"\n{'#' * i} ", src)
    src = re.sub(r"(?s)<[^>]+>", " ", src)
    src = html.unescape(src)
    src = re.sub(r"[ \t\xa0]+", " ", src)
    src = re.sub(r"\n\s*\n\s*\n+", "\n\n", src).strip()
    return src[:max_chars]


def _norm_url(url: str) -> str:
    """Normalize for dedup: scheme/host case, www, trailing slash, utm params."""
    url = (url or "").strip()
    url = re.sub(r"^https?://(www\.)?", "https://", url.lower())
    url = re.sub(r"[?#].*$", "", url)
    return url.rstrip("/")


def _strip_tags(t: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", t or "")).strip()


def _ddg_unwrap(url: str) -> str:
    u = urllib.parse.unquote(url or "")
    if "uddg=" in u:
        m = re.search(r"[?&]uddg=([^&]+)", u)
        if m:
            return urllib.parse.unquote(m.group(1))
    return u


def _bing_unwrap(url: str) -> str:
    """Bing ck/a redirect → real URL (u=a1 is base64 of the target)."""
    u = html.unescape(url or "")
    if "/ck/a" in u and "u=a1" in u:
        m = re.search(r"[?&]u=a1([^&]*)", u)
        if m:
            import base64
            b64 = (m.group(1) + "===").replace("+", "-").replace("/", "_")
            try:
                real = base64.b64decode(b64).decode("utf-8", "ignore")
                if real.startswith("http"):
                    return real
            except Exception:
                pass
    return u


# ======================================================================
# engine parsers — each returns List[{title,url,snippet}]; raises on failure
# ======================================================================
def _engine_ddg_html(query: str, n: int) -> List[dict]:
    body = urllib.parse.urlencode({"q": query, "kl": "us-en"}).encode()
    page = _get("https://html.duckduckgo.com/html/", timeout=12, data=body)
    if "anomaly" in page.lower() or "unusual traffic" in page.lower():
        raise RuntimeError("ddg anomaly (rate-limited)")
    out, seen = [], set()
    for href, title, snip in re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'(?:class="result__snippet"[^>]*>(.*?)</a>)?', page, re.S):
        url = _ddg_unwrap(html.unescape(href))
        t = _strip_tags(title)
        s = _strip_tags(snip)[:300]
        if t and url.startswith("http") and "duckduckgo.com/y.js" not in url:
            key = _norm_url(url)
            if key not in seen:
                seen.add(key)
                out.append({"title": t, "url": url, "snippet": s})
        if len(out) >= n:
            break
    if not out:
        raise RuntimeError("ddg empty")
    return out


def _engine_ddg_lite(query: str, n: int) -> List[dict]:
    body = urllib.parse.urlencode({"q": query}).encode()
    page = _get("https://lite.duckduckgo.com/lite/", timeout=12, data=body)
    out, seen = [], set()
    for href, title, snip in re.findall(
            r'<a[^>]+href="([^"]+)"[^>]*class="result-link"[^>]*>(.*?)</a>.*?'
            r'(?:<td[^>]*class="result-snippet"[^>]*>(.*?)</td>)?', page, re.S):
        url = _ddg_unwrap(html.unescape(href))
        t = _strip_tags(title)
        s = _strip_tags(snip)[:300]
        if t and url.startswith("http") and "duckduckgo.com/y.js" not in url:
            key = _norm_url(url)
            if key not in seen:
                seen.add(key)
                out.append({"title": t, "url": url, "snippet": s})
        if len(out) >= n:
            break
    if not out:
        raise RuntimeError("ddg-lite empty")
    return out


def _engine_bing(query: str, n: int) -> List[dict]:
    page = _get("https://www.bing.com/search?" + urllib.parse.urlencode(
        {"q": query, "setlang": "en", "cc": "us"}), timeout=12, ua=UA_DESKTOP)
    out, seen = [], set()
    for block in re.findall(r'<li class="b_algo".*?</li>', page, re.S)[:n * 3]:
        m = re.search(r'<h2[^>]*><a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        p = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
        if not m:
            continue
        url = _bing_unwrap(html.unescape(m.group(1)))
        t = _strip_tags(m.group(2))
        s = _strip_tags(p.group(1) if p else "")[:300]
        if t and url.startswith("http"):
            key = _norm_url(url)
            if key not in seen:
                seen.add(key)
                out.append({"title": t, "url": url, "snippet": s})
        if len(out) >= n:
            break
    if not out:
        raise RuntimeError("bing empty")
    return out


def _engine_searx(query: str, n: int) -> List[dict]:
    last = None
    for inst in SEARX_INSTANCES:
        try:
            page = _get(f"https://{inst}/search?" + urllib.parse.urlencode(
                {"q": query}), timeout=8, ua=UA_DESKTOP)
            out, seen = [], set()
            for block in re.findall(r'<article class="result.*?</article>', page, re.S):
                m = re.search(r'<h3><a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
                p = re.search(r'<p[^>]*class="content"[^>]*>(.*?)</p>', block, re.S)
                if not m:
                    continue
                url = html.unescape(m.group(1))
                t = _strip_tags(m.group(2))
                s = _strip_tags(p.group(1) if p else "")[:300]
                if t and url.startswith("http"):
                    key = _norm_url(url)
                    if key not in seen:
                        seen.add(key)
                        out.append({"title": t, "url": url, "snippet": s})
                if len(out) >= n:
                    break
            if out:
                return out
            last = RuntimeError(f"{inst} empty")
        except Exception as e:  # noqa: BLE001
            last = RuntimeError(f"{inst}: {type(e).__name__}")
    raise last or RuntimeError("searx empty")


def _engine_mojeek(query: str, n: int) -> List[dict]:
    page = _get("https://www.mojeek.com/search?" + urllib.parse.urlencode(
        {"q": query}), timeout=10, ua=UA_DESKTOP)
    out, seen = [], set()
    for href, title, snip in re.findall(
            r'<a[^>]+class="ob"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'<p class="s">(.*?)</p>', page, re.S):
        url = html.unescape(href)
        t = _strip_tags(title)
        s = _strip_tags(snip)[:300]
        if t and url.startswith("http"):
            key = _norm_url(url)
            if key not in seen:
                seen.add(key)
                out.append({"title": t, "url": url, "snippet": s})
        if len(out) >= n:
            break
    if not out:
        raise RuntimeError("mojeek empty")
    return out


def _engine_wikipedia(query: str, n: int) -> List[dict]:
    j = json.loads(_get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "format": "json",
         "srsearch": query, "srlimit": n + 2}), timeout=10))
    out = []
    for r in (j.get("query", {}).get("search") or [])[:n]:
        title = r.get("title", "")
        if not title:
            continue
        url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        snip = _strip_tags(r.get("snippet", ""))[:300]
        out.append({"title": f"{title} — Wikipedia", "url": url, "snippet": snip})
    if not out:
        raise RuntimeError("wikipedia empty")
    return out


def _engine_ddg_instant(query: str, n: int) -> List[dict]:
    j = json.loads(_get("https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": 1}), timeout=10))
    out: List[dict] = []
    if j.get("AbstractText") and j.get("AbstractURL"):
        out.append({"title": j.get("Heading", query), "url": j["AbstractURL"],
                    "snippet": j["AbstractText"][:400]})
    for topic in (j.get("RelatedTopics") or [])[:n]:
        if isinstance(topic, dict) and topic.get("Text"):
            out.append({"title": topic["Text"][:80], "url": topic.get("FirstURL", ""),
                        "snippet": topic["Text"][:250]})
    if not out:
        raise RuntimeError("instant empty")
    return out


# order matters: best quality first; rotation + demotion handles blocking
ENGINES: Dict[str, Any] = {
    "ddg": _engine_ddg_html,
    "ddg_lite": _engine_ddg_lite,
    "bing": _engine_bing,
    "searx": _engine_searx,
    "mojeek": _engine_mojeek,
    "wikipedia": _engine_wikipedia,
    "instant": _engine_ddg_instant,
}
DEFAULT_ORDER = ["ddg", "ddg_lite", "bing", "searx", "mojeek", "wikipedia", "instant"]
DEMOTE_SECONDS = 120           # a failing engine rests this long
CACHE_TTL = 600                # repeated identical queries hit cache, not engines
CACHE_MAX = 64


class WebTools:
    def __init__(self, max_results: int = 6, max_chars: int = 12000):
        self.max_results = max_results
        self.max_chars = max_chars
        self._last_fail: Dict[str, float] = {}      # engine -> when it failed
        self._cache: Dict[str, Tuple[float, List[dict], str]] = {}
        self._rr = 0                                 # rotation counter

    # ------------------------------------------------------------------
    def engine_status(self) -> str:
        now = time.time()
        lines = []
        for name in DEFAULT_ORDER:
            t = self._last_fail.get(name)
            lines.append(f"{name}: " + ("cooling" if t and now - t < DEMOTE_SECONDS
                                        else "ready"))
        return "\n".join(lines)

    def _ordered_engines(self, prefer: Optional[str] = None) -> List[str]:
        """Quality-first order with a rotating top-2 (ddg <-> bing) so no single
        engine gets hammered and Bing is always reached even while DDG is
        cooling. Engines that recently failed are demoted to the back."""
        now = time.time()
        if prefer and prefer in ENGINES:
            order = [prefer] + [e for e in DEFAULT_ORDER if e != prefer]
        else:
            # rotate ONLY the top two (quality) — rest stays in quality order
            self._rr = (self._rr + 1) % 2
            top = ["ddg", "bing"]
            top = top[self._rr:] + top[:self._rr]
            rest = [e for e in DEFAULT_ORDER if e not in top]
            order = top + rest
        healthy = [e for e in order
                   if now - self._last_fail.get(e, 0) >= DEMOTE_SECONDS]
        cooling = [e for e in order if e not in healthy]
        return healthy + cooling                     # demoted engines run last

    def _cache_get(self, key: str) -> Optional[List[dict]]:
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]
        return None

    def _cache_put(self, key: str, results: List[dict]) -> None:
        self._cache[key] = (time.time(), results, "")
        if len(self._cache) > CACHE_MAX:
            oldest = sorted(self._cache, key=lambda k: self._cache[k][0])
            for k in oldest[:len(self._cache) - CACHE_MAX]:
                self._cache.pop(k, None)

    # ------------------------------------------------------------------
    def web_search(self, query: str, max_results: int = 0,
                   engine: str = "auto") -> ToolResult:
        n = max(max_results or self.max_results, 1)
        q = (query or "").strip()

        cached = self._cache_get(q.lower())
        if cached is not None:
            text = self._format(cached, q)
            return ToolResult(True, output=text, data={"results": cached,
                                                       "from_cache": True})

        want = None if (engine == "auto" or engine not in ENGINES) else engine
        results: List[dict] = []
        seen: set = set()
        tried: List[str] = []
        errors: List[str] = []

        for name in self._ordered_engines(want):
            if len(results) >= n:                     # quota filled
                break
            if name not in ENGINES:
                continue
            if want is not None and name != want:
                continue
            tried.append(name)
            try:
                got = ENGINES[name](q, n - len(results))
                self._last_fail.pop(name, None)
                for r in got:
                    key = _norm_url(r.get("url", ""))
                    if key and key not in seen:
                        seen.add(key)
                        results.append(r)
            except Exception as e:  # noqa: BLE001
                self._last_fail[name] = time.time()
                errors.append(f"{name}: {str(e)[:60]}")

        if not results:
            hint = "; ".join(errors[-3:]) or "all engines failed"
            return ToolResult(False, error=(
                f"No results for '{q}' (tried {', '.join(tried) or 'none'}: {hint}). "
                "Simplify the query to 2-3 plain keywords and retry."),
                data={"tried": tried, "errors": errors})

        self._cache_put(q.lower(), results)
        text = self._format(results, q)
        return ToolResult(True, output=text, data={"results": results,
                                                   "engines": tried,
                                                   "from_cache": False})

    @staticmethod
    def _format(results: List[dict], q: str) -> str:
        lines = [f"Search results for '{q}':", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    {r['url']}")
            if r.get("snippet"):
                lines.append(f"    {r['snippet']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def web_fetch(self, url: str, max_chars: int = 0) -> ToolResult:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        why = url_blocked(url)
        if why:
            return ToolResult(False, error=f"SSRF blocked: {why}")
        body = ""
        last_err = ""
        # attempt 1: mobile UA (default), attempt 2: desktop UA (some sites 403 mobile)
        for ua in (UA_MOBILE, UA_DESKTOP):
            try:
                body = _get(url, timeout=30, ua=ua)
                break
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {str(e)[:80]}"
        if body.lstrip().startswith(("{", "[")):
            try:
                return ToolResult(True, output=json.dumps(
                    json.loads(body), indent=2)[:self.max_chars],
                    data={"url": url, "json": True})
            except json.JSONDecodeError:
                pass
        if not body:
            if url_blocked(url):
                return ToolResult(False, error=f"Fetch failed for {url}: {last_err}")
            try:
                body = _get("https://r.jina.ai/" + url, timeout=25, ua=UA_DESKTOP)
            except Exception as e:  # noqa: BLE001
                return ToolResult(False, error=f"Fetch failed for {url}: {last_err} "
                                               f"(proxy: {str(e)[:60]})")
        text = html_to_text(body, max_chars or self.max_chars)
        title = ""
        m = re.search(r"(?is)<title[^>]*>(.*?)</title>", body)
        if m:
            title = html.unescape(m.group(1)).strip()
        if not text:
            text = body[:max_chars or self.max_chars]
        return ToolResult(True, output=f"# {title}\nSource: {url}\n\n{text}",
                          data={"url": url, "title": title, "chars": len(text)})

    def http_request(self, url: str, method: str = "GET", body: str = "",
                     headers_json: str = "") -> ToolResult:
        try:
            why = url_blocked(url if url.startswith("http") else "https://" + url)
            if why:
                return ToolResult(False, error=f"SSRF blocked: {why}")
            hdr = {"User-Agent": UA}
            if headers_json:
                hdr.update(json.loads(headers_json))
            data = body.encode() if body else None
            req = urllib.request.Request(url, data=data, headers=hdr, method=method.upper())
            opener = urllib.request.build_opener(SafeRedirect)
            with opener.open(req, timeout=30) as r:
                out = r.read().decode("utf-8", "ignore")[:8000]
                return ToolResult(True, output=f"HTTP {r.status}\n{out}")
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, error=str(e))

    # ------------------------------------------------------------------
    def register(self, reg: ToolRegistry) -> None:
        S = {"type": "string"}
        I = {"type": "integer"}
        reg.add("web_search",
                "Search the web (DuckDuckGo + Bing + SearXNG + Wikipedia, auto-rotating). "
                "Use for facts, docs, current info. RULES: plain 2-4 keyword queries "
                "(never site:/filetype:/inurl: operators); if you get 'No results', "
                "SIMPLIFY the query to 2-3 keywords and retry — a failed long query is "
                "not a dead end. Engines rotate automatically and identical queries hit "
                "the cache, so never re-run the same query.",
                {"type": "object", "properties": {"query": S, "max_results": I,
                                                  "engine": {"type": "string",
                                                             "enum": ["auto", "ddg", "ddg_lite",
                                                                      "bing", "searx", "mojeek",
                                                                      "wikipedia", "instant"]}},
                 "required": ["query"]},
                self.web_search, Risk.NETWORK)
        reg.add("web_fetch", "Fetch a URL and return readable text/markdown. Use after web_search.",
                {"type": "object", "properties": {"url": S, "max_chars": I},
                 "required": ["url"]},
                self.web_fetch, Risk.NETWORK)
        reg.add("http_request", "Make a raw HTTP request (GET/POST) to an API endpoint.",
                {"type": "object", "properties": {
                    "url": S, "method": S, "body": S, "headers_json": S}, "required": ["url"]},
                self.http_request, Risk.NETWORK,
                agents=["supervisor", "coder", "worker", "researcher", "solo"])
