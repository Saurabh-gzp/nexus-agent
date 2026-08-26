"""Block loopback / RFC1918 / link-local / metadata SSRF."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler


def url_blocked(url: str) -> str:
    u = urlparse((url or "").strip())
    if u.scheme not in ("http", "https"):
        return f"blocked scheme: {u.scheme or 'none'}"
    host = (u.hostname or "").strip().lower()
    if not host:
        return "blocked: empty host"
    if host in {"localhost", "metadata.google.internal", "metadata"}:
        return f"blocked host: {host}"
    if host.endswith(".internal") or host.endswith(".local"):
        return f"blocked host: {host}"
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return f"blocked: cannot resolve {host}"
    for info in infos:
        ip_s = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_s)
        except ValueError:
            continue
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved or ip.is_unspecified
            or ip.is_site_local
        ):
            return f"blocked address: {ip}"
        if ip_s.startswith("169.254."):
            return f"blocked metadata: {ip}"
    return ""


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        why = url_blocked(newurl)
        if why:
            raise PermissionError(f"SSRF redirect blocked: {why}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)
