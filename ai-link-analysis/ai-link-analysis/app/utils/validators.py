"""URL validation and SSRF protection.

Defence layers:
1. Syntax: only http/https, no credentials, sane length, allow-listed ports.
2. Hostname: reject localhost-style / single-label / internal TLDs.
3. DNS: resolve the host and reject if ANY resolved address is not globally routable
   (private, loopback, link-local incl. cloud metadata 169.254.169.254, CGNAT, multicast, reserved...).
   This also neutralises tricks like http://2130706433/ or http://0x7f.1/, because we judge resolved IPs.

The fetcher calls validate_url() again for EVERY redirect hop.
Known residual risk: DNS rebinding between this check and the actual connection (see README).
"""
import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit, urlunsplit

from app.errors import BlockedAddressError, InvalidURLError, WebsiteFetchError

ALLOWED_PORTS = {80, 443, 8080, 8443}
MAX_URL_LENGTH = 2048
BLOCKED_SUFFIXES = (".local", ".localhost", ".internal", ".intranet", ".lan", ".home", ".corp", ".private")


def is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


async def _resolve(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise WebsiteFetchError(f"Could not resolve domain '{host}'. Check the URL is correct.")
    return sorted({info[4][0] for info in infos})


def normalize_url(raw: str) -> str:
    """Trim, default to https:// when no scheme is given, drop the fragment."""
    raw = (raw or "").strip()
    if not raw:
        raise InvalidURLError("URL must not be empty.")
    if len(raw) > MAX_URL_LENGTH:
        raise InvalidURLError("URL is too long.")
    if any(c.isspace() or ord(c) < 32 for c in raw):
        raise InvalidURLError("URL must not contain whitespace or control characters.")
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError:
        raise InvalidURLError("URL is malformed.")
    if parts.scheme not in ("http", "https"):
        raise InvalidURLError("Only http:// and https:// URLs are supported.")
    if parts.username or parts.password:
        raise InvalidURLError("URLs containing credentials are not allowed.")
    if not parts.hostname:
        raise InvalidURLError("URL has no hostname.")
    if port is not None and port not in ALLOWED_PORTS:
        raise BlockedAddressError(f"Port {port} is not allowed. Allowed ports: {sorted(ALLOWED_PORTS)}.")
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, ""))


async def validate_url(raw: str) -> str:
    """Return the normalised URL if it is syntactically valid and points to a public address."""
    url = normalize_url(raw)
    parts = urlsplit(url)
    host = parts.hostname.lower().rstrip(".")
    port = parts.port or (443 if parts.scheme == "https" else 80)

    is_ip_literal = True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        is_ip_literal = False

    if not is_ip_literal:
        if host == "localhost" or host.endswith(BLOCKED_SUFFIXES):
            raise BlockedAddressError("Internal or local hostnames are not allowed.")
        if "." not in host and not host.replace("-", "").isdigit():
            # single-label names (e.g. 'intranet') are internal; numeric ones are resolved below
            raise InvalidURLError("URL must contain a fully qualified domain name, e.g. example.com.")

    addresses = await _resolve(host, port)
    if not addresses:
        raise WebsiteFetchError(f"Could not resolve domain '{host}'.")
    for addr in addresses:
        if not is_public_ip(addr):
            raise BlockedAddressError("The URL resolves to a private or internal network address, which is not allowed.")
    return url
