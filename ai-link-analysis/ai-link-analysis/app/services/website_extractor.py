"""Fetch a website safely and turn its HTML into compact, LLM-ready text.

Safety: every redirect hop is re-validated (SSRF), downloads are size-capped, timeouts are enforced.
Efficiency: scripts/styles/svg removed, whitespace collapsed, duplicate lines dropped, text truncated to a budget.
"""
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from app.config import Settings
from app.errors import (
    AppError,
    UnsupportedContentError,
    WebsiteFetchError,
    WebsiteTimeoutError,
)
from app.utils.validators import validate_url

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; LinkAnalysisBot/1.0; +https://github.com/your-username/ai-link-analysis)"
STRIP_TAGS = ["script", "style", "noscript", "svg", "iframe", "canvas", "template", "form", "picture", "video", "audio"]
INTERESTING = ("about", "service", "product", "solution", "what-we-do", "offering", "features", "company", "industries", "portfolio", "pricing")
SKIP_EXT = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".zip", ".mp4", ".css", ".js", ".xml", ".ico")
MIN_USEFUL_CHARS = 150


@dataclass
class PageContent:
    url: str
    title: str = ""
    meta_description: str = ""
    site_name: str = ""
    structured_data: list[str] = field(default_factory=list)
    nav_items: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    links: list[tuple[str, str]] = field(default_factory=list)  # (absolute href, anchor text)
    text: str = ""


@dataclass
class WebsiteContent:
    final_url: str
    pages: list[PageContent]
    prompt_text: str = ""

    @property
    def home(self) -> PageContent:
        return self.pages[0]


# --------------------------------------------------------------------------- parsing
def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _jsonld_summaries(soup: BeautifulSoup) -> list[str]:
    out: list[str] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or "")
        except (ValueError, TypeError):
            continue
        nodes = data if isinstance(data, list) else data.get("@graph", [data]) if isinstance(data, dict) else []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            typ = node.get("@type")
            name, desc = node.get("name"), node.get("description")
            if typ and (name or desc):
                out.append(_clean(f"{typ}: {name or ''} - {desc or ''}")[:300])
    return out[:5]


def parse_html(html: str, url: str) -> PageContent:
    soup = BeautifulSoup(html, "html.parser")
    page = PageContent(url=url)

    if soup.title and soup.title.string:
        page.title = _clean(soup.title.string)[:200]
    meta = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if meta and meta.get("content"):
        page.meta_description = _clean(meta["content"])[:500]
    og_site = soup.find("meta", attrs={"property": "og:site_name"})
    if og_site and og_site.get("content"):
        page.site_name = _clean(og_site["content"])[:100]
    if not page.meta_description:
        og_desc = soup.find("meta", attrs={"property": "og:description"})
        if og_desc and og_desc.get("content"):
            page.meta_description = _clean(og_desc["content"])[:500]

    page.structured_data = _jsonld_summaries(soup)

    # Links + navigation labels BEFORE we strip layout tags
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        page.links.append((urljoin(url, href), _clean(a.get_text(" "))[:80]))
    nav_labels: list[str] = []
    for nav in soup.find_all("nav"):
        for a in nav.find_all("a"):
            label = _clean(a.get_text(" "))
            if 1 < len(label) <= 40 and label not in nav_labels:
                nav_labels.append(label)
    page.nav_items = nav_labels[:25]

    for tag in soup(STRIP_TAGS):
        tag.decompose()

    headings: list[str] = []
    for h in soup.find_all(["h1", "h2", "h3"]):
        t = _clean(h.get_text(" "))
        if t and t not in headings:
            headings.append(t[:150])
    page.headings = headings[:30]

    for tag in soup.find_all(["nav", "footer"]):
        tag.decompose()

    seen: set[str] = set()
    lines: list[str] = []
    for raw in soup.get_text("\n").splitlines():
        line = _clean(raw)
        if len(line) < 3:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    page.text = "\n".join(lines)
    return page


def select_internal_links(page: PageContent, limit: int) -> list[str]:
    """Pick same-site pages most likely to describe what the business does."""
    if limit <= 0:
        return []
    base = urlsplit(page.url)
    base_host = (base.hostname or "").removeprefix("www.")
    scored: dict[str, int] = {}
    for href, text in page.links:
        parts = urlsplit(href)
        if parts.scheme not in ("http", "https"):
            continue
        if (parts.hostname or "").removeprefix("www.") != base_host:
            continue
        path = parts.path.lower()
        if path in ("", "/") or path.endswith(SKIP_EXT):
            continue
        clean = f"{parts.scheme}://{parts.netloc}{parts.path}"
        if clean.rstrip("/") == page.url.rstrip("/"):
            continue
        haystack = f"{path} {text.lower()}"
        score = sum(1 for kw in INTERESTING if kw in haystack)
        if score and path.count("/") <= 3:
            scored[clean] = max(scored.get(clean, 0), score)
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], len(kv[0])))
    return [u for u, _ in ranked[:limit]]


# --------------------------------------------------------------------------- fetching
async def fetch_html(client: httpx.AsyncClient, url: str, settings: Settings) -> tuple[str, str]:
    """GET a page with manual, re-validated redirects. Returns (final_url, html)."""
    current = url
    for _ in range(settings.max_redirects + 1):
        current = await validate_url(current)
        try:
            async with client.stream("GET", current, follow_redirects=False) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not location:
                        raise WebsiteFetchError("Website sent a redirect without a destination.")
                    current = urljoin(current, location)
                    continue
                if resp.status_code in (401, 403, 429):
                    raise WebsiteFetchError(
                        f"The website refused access (HTTP {resp.status_code}); it may block automated requests."
                    )
                if resp.status_code >= 400:
                    raise WebsiteFetchError(f"The website returned HTTP {resp.status_code}.")
                ctype = resp.headers.get("content-type", "").lower()
                if ctype and "html" not in ctype and "xml" not in ctype and "text/plain" not in ctype:
                    raise UnsupportedContentError(f"URL does not return an HTML page (content-type: {ctype.split(';')[0]}).")
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body.extend(chunk)
                    if len(body) >= settings.max_download_bytes:
                        break
                encoding = resp.charset_encoding or "utf-8"
                return current, bytes(body[: settings.max_download_bytes]).decode(encoding, errors="replace")
        except httpx.TimeoutException:
            raise WebsiteTimeoutError("The website took too long to respond.")
        except httpx.TooManyRedirects:
            raise WebsiteFetchError("Too many redirects.")
        except httpx.RequestError as exc:
            raise WebsiteFetchError(f"Could not connect to the website ({type(exc).__name__}).")
    raise WebsiteFetchError("Too many redirects.")


def build_prompt_text(pages: list[PageContent], settings: Settings) -> str:
    """Compact, budgeted text block handed to the LLM."""
    home = pages[0]
    parts: list[str] = [f"URL: {home.url}"]
    if home.title:
        parts.append(f"Title tag: {home.title}")
    if home.meta_description:
        parts.append(f"Meta description: {home.meta_description}")
    if home.site_name:
        parts.append(f"Site name (og:site_name): {home.site_name}")
    if home.structured_data:
        parts.append("Structured data: " + " | ".join(home.structured_data))
    if home.nav_items:
        parts.append("Navigation: " + " | ".join(home.nav_items))
    if home.headings:
        parts.append("Headings: " + " | ".join(home.headings[:20]))
    parts.append("--- Homepage text ---\n" + home.text[: settings.max_home_chars])
    for p in pages[1:]:
        header = f"--- Page: {p.url}" + (f" (title: {p.title})" if p.title else "") + " ---"
        parts.append(header + "\n" + p.text[: settings.max_extra_page_chars])
    return "\n".join(parts)


async def extract_website(url: str, settings: Settings) -> WebsiteContent:
    timeout = httpx.Timeout(settings.fetch_timeout_seconds)
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5", "Accept-Language": "en"}
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        final_url, html = await fetch_html(client, url, settings)
        home = parse_html(html, final_url)
        if len(home.text) < MIN_USEFUL_CHARS and not home.meta_description:
            raise UnsupportedContentError(
                "The page has almost no readable text. It may be rendered with JavaScript or blocked for bots."
            )

        async def _extra(u: str):
            try:
                f_url, f_html = await fetch_html(client, u, settings)
                return parse_html(f_html, f_url)
            except AppError as exc:  # extra pages are best-effort
                log.info("Skipping extra page %s: %s", u, exc.message)
                return None

        targets = select_internal_links(home, settings.max_extra_pages)
        extras = await asyncio.gather(*(_extra(u) for u in targets)) if targets else []
        pages = [home] + [p for p in extras if p and len(p.text) >= 100]

    content = WebsiteContent(final_url=final_url, pages=pages)
    content.prompt_text = build_prompt_text(pages, settings)
    return content
