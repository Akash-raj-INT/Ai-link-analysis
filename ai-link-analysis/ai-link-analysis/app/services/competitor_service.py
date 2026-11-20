"""Competitor post-processing.

The LLM proposes competitors from its own knowledge, so they are labelled `ai_suggestion`.
We add light, real verification: does the suggested website resolve to a public address, respond, and what is its
page title? Unreachable suggestions are kept but flagged, and the brand itself is removed if the LLM listed it.
"""
import asyncio
import logging
from urllib.parse import urlsplit

import httpx

from app.config import Settings
from app.errors import AppError
from app.schemas import Competitor, CompetitorVerification
from app.services.website_extractor import USER_AGENT, fetch_html, parse_html
from app.utils.validators import normalize_url

log = logging.getLogger(__name__)


def _host(url: str) -> str:
    try:
        return (urlsplit(normalize_url(url)).hostname or "").removeprefix("www.")
    except AppError:
        return ""


async def _verify_one(client: httpx.AsyncClient, comp: Competitor, settings: Settings) -> None:
    if not comp.website:
        comp.verification = CompetitorVerification(reachable=False, note="No website provided by the model.")
        return
    try:
        url, html = await fetch_html(client, normalize_url(comp.website), settings)
        title = parse_html(html, url).title or None
        comp.verification = CompetitorVerification(
            reachable=True, checked_url=url, page_title=title, note="Website is reachable. Competitor choice itself is an AI suggestion."
        )
    except AppError as exc:
        comp.verification = CompetitorVerification(reachable=False, note=exc.message)


async def verify_competitors(competitors: list[Competitor], own_url: str, settings: Settings) -> list[Competitor]:
    own_host = _host(own_url)
    kept = [c for c in competitors if not (c.website and _host(c.website) == own_host)][:5]
    for c in kept:
        c.source = "ai_suggestion"
    if not settings.verify_competitors or not kept:
        return kept
    timeout = httpx.Timeout(min(settings.fetch_timeout_seconds, 8))
    async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
        await asyncio.gather(*(_verify_one(client, c, settings) for c in kept))
    return kept
