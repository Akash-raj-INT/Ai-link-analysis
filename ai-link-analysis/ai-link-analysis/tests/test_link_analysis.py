"""Offline test-suite: no network, no real LLM. Run with:  pytest -q"""
import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, estimate_cost_usd, get_settings
from app.errors import (
    BlockedAddressError, InvalidURLError, LLMOutputError, LLMServiceError, WebsiteFetchError,
)
from app.main import app
from app.schemas import LLMOutput
from app.services import analysis_service, competitor_service, llm_service
from app.services.llm_service import (
    AnthropicProvider, GeminiProvider, LLMUsage, OpenAIProvider, RawCompletion, extract_json,
)
from app.services.website_extractor import (
    PageContent, WebsiteContent, build_prompt_text, parse_html, select_internal_links,
)
from app.utils import validators


def run(coro):
    return asyncio.run(coro)


def settings(**kw) -> Settings:
    base = dict(anthropic_api_key="test-key", cache_ttl_seconds=0, verify_competitors=False, _env_file=None)
    base.update(kw)
    return Settings(**base)


# ------------------------------------------------------------------ fixtures
HTML = """<html><head><title>  Acme Web Studio | Websites for small business </title>
<meta name="description" content="We build fast, affordable websites for small businesses.">
<meta property="og:site_name" content="Acme Web Studio">
<script type="application/ld+json">{"@type":"Organization","name":"Acme Web Studio","description":"Web agency"}</script>
<style>.x{color:red}</style><script>var secret = 1;</script></head>
<body><nav><a href="/services">Services</a><a href="/about">About us</a><a href="/blog/post-1">Blog</a></nav>
<h1>Websites that grow your business</h1>
<p>We design and develop responsive websites, e-commerce stores and SEO services for small businesses.</p>
<p>We design and develop responsive websites, e-commerce stores and SEO services for small businesses.</p>
<h2>Our services</h2><p>Custom web development, online stores, search engine optimisation and hosting.</p>
<footer>Copyright Acme</footer></body></html>"""

VALID_OUTPUT = {
    "brand": {"name": "Acme Web Studio", "category": "Web Development", "summary": "A web agency for small businesses."},
    "analysis": {
        "competitor_analysis": {
            "brand_overview": "Acme builds websites.",
            "main_products_services": ["Web development"],
            "your_brand": {"main_service": "Web development", "target_market": "Small business", "key_strength": "Affordable", "opportunity": "Visibility"},
            "competitors": [
                {"name": "Wix", "website": "wix.com", "comparison": {"main_service": "Site builder"}, "strengths": ["Brand"], "weaknesses": ["Generic"]},
                {"name": "Self", "website": "acme.com", "strengths": [], "weaknesses": []},
            ],
            "strengths": ["Price"], "opportunities": ["Local SEO"], "notes": None,
        },
        "target_search_queries": [
            {"query": f"web design {i}", "platforms": ["Google", "x (twitter)"], "intent": "Service Discovery"} for i in range(6)
        ],
        "seo_improvements": {
            "primary_keyword": "web development", "recommended_title": "Web Development for Small Business | Acme",
            "meta_description": "Affordable websites.", "supporting_content": [{"type": "FAQ", "topic": "How much is a website?"}],
        },
        "top_services": [
            {"name": f"Service {i}", "description": "d", "source": "website"} for i in range(5)
        ],
        "target_audience": {"primary_audience": ["Small businesses"], "geographic_focus": None},
    },
}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    analysis_service.cache.clear()
    app.dependency_overrides[get_settings] = lambda: settings()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def public_dns(monkeypatch):
    async def fake(host, port):
        return ["93.184.216.34"]
    monkeypatch.setattr(validators, "_resolve", fake)


# ------------------------------------------------------------------ SSRF / URL validation
@pytest.mark.parametrize("url", [
    "http://localhost", "http://127.0.0.1", "http://127.0.0.1:8080/admin", "http://10.0.0.5", "http://192.168.1.1",
    "http://172.16.0.1", "http://169.254.169.254/latest/meta-data/", "http://[::1]/", "http://0.0.0.0",
    "http://2130706433/", "http://100.64.0.1", "http://[::ffff:127.0.0.1]/", "http://foo.internal", "http://printer.local",
])
def test_private_addresses_blocked(url):
    with pytest.raises((BlockedAddressError, InvalidURLError)):
        run(validators.validate_url(url))


@pytest.mark.parametrize("url", ["", "   ", "ftp://example.com", "file:///etc/passwd", "javascript:alert(1)",
                                 "http://user:pass@example.com", "http://exa mple.com", "http://intranet",
                                 "https://example.com:22", "http://"])
def test_invalid_urls_rejected(url, public_dns):
    with pytest.raises((InvalidURLError, BlockedAddressError)):
        run(validators.validate_url(url))


def test_public_domain_resolving_to_private_ip_blocked(monkeypatch):
    async def fake(host, port):
        return ["93.184.216.34", "10.0.0.8"]   # one bad record is enough
    monkeypatch.setattr(validators, "_resolve", fake)
    with pytest.raises(BlockedAddressError):
        run(validators.validate_url("https://evil.example.com"))


def test_valid_url_normalised(public_dns):
    assert run(validators.validate_url("Example.com/path#frag")) == "https://Example.com/path"
    assert run(validators.validate_url("https://example.com")) == "https://example.com/"


def test_unresolvable_domain(monkeypatch):
    async def fake(host, port):
        raise WebsiteFetchError("Could not resolve")
    monkeypatch.setattr(validators, "_resolve", fake)
    with pytest.raises(WebsiteFetchError):
        run(validators.validate_url("https://no-such-domain-xyz.com"))


# ------------------------------------------------------------------ extraction
def test_parse_html_extracts_and_cleans():
    page = parse_html(HTML, "https://acme.com/")
    assert page.title == "Acme Web Studio | Websites for small business"
    assert "affordable websites" in page.meta_description
    assert page.site_name == "Acme Web Studio"
    assert any("Organization" in s for s in page.structured_data)
    assert page.nav_items == ["Services", "About us", "Blog"]
    assert "Websites that grow your business" in page.headings
    assert "secret" not in page.text and "color:red" not in page.text and "Copyright" not in page.text
    assert page.text.lower().count("we design and develop responsive websites") == 1   # de-duplicated


def test_select_internal_links_prefers_service_pages():
    page = parse_html(HTML, "https://acme.com/")
    links = select_internal_links(page, 2)
    assert set(links) == {"https://acme.com/services", "https://acme.com/about"}   # blog is not selected
    assert select_internal_links(page, 0) == []


def test_prompt_text_respects_budget():
    p = PageContent(url="https://a.com/", title="T", text="x" * 50000)
    extra = PageContent(url="https://a.com/about", text="y" * 50000)
    text = build_prompt_text([p, extra], settings(max_home_chars=1000, max_extra_page_chars=500))
    assert "x" * 1000 in text and "x" * 1001 not in text
    assert "y" * 500 in text and "y" * 501 not in text


# ------------------------------------------------------------------ JSON + schema
def test_extract_json_variants():
    assert extract_json('{"a":1}') == {"a": 1}
    assert extract_json('```json\n{"a":1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": {"b": 2}} thanks') == {"a": {"b": 2}}
    with pytest.raises(json.JSONDecodeError):
        extract_json("no json here")


def test_schema_normalises_llm_quirks():
    out = LLMOutput.model_validate(VALID_OUTPUT)
    assert len(out.analysis.target_search_queries) == 5            # 6 trimmed to 5
    assert [p.value for p in out.analysis.target_search_queries[0].platforms] == ["Google", "X"]  # alias mapped


def test_schema_rejects_too_few_services():
    bad = json.loads(json.dumps(VALID_OUTPUT))
    bad["analysis"]["top_services"] = bad["analysis"]["top_services"][:3]
    with pytest.raises(Exception):
        LLMOutput.model_validate(bad)


def test_cost_estimate():
    assert estimate_cost_usd("claude-haiku-5-5", 5000, 2500) == pytest.approx(0.00175)
    assert estimate_cost_usd("unknown-model", 1, 1) is None


# ------------------------------------------------------------------ LLM providers (mock HTTP)
def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_anthropic_provider_parses_response():
    def handler(req: httpx.Request):
        body = json.loads(req.content)
        assert req.headers["x-api-key"] == "k" and body["model"] == "claude-haiku-5-5" and body["system"] == "sys"
        return httpx.Response(200, json={"content": [{"type": "text", "text": '{"ok":1}'}], "usage": {"input_tokens": 11, "output_tokens": 7}, "stop_reason": "end_turn"})
    async def go():
        async with _client(handler) as c:
            return await AnthropicProvider("claude-haiku-5-5", "k", settings()).complete(c, "sys", [{"role": "user", "content": "hi"}])
    r = run(go())
    assert (r.text, r.input_tokens, r.output_tokens, r.truncated) == ('{"ok":1}', 11, 7, False)


def test_openai_provider_parses_response():
    def handler(req):
        body = json.loads(req.content)
        assert req.headers["authorization"] == "Bearer k" and body["messages"][0]["role"] == "system"
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}], "usage": {"prompt_tokens": 5, "completion_tokens": 9}})
    async def go():
        async with _client(handler) as c:
            return await OpenAIProvider("gpt-6-luna", "k", settings()).complete(c, "s", [{"role": "user", "content": "x"}])
    r = run(go())
    assert (r.input_tokens, r.output_tokens, r.truncated) == (5, 9, True)


def test_gemini_provider_counts_thinking_tokens():
    def handler(req):
        body = json.loads(req.content)
        assert req.headers["x-goog-api-key"] == "k" and body["generationConfig"]["responseMimeType"] == "application/json"
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}],
                                         "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 40, "thoughtsTokenCount": 10}})
    async def go():
        async with _client(handler) as c:
            return await GeminiProvider("gemini-3.1-flash-lite", "k", settings()).complete(c, "s", [{"role": "user", "content": "x"}])
    r = run(go())
    assert (r.input_tokens, r.output_tokens) == (100, 50)


@pytest.mark.parametrize("status,needle", [(401, "rejected the API key"), (400, "rejected the request")])
def test_provider_http_errors_are_clean(status, needle):
    async def go():
        async with _client(lambda req: httpx.Response(status, json={"error": {"message": "bad"}})) as c:
            await AnthropicProvider("m", "super-secret-key", settings()).complete(c, "s", [{"role": "user", "content": "x"}])
    with pytest.raises(LLMServiceError) as e:
        run(go())
    assert needle in e.value.message and "super-secret-key" not in e.value.message


# ------------------------------------------------------------------ repair + fallback
class FakeProvider:
    def __init__(self, name, replies):
        self.name, self.model, self.replies, self.calls = name, f"{name}-model", list(replies), 0

    async def complete(self, client, system, messages):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return RawCompletion(r, 10, 20)


def test_repair_flow_fixes_invalid_json():
    p = FakeProvider("fake", ["this is not json", json.dumps(VALID_OUTPUT)])
    usage = LLMUsage("fake", "m")
    out = run(llm_service._structured_with_repair(p, None, "s", "u", LLMOutput, usage))
    assert out.brand.name == "Acme Web Studio" and usage.attempts == 2 and usage.input_tokens == 20


def test_repair_gives_up_after_second_failure():
    p = FakeProvider("fake", ["nope", "still nope"])
    with pytest.raises(LLMOutputError):
        run(llm_service._structured_with_repair(p, None, "s", "u", LLMOutput, LLMUsage("fake", "m")))


def test_truncated_output_is_reported_not_retried():
    class Trunc(FakeProvider):
        async def complete(self, client, system, messages):
            self.calls += 1
            return RawCompletion('{"brand":', 1, 1, truncated=True)
    p = Trunc("fake", [])
    with pytest.raises(LLMOutputError, match="cut off"):
        run(llm_service._structured_with_repair(p, None, "s", "u", LLMOutput, LLMUsage("fake", "m")))
    assert p.calls == 1


def test_fallback_provider_used_when_primary_fails(monkeypatch):
    primary = FakeProvider("anthropic", [LLMServiceError("down")])
    backup = FakeProvider("openai", [json.dumps(VALID_OUTPUT)])
    monkeypatch.setattr(llm_service, "configured_providers", lambda s: [primary, backup])
    out, usage = run(llm_service.run_structured("s", "u", LLMOutput, settings()))
    assert usage.used_fallback and usage.provider == "openai" and out.brand.name


def test_missing_api_key_raises_config_error():
    with pytest.raises(llm_service.LLMConfigError):
        llm_service.configured_providers(Settings(_env_file=None, anthropic_api_key=""))


# ------------------------------------------------------------------ competitors
def test_competitor_service_drops_self_and_marks_unreachable(monkeypatch, public_dns):
    out = LLMOutput.model_validate(VALID_OUTPUT)
    comps = out.analysis.competitor_analysis.competitors

    async def fake_fetch(client, url, s):
        if "wix" in url:
            return url, "<html><title>Wix.com</title></html>"
        raise WebsiteFetchError("down")
    monkeypatch.setattr(competitor_service, "fetch_html", fake_fetch)
    kept = run(competitor_service.verify_competitors(comps, "https://www.acme.com/", settings(verify_competitors=True)))
    assert [c.name for c in kept] == ["Wix"]                      # "acme.com" removed (it is the brand itself)
    assert kept[0].verification.reachable and kept[0].verification.page_title == "Wix.com"
    assert kept[0].source == "ai_suggestion"


# ------------------------------------------------------------------ endpoint
@pytest.fixture
def pipeline(monkeypatch, public_dns):
    async def fake_extract(url, s):
        home = parse_html(HTML, url)
        c = WebsiteContent(final_url=url, pages=[home])
        c.prompt_text = build_prompt_text(c.pages, s)
        return c

    async def fake_llm(system, user, model_cls, s, client=None):
        assert "<website_content>" in user and "Acme Web Studio" in user     # real content reaches the prompt
        return model_cls.model_validate(VALID_OUTPUT), LLMUsage("anthropic", "claude-haiku-5-5", 4000, 2000, 1234, 1)

    monkeypatch.setattr(analysis_service, "extract_website", fake_extract)
    monkeypatch.setattr(analysis_service, "run_structured", fake_llm)


def test_endpoint_success(pipeline):
    r = TestClient(app).post("/link-analysis", json={"url": "https://acme.com"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True and body["website"] == "https://acme.com/"
    assert body["brand"]["name"] == "Acme Web Studio"
    a = body["analysis"]
    assert len(a["target_search_queries"]) == 5 and len(a["top_services"]) == 5
    assert a["seo_improvements"]["current_title"] == "Acme Web Studio | Websites for small business"
    assert body["meta"]["estimated_cost_usd"] == pytest.approx(0.0014)
    assert body["meta"]["cached"] is False
    assert "search-volume" in " ".join(body["meta"]["notes"])


def test_endpoint_cache_hit(pipeline):
    app.dependency_overrides[get_settings] = lambda: settings(cache_ttl_seconds=60)
    c = TestClient(app)
    assert c.post("/link-analysis", json={"url": "https://acme.com"}).json()["meta"]["cached"] is False
    assert c.post("/link-analysis", json={"url": "https://acme.com"}).json()["meta"]["cached"] is True


@pytest.mark.parametrize("payload,status,code", [
    ({"url": "http://127.0.0.1"}, 400, "blocked_address"),
    ({"url": "http://169.254.169.254/latest/meta-data"}, 400, "blocked_address"),
    ({"url": "ftp://example.com"}, 400, "invalid_url"),
    ({"url": "not a url"}, 400, "invalid_url"),
    ({}, 422, "invalid_request"),
    ({"url": 123}, 422, "invalid_request"),
])
def test_endpoint_rejects_bad_input(payload, status, code):
    r = TestClient(app).post("/link-analysis", json=payload)
    assert r.status_code == status and r.json()["error"]["code"] == code and r.json()["success"] is False


def test_endpoint_website_unreachable(monkeypatch, public_dns):
    async def boom(url, s):
        raise WebsiteFetchError("The website returned HTTP 404.")
    monkeypatch.setattr(analysis_service, "extract_website", boom)
    r = TestClient(app).post("/link-analysis", json={"url": "https://example.com"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "website_unreachable"


def test_endpoint_llm_not_configured(public_dns):
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, anthropic_api_key="")
    r = TestClient(app).post("/link-analysis", json={"url": "https://example.com"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "llm_not_configured"


def test_endpoint_unexpected_error_hides_details(monkeypatch, public_dns):
    async def boom(url, s):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(analysis_service, "extract_website", boom)
    r = TestClient(app, raise_server_exceptions=False).post("/link-analysis", json={"url": "https://example.com"})
    assert r.status_code == 500 and "secret" not in r.text


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


# ------------------------------------------------------------------ fetcher (mock transport, real code path)
from app.errors import UnsupportedContentError, WebsiteTimeoutError  # noqa: E402
from app.services.website_extractor import fetch_html  # noqa: E402


def _fetch(handler, url="https://example.com/"):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await fetch_html(c, url, settings())
    return run(go())


def test_fetch_follows_safe_redirect(public_dns):
    def handler(req):
        if req.url.path == "/":
            return httpx.Response(301, headers={"location": "/home"})
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text="<html>ok</html>")
    final, html = _fetch(handler)
    assert final == "https://example.com/home" and "ok" in html


def test_fetch_blocks_redirect_to_internal_address(monkeypatch):
    async def fake(host, port):
        return ["169.254.169.254"] if host == "169.254.169.254" else ["93.184.216.34"]
    monkeypatch.setattr(validators, "_resolve", fake)
    handler = lambda req: httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})
    with pytest.raises(BlockedAddressError):
        _fetch(handler)


def test_fetch_redirect_loop(public_dns):
    with pytest.raises(WebsiteFetchError, match="redirect"):
        _fetch(lambda req: httpx.Response(302, headers={"location": "/again"}))


def test_fetch_http_errors_and_content_types(public_dns):
    with pytest.raises(WebsiteFetchError, match="refused access"):
        _fetch(lambda req: httpx.Response(403))
    with pytest.raises(WebsiteFetchError, match="404"):
        _fetch(lambda req: httpx.Response(404))
    with pytest.raises(UnsupportedContentError):
        _fetch(lambda req: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF"))


def test_fetch_timeout(public_dns):
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)
    with pytest.raises(WebsiteTimeoutError):
        _fetch(handler)


def test_fetch_caps_download_size(public_dns):
    big = b"<html>" + b"a" * 5_000_000
    final, html = _fetch(lambda req: httpx.Response(200, headers={"content-type": "text/html"}, content=big))
    assert len(html) <= 2_000_000
