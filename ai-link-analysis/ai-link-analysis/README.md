# AI Link Analysis API

A FastAPI service that takes a website URL, extracts the real page content, and uses an LLM to produce five structured analyses: **competitors vs your brand**, **top 5 target search queries**, **SEO improvements**, **top 5 services** and **target audience**.

- Python 3.12 · FastAPI · Uvicorn · Pydantic v2 · HTTPX · BeautifulSoup
- LLM providers (pick via `.env`): **Anthropic**, **OpenAI**, **Google Gemini**, with optional automatic fallback
- SSRF protection, graceful error envelopes, JSON validation with one automatic repair retry, in-memory cache
- 63 offline tests (no network, no API key needed)

---

## 1. Quick start

```bash
git clone <your-repo-url> && cd ai-link-analysis
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then put ONE API key in .env
uvicorn app.main:app --host 0.0.0.0 --port 4000
```

- API: `POST http://localhost:4000/link-analysis`
- Interactive docs: http://localhost:4000/docs

```bash
curl -X POST http://localhost:4000/link-analysis \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

Run the tests: `pytest -q` (fully offline, uses mocks).

### Configuration (`.env`)

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER` | `anthropic`, `openai` or `gemini` | `anthropic` |
| `LLM_FALLBACK_PROVIDER` | Used if the primary fails (needs its key) | none |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` | Credentials. **Never commit `.env`** (it is git-ignored) | none |
| `ANTHROPIC_MODEL` / `OPENAI_MODEL` / `GEMINI_MODEL` | Model IDs. Verify against the provider's current list | see `.env.example` |
| `MAX_EXTRA_PAGES`, `MAX_HOME_CHARS`, `MAX_EXTRA_PAGE_CHARS` | Content budget sent to the LLM (cost control) | 2 / 8000 / 3000 |
| `VERIFY_COMPETITORS` | Check that suggested competitor sites are reachable | `true` |
| `CACHE_TTL_SECONDS` | In-memory cache lifetime (`0` disables) | `3600` |

---

## 2. How it works

```
POST /link-analysis {"url"}
  1  validate_url        scheme/port/credentials checks + DNS resolution + private-IP rejection (SSRF)
  2  config check        fail fast (503) if no API key, before any fetching
  3  cache lookup        key = provider:model:url
  4  extract_website     fetch homepage (+ up to 2 relevant internal pages such as /about, /services)
                         -> strip scripts/styles/nav/footer, dedupe lines, keep title/meta/headings/JSON-LD, truncate to budget
  5  run_structured      one LLM call -> parse JSON -> validate with Pydantic -> one repair retry -> optional provider fallback
  6  competitor_service  drop the brand itself, check each suggested competitor site is reachable, label as "ai_suggestion"
  7  response            five sections + real current title/meta + meta block (tokens, latency, cost, pages used, notes)
```

### Project structure

```
app/
  main.py                      FastAPI app, routes, error handlers
  config.py                    Settings (.env), model price table, cost estimate
  schemas.py                   Request/response models AND the LLM output contract
  errors.py                    Typed errors -> HTTP status + stable error code
  services/
    website_extractor.py       Safe fetch (manual redirects, size cap) + HTML cleaning
    llm_service.py             Anthropic/OpenAI/Gemini clients, JSON repair, fallback, usage tracking
    analysis_service.py        Pipeline orchestration, prompts, cache
    competitor_service.py      Competitor post-processing and reachability verification
  utils/validators.py          URL validation + SSRF protection
tests/test_link_analysis.py    63 offline tests
scripts/benchmark.py           Measures real latency/tokens/cost per provider
```

### Design decisions

| Decision | Why |
|---|---|
| Plain HTTPS via `httpx`, no vendor SDKs | Fewer dependencies, one consistent code path, easy to read what is sent |
| Pydantic model as the LLM contract | Output is validated, not trusted. Common quirks are normalised ("Twitter" becomes "X", 6 queries trimmed to 5, `null` becomes `[]`) |
| One repair retry, then fail with 502 | Fixes most malformed JSON cheaply; avoids unbounded retry spend. Truncated output is reported, not retried |
| Website text wrapped in `<website_content>` and declared untrusted | Basic prompt-injection mitigation: the model is told never to follow instructions found in scraped pages |
| Competitors labelled `ai_suggestion` + reachability check | The brief says not to present unresearched competitors as fact. Only reachability is verified, and the response says so |
| No search volumes anywhere | None come from a real data source, so none are returned (stated in `meta.notes`) |
| `source: website \| inferred` on every service | Distinguishes what the page says from what the model inferred |
| Real `current_title` / `current_meta_description` returned beside the recommendations | Lets a reader compare "before" and "after" for the SEO section |
| Cost computed from **measured** token counts | `meta.estimated_cost_usd` = actual tokens x list price. Still an estimate (prices change, cache discounts ignored) |

### Security (SSRF and credentials)

- Only `http`/`https`, ports 80/443/8080/8443, no `user:pass@` URLs, no whitespace/control characters, 2048-char limit.
- The host is **resolved** and every address must be globally routable. This blocks loopback, RFC1918, link-local (cloud metadata `169.254.169.254`), CGNAT, multicast, IPv4-mapped IPv6, and encoded forms like `http://2130706433/`.
- Redirects are followed **manually** and each hop is re-validated, so a public site cannot redirect into the internal network.
- Downloads are capped at 2 MB, with timeouts and a redirect limit.
- API keys are read only from the environment; error messages never include them; `.env` is git-ignored.
- **Known limitation:** a DNS-rebinding attacker could change DNS between our check and the connection. Production hardening would pin the validated IP in a custom transport or fetch via an egress proxy that blocks internal ranges.

### Error responses

All errors use one envelope: `{"success": false, "error": {"code": "...", "message": "..."}}`.

| HTTP | `code` | Cause |
|---|---|---|
| 400 | `invalid_url` / `blocked_address` | Malformed URL, unsupported scheme, private/internal target |
| 422 | `invalid_request` / `unsupported_content` | Bad JSON body; page is not HTML or has almost no readable text (e.g. JS-only site) |
| 502 | `website_unreachable` / `llm_error` / `llm_invalid_output` | Site refused, 4xx/5xx, DNS failure; provider failure; unparseable model output |
| 503 | `llm_not_configured` | No API key set |
| 504 | `website_timeout` | Site too slow |

---

## 3. Sample request and response

> **Note:** the response below is a **hand-shortened illustration of the shape**, not captured output. Replace it (and add real screenshots/logs in `docs/`) with output from your own run before submitting.

```json
{
  "success": true,
  "website": "https://example-agency.com/",
  "brand": { "name": "Example Agency", "category": "Web Development", "summary": "..." },
  "analysis": {
    "competitor_analysis": {
      "brand_overview": "...",
      "main_products_services": ["..."],
      "your_brand": { "main_service": "...", "target_market": "...", "key_strength": "...", "opportunity": "..." },
      "competitors": [
        {
          "name": "...", "website": "competitor.com",
          "comparison": { "main_service": "...", "target_market": "...", "key_strength": "...", "opportunity": "..." },
          "strengths": ["..."], "weaknesses": ["..."],
          "source": "ai_suggestion",
          "verification": { "reachable": true, "checked_url": "https://competitor.com/", "page_title": "...", "note": "..." }
        }
      ],
      "strengths": ["..."], "opportunities": ["..."], "notes": null
    },
    "target_search_queries": [
      { "query": "...", "platforms": ["Google", "LinkedIn"], "intent": "Service Discovery", "business_relevance": "...", "target_audience": "..." }
    ],
    "seo_improvements": {
      "primary_keyword": "...", "secondary_keywords": [], "long_tail_keywords": [], "business_keywords": [],
      "recommended_title": "...", "title_structure": "...", "meta_description": "...",
      "supporting_content": [{ "type": "FAQ", "topic": "...", "rationale": "..." }],
      "current_title": "(real <title> from the page)", "current_meta_description": "(real meta description)"
    },
    "top_services": [
      { "name": "...", "description": "...", "target_industry": "...", "target_customer": "...", "main_benefit": "...", "source": "website" }
    ],
    "target_audience": {
      "primary_audience": [], "secondary_audience": [], "industries": [], "business_types": [], "customer_profiles": [],
      "geographic_focus": null, "customer_needs": [], "customer_interests": [],
      "recommended_segments": [{ "segment": "...", "business_type": "...", "requirement": "..." }]
    }
  },
  "meta": {
    "provider": "anthropic", "model": "claude-haiku-5-5", "cached": false, "used_fallback": false, "llm_attempts": 1,
    "total_ms": 0, "fetch_ms": 0, "llm_ms": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0,
    "pages_analyzed": ["https://example-agency.com/"], "content_chars_sent": 0,
    "notes": ["Competitors are AI-generated suggestions; ...", "No search-volume or ranking data is included; ..."]
  }
}
```

Error example (`POST` with `{"url": "http://169.254.169.254/latest/meta-data"}`), captured from the running server:

```json
{"success":false,"error":{"code":"blocked_address","message":"The URL resolves to a private or internal network address, which is not allowed."}}
```

---

## 4. LLM platform and model research

> Prices are **list prices in USD per 1M tokens**, collected on **2026-11-17** from provider pricing pages and third-party price trackers. Trackers sometimes disagree or lag, so **re-check the official pricing page before relying on any number**. Anything marked *estimate* is not measured.
