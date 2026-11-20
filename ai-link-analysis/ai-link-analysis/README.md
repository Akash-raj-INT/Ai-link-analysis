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

### A. Comparison

| Criteria | Anthropic **Claude Haiku 5.5** | OpenAI **GPT-6 Luna** | Google **Gemini 3.1 Flash-Lite** |
|---|---|---|---|
| API ID | `claude-haiku-5-5` | `gpt-6-luna` | `gemini-3.1-flash-lite` *(verify exact ID in AI Studio)* |
| Input / output price | $0.10 / $0.50 (prompts up to 100K tokens; $0.50 / $2.50 above) | $0.10 / $0.50 | $0.25 / $1.50 |
| Cached input | $0.01 | $0.01 | $0.025 |
| Batch price | $0.05 / $0.25 | $0.05 / $0.25 | not verified |
| Context window | 1M | ~1.05M | ~1M *(verify)* |
| Max output | not verified | 128K | not verified |
| Speed | **Not measured yet** | A third-party tracker lists ~109 tokens/s and ~0.9 s time-to-first-token (not measured by me) | **Not measured yet** |
| JSON support | Prompted JSON, validated + repaired in code | Native `json_object` mode | Native `responseMimeType: application/json` |
| Accuracy on this task | **Not tested yet** | **Not tested yet** | **Not tested yet** |

Other tiers I looked at but did not choose for the default: Claude Sonnet 5.5 ($2 / $10), GPT-6 Sol ($2 / $10), Gemini 3 Flash ($0.50 / $3). They cost 5 to 20x more, which is hard to justify for a summarisation-and-structuring job until a benchmark shows the cheap tier is inadequate.

Sources: platform.claude.com pricing page and release coverage (Haiku 5.5, Sonnet 5.5); developers.openai.com pricing via tracker pages (GPT-6 Luna); Google AI Studio / Vertex pricing pages and tracker sites (Gemini). Gemini figures were the least consistent across sources.

### B. Selection criteria for this application

1. **Reliable structured JSON** is mandatory (one malformed answer is a failed request).
2. **Cost per analysis**: the workload is "read about 5K tokens of text, write about 2.5K tokens of JSON", which needs reading comprehension and reasonable marketing knowledge, not frontier reasoning.
3. **Latency**: output tokens dominate response time, so the output size is kept bounded.
4. **Context**: irrelevant here. Inputs are capped near 5K tokens, far below every model's window (and below Haiku's 100K price step).
5. **Competitor knowledge**: depends on training data; a weakness of every small model (see mitigation in section 2).

### C. Cost optimisation (what this code already does)

| Technique | Implemented? | Where |
|---|---|---|
| Strip scripts, styles, svg, nav, footer, forms before the LLM | Yes | `parse_html` |
| Remove duplicate lines / collapse whitespace | Yes | `parse_html` |
| Hard character budget per page (8000 home, 3000 per extra page) | Yes | `build_prompt_text`, `.env` |
| Only 2 extra pages, chosen by relevance (`/about`, `/services` ...) | Yes | `select_internal_links` |
| Compact prompt: JSON skeleton instead of a long schema/examples | Yes | `analysis_service.py` |
| One LLM call per analysis (not one per section) | Yes | pipeline |
| Cheapest adequate model tier by default | Yes | `config.py` |
| Cache repeated URLs | Yes (in-memory, TTL) | `TTLCache`; use Redis if multi-instance |
| Cap on retries (one repair, one fallback) | Yes | `llm_service.py` |
| Fail before fetching when unconfigured | Yes | `analysis_service.analyze` |
| Prompt caching of the static system prompt + JSON skeleton | Not yet | The shared prefix is ~1K tokens; Anthropic and OpenAI list cached reads at 10% of input price. Worth it only at high volume |
| Batch API for non-urgent bulk jobs (50% off) | Not yet | Suitable for offline bulk analysis |
| Smaller model for extraction, larger for insights | Not yet | Possible two-stage design (see below) |

### D. Final recommendation (provisional until you run the benchmark)

| Item | Value |
|---|---|
| **Recommended provider / model** | Anthropic **Claude Haiku 5.5** (`claude-haiku-5-5`) |
| **Fallback** | OpenAI **GPT-6 Luna** (`gpt-6-luna`): same list price, different vendor, so a single-provider outage does not take the API down |
| Third option | Gemini 3.1 Flash-Lite, about 2.5x the price of the two above. Keep as a second fallback or if it wins your quality test |
| Est. input tokens / analysis | ~4,500 to 6,000 *(estimate: prompt + JSON skeleton ~1K, page text up to ~14K chars)* |
| Est. output tokens / analysis | ~2,000 to 3,000 *(estimate)* |
| Est. cost / analysis | ~**$0.0018** on Haiku 5.5 or Luna (5K in, 2.5K out) *(calculated from list prices)* |
| Same workload on other models | Gemini 3.1 Flash-Lite ~$0.005, Gemini 3 Flash ~$0.010, Sonnet 5.5 ~$0.035 |
| Cost per 1,000 analyses | ~$1.75 (Haiku/Luna), ~$5 (Gemini Flash-Lite), ~$35 (Sonnet 5.5) |
| Expected response time | **Estimate only: ~10 to 35 s** end to end, dominated by generating ~2.5K output tokens (fetching adds ~1 to 3 s). **Measure it** (below) |

**Why Haiku 5.5 first:** it is tied for the lowest list price of the three compared, has a very large context window, and is the default wired into this repository. **This is a cost-and-fit argument, not an accuracy claim.** I have not run a head-to-head quality test, and neither Haiku nor Luna is proven better. The decision rule is below.

**Balancing accuracy, speed and cost:** start on the cheapest tier. If the rubric scores show weak competitor or SEO quality, move *only that part* up a tier, for example a cheap model for extraction and a Sonnet-class model for competitor/SEO insight, rather than upgrading every call. Add the fallback for resilience, and add prompt caching/batching once volume justifies it.

### E. Measured vs estimated: fill this in from your own run

The brief asks to distinguish measured results from estimates, so run the benchmark on 5 to 10 diverse sites:

```bash
python -m scripts.benchmark https://site1.com https://site2.com https://site3.com --providers anthropic openai gemini
```

It prints a table of **measured** latency, token counts and cost and saves every full output to `benchmark_results.json`. Paste your results here:

| Provider | Model | Sites OK | Avg LLM time (s) | Avg input tok | Avg output tok | Avg cost (USD) |
|---|---|---|---|---|---|---|
| anthropic | claude-haiku-5-5 | _fill in_ | | | | |
| openai | gpt-6-luna | _fill in_ | | | | |
| gemini | gemini-3.1-flash-lite | _fill in_ | | | | |

**Accuracy rubric (manual, 1 to 5 each, per site):** (a) brand/category correct, (b) services match the page, (c) competitors real and relevant, (d) search queries realistic, (e) SEO title/meta usable, (f) JSON valid first try. Record the scores here and **switch the default model if another provider clearly wins**.

---

## 5. Likely interview questions (and short answers)

- **Why not just fetch the page and send raw HTML?** Raw HTML is mostly markup, so it wastes tokens and money and buries the content. Cleaning cuts input cost and improves focus.
- **How do you stop SSRF?** Resolve the hostname and reject any non-public IP, re-validate every redirect, cap size/time, restrict ports. Residual risk is DNS rebinding (see Security).
- **What if the LLM returns bad JSON?** Tolerant parsing (fences/prose), Pydantic validation, one repair prompt that includes the validation error, then a 502 or the fallback provider.
- **How do you avoid hallucinated competitors/volumes?** Prompt forbids invention and allows fewer than 3 competitors; competitors are labelled AI suggestions and reachability-checked; no volumes are returned.
- **Prompt injection from a malicious website?** Content is delimited and declared untrusted, and output is schema-validated. The model has no tools to abuse. This reduces risk but cannot guarantee it.
- **JavaScript-rendered sites?** Plain HTTP fetch sees little text, so the API returns a clear 422. A headless browser (Playwright) is the extension, at higher cost and with its own SSRF surface.
- **How would you scale it?** Move analysis to a background queue with job IDs (a 10 to 30 s synchronous call is a poor fit for web clients), Redis cache, per-IP rate limiting, prompt caching, observability.

## 6. Limitations

- Content comes from a plain HTTP fetch: no JavaScript rendering, and some sites block bots.
- Competitor suggestions depend on the model's training data; only reachability is checked, not market relevance.
- The cache is per-process memory; no authentication or rate limiting is included.
- Gemini model ID and some pricing details should be re-verified before use.
