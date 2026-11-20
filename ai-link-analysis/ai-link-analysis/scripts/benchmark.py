"""Measure real latency, token usage and cost per provider on real websites.

Usage (needs the relevant API keys in .env):
    python -m scripts.benchmark https://example.com https://another-site.com --providers anthropic openai gemini

Prints a Markdown table you can paste into the README and writes benchmark_results.json.
Accuracy is NOT scored automatically: read each output and score it yourself (see README rubric).
"""
import argparse
import asyncio
import json
import statistics

from app.config import get_settings
from app.errors import AppError
from app.services import analysis_service


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--providers", nargs="+", default=["anthropic", "openai", "gemini"])
    args = ap.parse_args()

    base = get_settings()
    rows, raw = [], []
    for provider in args.providers:
        if not base.api_key_for(provider):
            print(f"# skipping {provider}: no API key set")
            continue
        s = base.model_copy(update={"llm_provider": provider, "llm_fallback_provider": None, "cache_ttl_seconds": 0, "verify_competitors": False})
        results = []
        for url in args.urls:
            try:
                r = await analysis_service.analyze(url, s)
                results.append(r.meta)
                raw.append({"provider": provider, "url": url, "meta": r.meta.model_dump(), "output": r.model_dump(exclude={"meta"})})
            except AppError as exc:
                print(f"# {provider} {url}: FAILED - {exc.message}")
        if results:
            rows.append((provider, results[0].model, len(results), len(args.urls),
                         statistics.mean(m.llm_ms for m in results) / 1000,
                         statistics.mean(m.total_ms for m in results) / 1000,
                         statistics.mean(m.input_tokens for m in results),
                         statistics.mean(m.output_tokens for m in results),
                         statistics.mean(m.estimated_cost_usd or 0 for m in results)))

    print("\n| Provider | Model | OK/Total | Avg LLM time (s) | Avg total (s) | Avg input tok | Avg output tok | Avg cost (USD) |")
    print("|---|---|---|---|---|---|---|---|")
    for p, m, ok, tot, llm, total, i, o, c in rows:
        print(f"| {p} | {m} | {ok}/{tot} | {llm:.1f} | {total:.1f} | {i:.0f} | {o:.0f} | ${c:.5f} |")
    with open("benchmark_results.json", "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print("\nFull outputs saved to benchmark_results.json (gitignored - review them to score accuracy).")


if __name__ == "__main__":
    asyncio.run(main())
