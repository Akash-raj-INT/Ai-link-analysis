"""Settings loaded from environment variables / .env. API keys never leave this module's Settings object."""
from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "openai", "gemini"]

# List prices in USD per 1M tokens: (input, output).
# Snapshot taken 2026-11-17 from provider pages and price trackers. ALWAYS re-verify before relying on them.
# Used only to turn *measured* token counts into an estimated cost.
PRICING_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),   # <=100K-token prompts; higher above that
    "claude-sonnet-5-5": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-sol": (2.00, 10.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Provider = "anthropic"
    llm_fallback_provider: Optional[Provider] = None

    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    anthropic_model: str = "claude-haiku-5-5"
    openai_model: str = "gpt-6-luna"
    gemini_model: str = "gemini-3.1-flash-lite"

    llm_timeout_seconds: float = 90
    llm_max_output_tokens: int = 4000
    # None = omit the parameter (some newer models reject custom temperature)
    llm_temperature: Optional[float] = None

    fetch_timeout_seconds: float = 15
    max_download_bytes: int = 2_000_000
    max_redirects: int = 5
    max_extra_pages: int = 2
    max_home_chars: int = 8000
    max_extra_page_chars: int = 3000

    verify_competitors: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 256

    def api_key_for(self, provider: str) -> Optional[str]:
        key = getattr(self, f"{provider}_api_key", None)
        return key.strip() if key and key.strip() else None

    def model_for(self, provider: str) -> str:
        return getattr(self, f"{provider}_model")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    price = PRICING_PER_M_TOKENS.get(model)
    if not price:
        return None
    return round((input_tokens * price[0] + output_tokens * price[1]) / 1_000_000, 6)
