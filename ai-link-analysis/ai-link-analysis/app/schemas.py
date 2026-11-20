"""Pydantic models: API request/response AND the contract the LLM output is validated against."""
from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field, field_validator


def _none_to_list(v):
    return [] if v is None else v


def _none_to_str(v):
    return "" if v is None else v


StrList = Annotated[list[str], BeforeValidator(_none_to_list)]
Str = Annotated[str, BeforeValidator(_none_to_str)]


class Platform(str, Enum):
    GOOGLE = "Google"
    FACEBOOK = "Facebook"
    INSTAGRAM = "Instagram"
    LINKEDIN = "LinkedIn"
    X = "X"


_PLATFORM_ALIASES = {
    "google": "Google", "google search": "Google",
    "facebook": "Facebook", "fb": "Facebook",
    "instagram": "Instagram", "ig": "Instagram",
    "linkedin": "LinkedIn", "linked in": "LinkedIn",
    "x": "X", "twitter": "X", "x (twitter)": "X", "x/twitter": "X", "x.com": "X",
}


# ----------------------------------------------------------------- request
class LinkAnalysisRequest(BaseModel):
    url: str = Field(..., min_length=3, max_length=2048, examples=["https://example.com"])


# ----------------------------------------------------------------- shared
class BrandInfo(BaseModel):
    name: str
    category: str
    summary: str


class ProfileComparison(BaseModel):
    """One row-set of the 'Your Brand vs Competitor' comparison table."""
    main_service: Str = ""
    target_market: Str = ""
    key_strength: Str = ""
    opportunity: Str = ""


# ----------------------------------------------------------------- section 1
class CompetitorVerification(BaseModel):
    reachable: bool
    checked_url: Optional[str] = None
    page_title: Optional[str] = None
    note: str = ""


class Competitor(BaseModel):
    name: str
    website: Optional[str] = None
    comparison: ProfileComparison = Field(default_factory=ProfileComparison)
    strengths: StrList = []
    weaknesses: StrList = []
    # Set by the service, not by the LLM:
    source: str = "ai_suggestion"
    verification: Optional[CompetitorVerification] = None


class CompetitorAnalysis(BaseModel):
    brand_overview: str
    main_products_services: StrList = []
    your_brand: ProfileComparison = Field(default_factory=ProfileComparison)
    competitors: list[Competitor] = []
    strengths: StrList = []          # the brand's competitive advantages
    opportunities: StrList = []      # ways to improve market positioning
    notes: Optional[str] = None      # e.g. why fewer than 3 competitors were listed


# ----------------------------------------------------------------- section 2
class SearchQuery(BaseModel):
    query: str
    platforms: list[Platform] = Field(min_length=1)
    intent: str
    business_relevance: Str = ""
    target_audience: Str = ""

    @field_validator("platforms", mode="before")
    @classmethod
    def _normalise_platforms(cls, v):
        if isinstance(v, str):
            v = [v]
        out = []
        for item in v or []:
            key = str(item).strip().lower()
            mapped = _PLATFORM_ALIASES.get(key, item)
            if mapped not in out:
                out.append(mapped)
        return out


# ----------------------------------------------------------------- section 3
class SupportingContent(BaseModel):
    type: str          # e.g. "FAQ", "Blog topic", "Service landing page", "Location page"
    topic: str
    rationale: Str = ""


class SEOImprovements(BaseModel):
    primary_keyword: str
    secondary_keywords: StrList = []
    long_tail_keywords: StrList = []
    business_keywords: StrList = []
    recommended_title: str
    title_structure: Str = ""
    meta_description: str
    meta_description_keywords: StrList = []
    supporting_content: list[SupportingContent] = []
    # Filled from the real page by the service (not by the LLM):
    current_title: Optional[str] = None
    current_meta_description: Optional[str] = None


# ----------------------------------------------------------------- section 4
class Service(BaseModel):
    name: str
    description: str
    target_industry: Str = ""
    target_customer: Str = ""
    main_benefit: Str = ""
    source: Literal["website", "inferred"] = "inferred"


# ----------------------------------------------------------------- section 5
class AudienceSegment(BaseModel):
    segment: str
    business_type: Str = ""
    requirement: Str = ""


class TargetAudience(BaseModel):
    primary_audience: StrList = []
    secondary_audience: StrList = []
    industries: StrList = []
    business_types: StrList = []
    customer_profiles: StrList = []
    geographic_focus: Optional[str] = None
    customer_needs: StrList = []
    customer_interests: StrList = []
    recommended_segments: list[AudienceSegment] = []


# ----------------------------------------------------------------- analysis container
def _trim_to_five(v):
    return v[:5] if isinstance(v, list) and len(v) > 5 else v


class Analysis(BaseModel):
    competitor_analysis: CompetitorAnalysis
    target_search_queries: list[SearchQuery] = Field(min_length=5, max_length=5)
    seo_improvements: SEOImprovements
    top_services: list[Service] = Field(min_length=5, max_length=5)
    target_audience: TargetAudience

    @field_validator("target_search_queries", "top_services", mode="before")
    @classmethod
    def _trim(cls, v):
        # LLMs sometimes return 6-7 items; keep the first five instead of failing.
        return _trim_to_five(v)


class LLMOutput(BaseModel):
    """Exactly what the LLM must return."""
    brand: BrandInfo
    analysis: Analysis


# ----------------------------------------------------------------- response
class AnalysisMeta(BaseModel):
    provider: str
    model: str
    cached: bool = False
    used_fallback: bool = False
    llm_attempts: int = 0
    total_ms: int = 0
    fetch_ms: int = 0
    llm_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: Optional[float] = Field(
        None, description="Measured tokens x list price. An estimate: prices may change; cached input discounts ignored."
    )
    pages_analyzed: list[str] = []
    content_chars_sent: int = 0
    notes: list[str] = []


class LinkAnalysisResponse(BaseModel):
    success: bool = True
    website: str
    brand: BrandInfo
    analysis: Analysis
    meta: AnalysisMeta


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorBody
