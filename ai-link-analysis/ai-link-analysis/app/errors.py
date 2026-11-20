"""Application errors. Each maps to an HTTP status and a stable machine-readable code."""


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidURLError(AppError):
    status_code = 400
    code = "invalid_url"


class BlockedAddressError(AppError):
    status_code = 400
    code = "blocked_address"


class WebsiteFetchError(AppError):
    status_code = 502
    code = "website_unreachable"


class WebsiteTimeoutError(AppError):
    status_code = 504
    code = "website_timeout"


class UnsupportedContentError(AppError):
    status_code = 422
    code = "unsupported_content"


class LLMConfigError(AppError):
    status_code = 503
    code = "llm_not_configured"


class LLMServiceError(AppError):
    status_code = 502
    code = "llm_error"


class LLMOutputError(AppError):
    status_code = 502
    code = "llm_invalid_output"
