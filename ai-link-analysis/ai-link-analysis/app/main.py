"""FastAPI entrypoint. Run:  uvicorn app.main:app --host 0.0.0.0 --port 4000"""
import logging

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.errors import AppError
from app.schemas import ErrorResponse, LinkAnalysisRequest, LinkAnalysisResponse
from app.services import analysis_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = FastAPI(
    title="AI Link Analysis API",
    version="1.0.0",
    description="Analyses a website with an LLM: competitors, search queries, SEO, top services and target audience.",
)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"success": False, "error": {"code": code, "message": message}})


@app.exception_handler(AppError)
async def handle_app_error(_: Request, exc: AppError):
    return _error(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError):
    problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
    return _error(422, "invalid_request", f"Invalid request body - {problems}")


@app.exception_handler(Exception)
async def handle_unexpected(_: Request, exc: Exception):
    log.exception("Unhandled error")  # details stay in the server log, not in the response
    return _error(500, "internal_error", "An unexpected error occurred.")


@app.post(
    "/link-analysis",
    response_model=LinkAnalysisResponse,
    responses={code: {"model": ErrorResponse} for code in (400, 422, 502, 503, 504)},
    summary="Analyse a website URL",
)
async def link_analysis(body: LinkAnalysisRequest, settings: Settings = Depends(get_settings)):
    return await analysis_service.analyze(body.url, settings)


@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}
