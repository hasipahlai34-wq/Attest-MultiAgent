"""Unified error contract.

Every failure returns the same envelope so the frontend and the audit trail can
branch on a stable machine code instead of parsing prose::

    {"detail": "<human text>", "error": {"code": "...", "message": "...",
     "request_id": "...", "details": {...}}}

``detail`` is kept for FastAPI/前端兼容；``error`` 是新增的稳定结构。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.request_context import current_request_id

logger = logging.getLogger("attest.error")


class ErrorCode:
    AUTH_REQUIRED = "AUTH_REQUIRED"
    INVALID_CREDENTIAL = "INVALID_CREDENTIAL"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    RATE_LIMITED = "RATE_LIMITED"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    INTERNAL = "INTERNAL"


class AppError(Exception):
    """业务异常：携带稳定错误码，由全局处理器转成统一响应。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def _envelope(code: str, message: str, details: dict | None = None) -> dict:
    return {
        "detail": message,
        "error": {
            "code": code,
            "message": message,
            "request_id": current_request_id(),
            "details": details or {},
        },
    }


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=_envelope(exc.code, exc.message, exc.details))

    @app.exception_handler(HTTPException)
    async def http_error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        details = {} if isinstance(exc.detail, str) else {"raw": exc.detail}
        code = {
            401: ErrorCode.AUTH_REQUIRED,
            403: ErrorCode.FORBIDDEN,
            404: ErrorCode.NOT_FOUND,
            409: ErrorCode.CONFLICT,
            429: ErrorCode.RATE_LIMITED,
        }.get(exc.status_code, ErrorCode.INTERNAL if exc.status_code >= 500 else ErrorCode.VALIDATION_FAILED)
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(code, detail, details),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_envelope(ErrorCode.VALIDATION_FAILED, "请求参数校验失败", {"errors": exc.errors()}),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content=_envelope(ErrorCode.INTERNAL, "服务内部错误"))
