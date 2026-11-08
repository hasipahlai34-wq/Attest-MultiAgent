"""Request-scoped tracing.

One request id is generated per HTTP call and flows into every log line, error
payload and audit row, so a failed agent run can be reconstructed from the
server log alone. ``X-Request-ID`` is honoured when the caller already has one.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar

from fastapi import FastAPI, Request

REQUEST_ID_HEADER = "X-Request-ID"
_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_access_logger = logging.getLogger("attest.access")


def current_request_id() -> str:
    return _request_id.get()


class RequestContextFilter(logging.Filter):
    """Injects the active request id into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def install_request_context(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = (request.headers.get(REQUEST_ID_HEADER) or "").strip()[:64]
        request_id = incoming or uuid.uuid4().hex
        token = _request_id.set(request_id)
        request.state.request_id = request_id
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            duration_ms = int((time.perf_counter() - started) * 1000)
            _access_logger.info(
                "%s %s -> %s",
                request.method,
                request.url.path,
                status_code,
                extra={"method": request.method, "path": request.url.path, "status": status_code, "duration_ms": duration_ms},
            )
            _request_id.reset(token)
