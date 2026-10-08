"""Request observability primitives for the production inference API."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

LOGGER = logging.getLogger("mlforge.serving")
REQUEST_ID_HEADER = "X-Request-ID"


def configure_logging(level: str) -> None:
    """Configure service logging without recording request payloads."""
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))


def _request_id(request: Request) -> str:
    supplied = request.headers.get(REQUEST_ID_HEADER, "").strip()
    if supplied and len(supplied) <= 128:
        return supplied
    return str(uuid.uuid4())


def _log_event(**fields: object) -> None:
    LOGGER.info(json.dumps(fields, separators=(",", ":"), sort_keys=True))


class RequestObservabilityMiddleware(BaseHTTPMiddleware):
    """Attach correlation IDs and emit payload-free structured request logs."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = _request_id(request)
        request.state.request_id = request_id
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            _log_event(
                event="http_request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                latency_ms=latency_ms,
            )
            if "response" in locals():
                response.headers[REQUEST_ID_HEADER] = request_id
