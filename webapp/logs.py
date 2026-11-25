"""Structured logs, request ids and an access-log middleware.

Every line is one JSON object — ts, level, logger, request_id, msg, plus the
traceback when there is one — so a log shipper can parse it and an operator can
grep a whole request by the id the error page showed.

The middleware is plain ASGI, not Starlette's BaseHTTPMiddleware: the latter
wraps the response in its own task, which swallows the client-disconnect
cancellation the chat stream relies on to save a partial turn.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

# Set per request by AccessLogMiddleware and read by JsonFormatter, so a log
# line written deep inside a library still carries the request that caused it.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

REQUEST_ID_HEADER = "x-request-id"
_ID_OK = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# Fields logging puts on every record; anything else on the record is an
# intentional extra and belongs in the JSON line.
_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "module", "msecs",
    "message", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "stacklevel", "thread", "threadName",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            ) + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_") or key in payload:
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = repr(value)
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: Optional[str] = None) -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    root = logging.getLogger()
    wanted = (level or os.environ.get("SYSML_LOG_LEVEL") or "INFO").upper()
    root.setLevel(getattr(logging, wanted, logging.INFO))
    handler = next(
        (h for h in root.handlers if getattr(h, "_sysml_json", False)), None
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler._sysml_json = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    handler.setFormatter(JsonFormatter())


def request_id_from(scope: Any) -> str:
    """Reuse a sane X-Request-Id from the proxy, else mint one."""
    for name, value in scope.get("headers") or []:
        if name.decode("latin-1").lower() == REQUEST_ID_HEADER:
            candidate = value.decode("latin-1").strip()
            if _ID_OK.match(candidate):
                return candidate
    return uuid.uuid4().hex


class AccessLogMiddleware:
    """Mint/propagate X-Request-Id and log one line per request.

    Kept to ~the minimum of the ASGI protocol: it wraps ``send`` to learn the
    status code, then logs method, path, status and duration when the response
    ends. Nothing is buffered, so SSE keeps streaming through it.
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self.logger = logging.getLogger("webapp.access")

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = request_id_from(scope)
        token = request_id_var.set(request_id)
        # Mirror it onto the scope: the exception handler runs *after* this
        # middleware unwinds (and resets the contextvar), so the ambient value is
        # gone precisely when a 500 needs to report it.
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status = 500

        async def send_wrapper(message: Dict[str, Any]) -> None:
            nonlocal status
            if message.get("type") == "http.response.start":
                status = int(message.get("status", 500))
                headers = list(message.get("headers") or [])
                headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            self.logger.info(
                "%s %s -> %s",
                scope.get("method", "-"),
                scope.get("path", "-"),
                status,
                extra={
                    # Carried on the record, not only in the contextvar: the
                    # record is formatted after this request has finished.
                    "request_id": request_id,
                    "status": status,
                    "duration_ms": elapsed_ms,
                    "method": scope.get("method", "-"),
                    "path": scope.get("path", "-"),
                    "client": (scope.get("client") or ("-", 0))[0],
                },
            )
            request_id_var.reset(token)
