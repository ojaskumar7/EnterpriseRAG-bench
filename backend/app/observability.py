"""Safe structured logging and request correlation."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")
logger = logging.getLogger("enterprise_rag")
FORBIDDEN_FIELDS = {
    "answer",
    "authorization",
    "content",
    "password",
    "prompt",
    "question",
    "secret",
    "token",
}


class JsonFormatter(logging.Formatter):
    """Render allow-listed event fields as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "request_id": REQUEST_ID.get(),
        }
        fields = getattr(record, "fields", {})
        if isinstance(fields, dict):
            payload.update(fields)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "event", record.getMessage())
        fields = getattr(record, "fields", {})
        suffix = f" {fields}" if fields else ""
        return f"{record.levelname} {event} request_id={REQUEST_ID.get()}{suffix}"


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else TextFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def log_event(level: int, event: str, **fields: Any) -> None:
    """Emit only caller-selected fields; sensitive payloads are not accepted by convention."""

    forbidden = FORBIDDEN_FIELDS.intersection(fields)
    if forbidden:
        raise ValueError(f"sensitive log fields are forbidden: {sorted(forbidden)}")
    logger.log(level, event, extra={"event": event, "fields": fields})


def new_request_id(value: str | None = None) -> str:
    candidate = (value or "").strip()
    return candidate[:128] if candidate else str(uuid.uuid4())


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)
