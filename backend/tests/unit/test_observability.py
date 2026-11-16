from __future__ import annotations

import logging

import pytest

from app.observability import JsonFormatter, log_event, new_request_id


def test_request_id_is_generated_and_bounded() -> None:
    assert len(new_request_id("x" * 200)) == 128
    assert new_request_id("")


def test_json_formatter_contains_safe_event_fields() -> None:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "ignored", (), None)
    record.event = "query_done"
    record.fields = {"status_code": 200}

    rendered = JsonFormatter().format(record)

    assert '"event": "query_done"' in rendered
    assert '"status_code": 200' in rendered


def test_sensitive_log_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="sensitive"):
        log_event(logging.INFO, "bad", token="secret")
