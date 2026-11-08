"""Structured logging.

``LOG_FORMAT=json`` 输出单行 JSON，便于采集到 Loki/ELK；默认 text 保持人读友好。
两条日志都会带上当前请求的 request_id。
"""

import json
import logging
import sys

from app.core.config import settings
from app.core.request_context import RequestContextFilter

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"request_id", "message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    if settings.log_format.lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] [%(name)s] %(message)s"))
    handler.addFilter(RequestContextFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
