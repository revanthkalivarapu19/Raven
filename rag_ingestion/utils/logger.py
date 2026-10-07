# src/utils/logger.py
"""Central logging utility for the project.

The logger supports:
* Console logging with a readable format
* File logging with rotation (10 MB per file, keep 5 backups)
* JSON-formatted log records for easy ingestion by monitoring tools

Usage::

    from rag_ingestion.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Message", extra={"key": "value"})
"""

import logging
import os
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Determine base log directory – can be overridden via env var LOG_PATH
LOG_DIR = Path(os.getenv("LOG_PATH", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_BYTES = 10 * 1024 * 1024  # 10 MiB
BACKUP_COUNT = 5

class JsonFormatter(logging.Formatter):
    """Serialize a LogRecord to a single-line JSON string."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_obj.update(record.extra)
        else:
            standard_attrs = {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            }
            extra_items = {k: v for k, v in record.__dict__.items() if k not in standard_attrs}
            log_obj.update(extra_items)
        return json.dumps(log_obj, ensure_ascii=False)

def _configure_logger(name: str, *, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        json_formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z")
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

    return logger

def get_logger(name: str = "project", *, log_file: Optional[Path] = None) -> logging.Logger:
    return _configure_logger(name, log_file=log_file)

# Specialized Loggers for Pipeline
def get_crawler_logger(name: str = "crawler") -> logging.Logger:
    return get_logger(name, log_file=LOG_DIR / "crawler.log")

def get_ingestion_logger(name: str = "ingestion") -> logging.Logger:
    return get_logger(name, log_file=LOG_DIR / "ingestion.log")

def get_duplicates_logger(name: str = "duplicates") -> logging.Logger:
    return get_logger(name, log_file=LOG_DIR / "duplicates.log")

def get_errors_logger(name: str = "errors") -> logging.Logger:
    # We set this one to capture errors
    logger = get_logger(name, log_file=LOG_DIR / "errors.log")
    return logger

default_logger = get_logger(log_file=LOG_DIR / "project.log")
