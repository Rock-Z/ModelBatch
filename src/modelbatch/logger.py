"""
General-purpose logging system with environment-based configuration.

Features:
- Environment variable configuration
- Console and file output
- Structured JSON logging
- Context managers for timing and custom contexts
- Logger hierarchy support
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import time
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields from record
        log_data.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key
                not in [
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "message",
                ]
            }
        )

        return json.dumps(log_data, default=str)


class ContextLogger:
    """Logger wrapper with context management capabilities."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._context: dict[str, Any] = {}

    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log message with current context."""
        # Check if logging is enabled for this level first
        if not self._logger.isEnabledFor(level):
            return

        # Merge context into extra
        extra = kwargs.get("extra", {})
        extra.update(self._context)
        kwargs["extra"] = extra
        self._logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)

    def isEnabledFor(self, level: int) -> bool:  # noqa: N802
        return self._logger.isEnabledFor(level)

    @contextmanager
    def context(self, **ctx_vars):
        """Add context variables for nested logging."""
        old_context = self._context.copy()
        self._context.update(ctx_vars)
        try:
            yield
        finally:
            self._context = old_context

    @contextmanager
    def timer(self, operation: str, log_level: int = logging.INFO):
        """Time an operation and log the duration."""
        start_time = time.time()
        self._log_with_context(log_level, f"Starting {operation}")
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._log_with_context(log_level, f"Completed {operation}",
                                 extra={"operation": operation, "duration_seconds": duration})


class LoggerManager:
    """Manages logger configuration and creation."""

    def __init__(self):
        self._configured = False
        self._loggers: dict[str, ContextLogger] = {}

    def configure(
        self,
        level: str | None = None,
        log_file: str | None = None,
        log_format: str = "console",
        *,
        console: bool = True,
    ) -> None:
        """Configure logging system."""
        if self._configured:
            return

        # Get configuration from environment
        level = level or os.getenv("LOG_LEVEL", "INFO")
        log_file = log_file or os.getenv("LOG_FILE")
        log_format = os.getenv("LOG_FORMAT", log_format)

        # Convert level string to logging constant
        numeric_level = getattr(logging, level.upper(), logging.INFO)

        # Create formatters
        if log_format.lower() == "json":
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S"
            )

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        # Clear existing handlers
        root_logger.handlers.clear()

        # Add console handler
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        self._configured = True

    def get_logger(self, name: str) -> ContextLogger:
        """Get or create a context logger."""
        if not self._configured:
            self.configure()

        if name not in self._loggers:
            base_logger = logging.getLogger(name)
            self._loggers[name] = ContextLogger(base_logger)

        return self._loggers[name]

    def set_level(self, level: str) -> None:
        """Change logging level at runtime."""
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric_level)

    def add_file_handler(self, log_file: str, log_format: str = "console") -> None:
        """Add file handler to existing configuration."""
        if log_format.lower() == "json":
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S"
            )

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


# Global manager instance
_manager = LoggerManager()

# Public API
def get_logger(name: str) -> ContextLogger:
    """Get a context logger by name."""
    return _manager.get_logger(name)

def configure_logging(**kwargs) -> None:
    """Configure the logging system."""
    _manager.configure(**kwargs)

def set_log_level(level: str) -> None:
    """Set logging level at runtime."""
    _manager.set_level(level)

def add_file_handler(log_file: str, log_format: str = "console") -> None:
    """Add file handler to existing configuration."""
    _manager.add_file_handler(log_file, log_format)

# Convenience function for common logger names
def get_core_logger() -> ContextLogger:
    """Get logger for core module."""
    return get_logger("modelbatch.core")

def get_optuna_logger() -> ContextLogger:
    """Get logger for optuna integration."""
    return get_logger("modelbatch.optuna")

def get_training_logger() -> ContextLogger:
    """Get logger for training operations."""
    return get_logger("modelbatch.training")
