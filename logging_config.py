"""
Structured logging configuration for the RAG application.

Provides JSON logging for production and readable console logging for development.
"""

import logging
import sys
from datetime import datetime, timezone
from typing import Any
import json

from config import settings


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production environments."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for development."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        formatted = (
            f"{color}{timestamp} | {record.levelname:8} | "
            f"{record.name}:{record.funcName}:{record.lineno} | "
            f"{record.getMessage()}{self.RESET}"
        )
        
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


def setup_logging() -> None:
    """Configure application logging based on environment."""
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Use JSON formatter in production, colored console in development
    if settings.is_production:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ColoredConsoleFormatter())
    
    root_logger.addHandler(console_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name, typically __name__ from the calling module.
    
    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


# Initialize logging on module import
setup_logging()
