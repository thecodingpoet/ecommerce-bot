"""Logging configuration for the multi-agent RAG system."""

import logging
import sys

CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and icons."""

    COLORS = {
        "DEBUG": CYAN,
        "INFO": GREEN,
        "WARNING": YELLOW,
        "ERROR": RED,
        "CRITICAL": MAGENTA,
        "RESET": RESET,
    }

    ICONS = {
        "DEBUG": "🔍",
        "INFO": "ℹ️ ",
        "WARNING": "⚠️ ",
        "ERROR": "❌",
        "CRITICAL": "🚨",
    }

    def format(self, record):
        """Format log record with colors and icons."""
        icon = self.ICONS.get(record.levelname, "")
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        record.levelname = f"{color}{icon} {record.levelname}{reset}"
        return "\n" + super().format(record)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with colored output.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = ColoredFormatter(
        "%(asctime)s %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger
