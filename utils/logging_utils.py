import logging
import os
from typing import Optional


def _get_level_from_env(env_var: str = "FBTOOLS_LOG_LEVEL", default: str = "INFO") -> int:
    value = os.getenv(env_var, default).upper()
    return getattr(logging, value, logging.INFO)


def configure_logging(env_var: str = "FBTOOLS_LOG_LEVEL", default_level: str = "INFO") -> None:
    """Configure root logging once. Safe to call multiple times."""
    if getattr(configure_logging, "_configured", False):
        return

    level = _get_level_from_env(env_var, default_level)
    logging.basicConfig(
        level=level,
        format="fbTools:%(levelname)s:%(name)s:%(message)s",
    )
    configure_logging._configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger configured for fbTools; configures logging on first call."""
    configure_logging()
    return logging.getLogger(name or "fbTools")
