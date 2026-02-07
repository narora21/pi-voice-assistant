import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config.schema import LoggingConfig

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(config: LoggingConfig) -> None:
    """Configure root logger with rotating file handler and console output."""
    level = getattr(logging, config.level.upper(), logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Rotating file handler
    log_path = Path(config.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
