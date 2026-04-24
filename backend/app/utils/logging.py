"""Structured logging setup with JSON export."""
from __future__ import annotations
import sys
import json
from pathlib import Path
from loguru import logger


def setup_logging(log_dir: str = "./logs", level: str = "INFO", json_logs: bool = True) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()

    # Console handler
    logger.add(sys.stdout, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> — {message}")

    # File handler
    log_file = Path(log_dir) / "afspl_{time:YYYY-MM-DD}.log"
    logger.add(str(log_file), level=level, rotation="10 MB", retention="7 days",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}")

    if json_logs:
        json_file = Path(log_dir) / "afspl_structured.jsonl"
        logger.add(str(json_file), level=level, serialize=True, rotation="50 MB")

    logger.info(f"Logging initialized. Output: {log_dir}")
