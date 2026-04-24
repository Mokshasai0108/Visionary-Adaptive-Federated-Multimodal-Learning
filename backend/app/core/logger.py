"""
core/logger.py — Structured JSON logging for experiments
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            payload.update(record.extra)
        return json.dumps(payload)


def get_logger(name: str, log_file: str = None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.handlers:
        return logger  # avoid duplicate handlers

    formatter = JSONFormatter()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class RoundMetricsLogger:
    """Persists per-round metrics to a JSON file."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: list = []

    def log(self, round_num: int, metrics: Dict[str, Any]):
        entry = {"round": round_num, "timestamp": datetime.utcnow().isoformat(), **metrics}
        self.history.append(entry)
        with open(self.output_path, "w") as f:
            json.dump(self.history, f, indent=2, default=float)

    def get_history(self) -> list:
        return self.history
