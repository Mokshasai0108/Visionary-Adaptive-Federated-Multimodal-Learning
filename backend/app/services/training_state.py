"""
app/services/training_state.py — Shared training state for API layer
"""
from typing import Optional, Dict, List, Any
import threading


class TrainingState:
    """Thread-safe singleton for training state accessible from API routes."""

    def __init__(self):
        self._lock = threading.Lock()
        self.is_running = False
        self.current_round = 0
        self.total_rounds = 0
        self.server = None
        self.history: List[Dict] = []
        self.status_message = "idle"
        self.error: Optional[str] = None

    def start(self, total_rounds: int):
        with self._lock:
            self.is_running = True
            self.total_rounds = total_rounds
            self.current_round = 0
            self.status_message = "training"
            self.error = None

    def update_round(self, round_num: int, metrics: Dict):
        with self._lock:
            self.current_round = round_num
            self.history.append({"round": round_num, **metrics})

    def stop(self, message: str = "completed"):
        with self._lock:
            self.is_running = False
            self.status_message = message

    def set_error(self, error: str):
        with self._lock:
            self.is_running = False
            self.status_message = "error"
            self.error = error

    def get_status(self) -> Dict:
        with self._lock:
            return {
                "is_running": self.is_running,
                "current_round": self.current_round,
                "total_rounds": self.total_rounds,
                "status": self.status_message,
                "error": self.error,
                "progress": (self.current_round / max(self.total_rounds, 1)) * 100,
            }

    def get_history(self) -> List[Dict]:
        with self._lock:
            return list(self.history)


# Global singleton
training_state = TrainingState()
