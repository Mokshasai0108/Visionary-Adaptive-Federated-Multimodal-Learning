import threading
from loguru import logger

class ModelRegistry:
    """
    Thread-safe global hub for the AFSPL model and its research metadata.
    Allows the Inference API to access the latest trained prompts from the background thread.
    """
    _model = None
    _meta = {}
    _lock = threading.RLock()  # allow nested reads

    @classmethod
    def set(cls, model, meta: dict = None):
        with cls._lock:
            cls._model = model
            cls._meta = meta or {}
            logger.info(f"🧠 ModelRegistry: Updated with Round {meta.get('round', '?')}")

    @classmethod
    def get(cls):
        with cls._lock:
            return cls._model, cls._meta

    @classmethod
    def get_active_model_name(cls):
        with cls._lock:
            return cls._meta.get("model_name", "default")

    @classmethod
    def get_round(cls, model) -> int:
        with cls._lock:
            return cls._meta.get("round", 0)

    @classmethod
    def get_metrics(cls, model) -> dict:
        with cls._lock:
            return cls._meta.get("metrics", {})

    @classmethod
    def has_model(cls):
        with cls._lock:
            return cls._model is not None
