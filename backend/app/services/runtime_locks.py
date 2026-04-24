import threading
from loguru import logger

# Global lock to prevent GPU OOM when both training and inference try to run simultaneously
training_lock = threading.Lock()

def acquire_training():
    """Reserved for the Training Service thread."""
    training_lock.acquire()
    logger.debug("🔒 Training Lock: ACQUIRED")

def release_training():
    """Reserved for the Training Service thread."""
    if training_lock.locked():
        training_lock.release()
        logger.debug("🔓 Training Lock: RELEASED")

def can_infer_on_gpu():
    """Returns True if the GPU is free for inference calls."""
    return not training_lock.locked()
