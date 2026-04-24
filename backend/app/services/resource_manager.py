import torch
import threading
from loguru import logger

class MultimodalResourceManager:
    """
    Deterministic VRAM Lifecycle Manager for Phase 3.
    Ensures symmetric swaps between Captioning (CLIP/T5) and Diffusion (SD-Turbo).
    """
    _instance = None
    _lock = threading.Lock()
    _active_mode = None # "captioner" | "diffusion"

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(MultimodalResourceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # We don't re-init if singleton already exists
        if not hasattr(self, 'initialized'):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.initialized = True
            logger.info(f"🚀 Multimodal Resource Manager Initialized. Device: {self.device}")

    def _cleanup(self):
        """Standard cache cleanup."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def activate_captioning(self, model, diffusion_pipe=None):
        """Symmetric swap: Offload Diffusion -> Load Captioning."""
        with self._lock:
            if self._active_mode == "captioner":
                return # Already ready

            logger.info("⚙️ ResourceManager: Activating Captioning Suite (CLIP + T5)...")
            
            # 1. Offload Diffusion
            if diffusion_pipe is not None:
                diffusion_pipe.to("cpu")
            
            # 2. Cleanup
            self._cleanup()

            # 3. Load Captioning model
            model.to(self.device)
            self._active_mode = "captioner"
            logger.info("✅ Captioning Suite Active on GPU.")

    def activate_diffusion(self, model, diffusion_pipe):
        """Symmetric swap: Offload Captioning -> Load Diffusion."""
        with self._lock:
            if self._active_mode == "diffusion":
                return # Already ready

            logger.info("⚙️ ResourceManager: Activating Diffusion Suite (SD-Turbo)...")
            
            # 1. Offload Captioning
            model.to("cpu")
            
            # 2. Cleanup
            self._cleanup()

            # 3. Load Diffusion pipe
            if diffusion_pipe is not None:
                diffusion_pipe.to(self.device)
            
            self._active_mode = "diffusion"
            logger.info("✅ Diffusion Suite Active on GPU.")

    def get_active_mode(self):
        return self._active_mode
