"""
test_training_lite.py
--------------------
A quick sanity check for the training pipeline.
Runs 1 round with a tiny data subset.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import get_config
from app.schemas.train import TrainStartRequest
from app.services.training_service import _run_training
from loguru import logger

if __name__ == "__main__":
    config = get_config()
    
    # Lite settings for quick test
    request = TrainStartRequest(
        num_rounds=2,
        num_clients=2,
        subset_ratio=0.01, # 1% data
        use_test_mode=True,
        fusion_strategy="dynamic",
        batch_size=4,
        learning_rate=1e-4,
        k_ratio=0.3
    )
    
    # Speed overrides
    config.model.max_seq_len = 16
    config.training.num_workers = 0 # No multiprocessing for stability
    config.training.max_rounds = 2
    
    logger.info("Starting LITE training test...")
    try:
        _run_training(request, config)
        logger.success("LITE training test completed successfully!")
    except Exception as e:
        logger.error(f"LITE training test failed: {e}")
        sys.exit(1)
