"""Standalone training script (no API)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from app.core.config import get_config
from app.schemas.train import TrainStartRequest
from app.services.training_service import _run_training

if __name__ == "__main__":
    config = get_config()
    request = TrainStartRequest(
        num_rounds=20,               # Equilibrium session (20 rounds)
        num_clients=10,
        fusion_strategy="dynamic",
        subset_ratio=0.3,            # Balanced 30% diversity
        learning_rate=5e-5,          # Baseline start
        k_ratio=0.25,                # Stable density
        is_recovery=False,            # Phase 5: Clean Slate Restart (Kill contaminated checkpoints)
        checkpoint_path=None,
        lambda1=1.0,
        lambda2=0.05,
        prompt_length=16,
        seed=42
    )
    # Execution Overrides for Windows stability
    config.model.max_seq_len = 32
    config.training.num_workers = 0 
    config.training.batch_size = 16
    config.prompt.init_std = 0.01    # Phase 5: Quiet initialization for safety
    _run_training(request, config)
    print("Training complete.")
