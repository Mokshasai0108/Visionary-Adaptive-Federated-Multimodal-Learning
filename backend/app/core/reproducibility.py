"""Reproducibility utilities: seed control, determinism."""
import random
import numpy as np
import torch
from loguru import logger


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed} | deterministic={deterministic}")
