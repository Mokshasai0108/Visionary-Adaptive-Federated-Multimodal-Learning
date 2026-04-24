import numpy as np

def sample_subset(samples, ratio=0.3, seed=42):
    """Randomly samples a subset of the given samples."""
    rng = np.random.default_rng(seed)
    n = int(len(samples) * ratio)
    if n == 0: return []
    indices = rng.choice(len(samples), n, replace=False)
    return [samples[i] for i in indices]

def split_train_val(samples, val_ratio=0.1):
    """Splits samples into train and validation sets."""
    split = int(len(samples) * (1 - val_ratio))
    return samples[:split], samples[split:]
