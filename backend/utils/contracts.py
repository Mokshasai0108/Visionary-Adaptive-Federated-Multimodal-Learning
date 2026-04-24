"""
AFSPL Unified Generation Contract.
Centralized utility to enforce tensor and linguistic invariants across the pipeline.
"""
import torch
from loguru import logger

def enforce_generation_contract(sequences, batch_size, pad_id, device):
    """
    Standard AFSPL Generation Contract (Final Robust Version).
    Enforces (B, L) shape with Beam-Search awareness.
    """
    if sequences is None:
        return torch.full((batch_size, 8), pad_id, device=device, dtype=torch.long)

    # Phase 1.2: Use reshape to handle non-contiguous tensors
    if sequences.dim() != 2:
        sequences = sequences.reshape(sequences.size(0), -1)

    # Phase 1.3: Beam-Aware Batch Recovery
    # If sequences is (B*beams, L), extract the first beam for each sample
    if sequences.size(0) > batch_size and sequences.size(0) % batch_size == 0:
        num_beams = sequences.size(0) // batch_size
        sequences = sequences.reshape(batch_size, num_beams, -1)[:, 0, :]
    
    # Final safety truncation
    if sequences.size(0) != batch_size:
        sequences = sequences[:batch_size]

    return sequences

def validate_captions(captions: list[str] | str | None, batch_size: int) -> list[str]:
    """
    Standardizes caption lists and handles fallback to empty strings.
    Ref R1: Use empty strings instead of "fail" to prevent distribution bias.
    """
    if captions is None:
        return [""] * batch_size
        
    if isinstance(captions, str):
        captions = [captions]
        
    if not isinstance(captions, list):
        try:
            captions = list(captions)
        except Exception:
            return [""] * batch_size
            
    # Normalize length
    if len(captions) < batch_size:
        captions += [""] * (batch_size - len(captions))
    elif len(captions) > batch_size:
        captions = captions[:batch_size]
        
    return captions
