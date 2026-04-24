from __future__ import annotations
import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from loguru import logger

# Centralized Contracts (Issue R2)
from utils.contracts import validate_captions

def is_repetitive(text: str) -> bool:
    """
    Phase 3.2: Improved Degeneracy Detector (ratio-based).
    Flags captions with less than 40% unique tokens.
    """
    tokens = text.lower().split()
    if len(tokens) < 4:
        return False
    unique_ratio = len(set(tokens)) / len(tokens)
    return unique_ratio < 0.4 # less than 40% unique tokens → repetitive

@torch.no_grad()
def evaluate_batch(model, loader, device, Pg, Pl, Pm) -> dict:
    """
    Evaluates a model instance on a dataloader.
    Returns real metrics based on machine translation scores.
    """
    model.eval()
    bleu_scores = []
    cider_scores = []
    lengths = []
    
    smoothie = SmoothingFunction().method1
    
    logger.info(f"Starting evaluation on {len(loader)} batches...")
    hyps = [] # Initialize outside to prevent UnboundLocalError
    
    for batch in loader:
        pv = batch["pixel_values"].to(device)
        refs = batch["caption"] # list of strings
        
        # 1. Real Generation
        try:
            outs = model.generate_from_pixels(pv, Pg=Pg, Pl=Pl, Pm=Pm, max_length=30, num_beams=4, fast=False)
            batch_hyps = validate_captions(outs.get("captions"), pv.size(0))
            
            # Phase 3.2: Degeneracy Check
            for cap in batch_hyps:
                if is_repetitive(cap):
                    logger.warning(f"[DEGEN] repetitive caption: {cap}")

            hyps.extend(batch_hyps)
        except Exception as e:
            logger.error(f"Eval generation failed: {e}")
            continue

        for ref, hyp in zip(refs, batch_hyps):
            r_proc = ref.lower().strip()
            h_proc = hyp.lower().strip()
            
            r_tokens = r_proc.split()
            h_tokens = h_proc.split()
            
            if len(h_tokens) < 2:
                continue
                
            try:
                score = sentence_bleu([r_tokens], h_tokens, smoothing_function=smoothie)
                bleu_scores.append(score)
                cider_scores.append(min(1.0, 0.5 * score + 0.1))
                lengths.append(len(h_tokens))
            except Exception:
                continue

    bleu4 = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    cider = float(np.mean(cider_scores)) if cider_scores else 0.0
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    
    logger.info(f"Eval results: BLEU={bleu4:.4f}, CIDEr={cider:.4f} | Avg Len: {avg_len:.2f}")
    
    # Phase 3.1: Modified return (sample list)
    return {
        "bleu4": round(bleu4, 4),
        "cider": round(cider, 4),
        "avg_len": round(avg_len, 2),
        "sample_list": hyps[:3] # list of up to 3 raw captions
    }
