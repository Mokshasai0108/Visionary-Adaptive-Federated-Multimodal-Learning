"""
Evaluation metrics: BLEU, ROUGE, CIDEr, Recall@K, CLIP similarity.
"""
from __future__ import annotations
import numpy as np
import torch
from loguru import logger

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


def compute_bleu(references: list[list[str]], hypotheses: list[str]) -> dict:
    """Compute BLEU-1 to BLEU-4."""
    if not NLTK_AVAILABLE:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    sf = SmoothingFunction().method1
    refs_tok = [[r.lower().split() for r in refs] for refs in references]
    hyps_tok = [h.lower().split() for h in hypotheses]

    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n)
        try:
            score = corpus_bleu(refs_tok, hyps_tok, weights=weights, smoothing_function=sf)
        except Exception:
            score = 0.0
        scores[f"bleu{n}"] = round(float(score), 4)
    return scores


def compute_rouge(references: list[str], hypotheses: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L."""
    if not ROUGE_AVAILABLE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []

    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": round(float(np.mean(r1)), 4),
        "rouge2": round(float(np.mean(r2)), 4),
        "rougeL": round(float(np.mean(rl)), 4),
    }


def compute_cider_simple(references: list[list[str]], hypotheses: list[str]) -> float:
    """
    Simplified TF-IDF based CIDEr approximation.
    For full CIDEr, use pycocoevalcap.
    """
    from collections import Counter
    import math

    def ngrams(tokens: list[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def tfidf_score(hyp_ngrams, ref_ngrams_list, all_ref_ngrams_list):
        score = 0.0
        n_refs = len(all_ref_ngrams_list)
        for ng, count in hyp_ngrams.items():
            df = sum(1 for rng in all_ref_ngrams_list if ng in rng)
            idf = math.log((n_refs + 1.0) / (df + 1.0))
            ref_count = max((rng.get(ng, 0) for rng in ref_ngrams_list), default=0)
            score += min(count, ref_count) * idf
        return score

    all_ref_ngrams = [ngrams(r.lower().split(), 2) for refs in references for r in refs]
    cider_scores = []
    for refs, hyp in zip(references, hypotheses):
        hyp_ng = ngrams(hyp.lower().split(), 2)
        ref_ng_list = [ngrams(r.lower().split(), 2) for r in refs]
        s = tfidf_score(hyp_ng, ref_ng_list, all_ref_ngrams)
        cider_scores.append(s)

    return round(float(np.mean(cider_scores)) if cider_scores else 0.0, 4)


def compute_recall_at_k(
    image_embeds: torch.Tensor,  # (N, D)
    text_embeds: torch.Tensor,   # (N, D)
    k_values: list[int] = [1, 5, 10],
) -> dict:
    """
    Compute image-to-text and text-to-image Recall@K.
    Assumes index i corresponds to matched pair (i, i).
    """
    # Normalize
    img_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    txt_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    sim = torch.matmul(img_norm, txt_norm.T)  # (N, N)
    N = sim.size(0)

    results = {}
    for k in k_values:
        if k > N:
            results[f"r@{k}"] = 1.0
            continue
        # I2T
        topk_idx = sim.topk(k, dim=1).indices
        gt = torch.arange(N, device=sim.device).unsqueeze(1)
        i2t = (topk_idx == gt).any(dim=1).float().mean().item()
        # T2I
        topk_idx_t = sim.T.topk(k, dim=1).indices
        t2i = (topk_idx_t == gt).any(dim=1).float().mean().item()
        results[f"r@{k}"] = round((i2t + t2i) / 2, 4)

    return results


def compute_clip_similarity(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
) -> float:
    """Mean diagonal cosine similarity."""
    img_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    txt_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    diag_sim = (img_norm * txt_norm).sum(dim=-1).mean().item()
    return round(float(diag_sim), 4)
