"""
AFSPL Loss Functions.

Total loss: L = λ1 * L_ce + λ2 * L_clip

L_ce  = CrossEntropy caption generation loss
L_clip = CLIP contrastive alignment loss:
  L_clip = -log( exp(sim(I,T)/τ) / Σ_j exp(sim(I,T_j)/τ) )
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, seq_len, vocab_size)
        labels: (B, seq_len)
        """
        B, S, V = logits.shape
        return self.ce(logits.view(B * S, V), labels.view(B * S))


class CLIPContrastiveLoss(nn.Module):
    """
    Symmetric CLIP-style contrastive loss.
    L_clip = -log( exp(sim(I,T)/τ) / Σ_j exp(sim(I,T_j)/τ) )
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_embeds: torch.Tensor,  # (B, D) L2-normalized
        text_embeds: torch.Tensor,   # (B, D) L2-normalized
    ) -> torch.Tensor:
        # Cosine similarity matrix: (B, B)
        sim = torch.matmul(image_embeds, text_embeds.T) / self.temperature

        # Labels: diagonal (each image matches its text)
        labels = torch.arange(sim.size(0), device=sim.device)

        # Symmetric loss
        loss_i2t = F.cross_entropy(sim, labels)
        loss_t2i = F.cross_entropy(sim.T, labels)
        return (loss_i2t + loss_t2i) / 2.0


class AFSPLLoss(nn.Module):
    """
    Combined AFSPL loss:
    L = λ1 * L_ce + λ2 * L_clip
    """

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 0.5,
        clip_temperature: float = 0.07,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.ce_loss = CrossEntropyLoss(label_smoothing=0.1)
        self.clip_loss = CLIPContrastiveLoss(clip_temperature)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        image_embeds: torch.Tensor | None = None,
        text_embeds: torch.Tensor | None = None,
        effective_lambda2: float | None = None,
    ) -> tuple[torch.Tensor, dict]:
        l_ce = self.ce_loss(logits, labels)
        components = {"l_ce": l_ce.item(), "l_clip": 0.0}

        total = self.lambda1 * l_ce
        
        # Use effective lambda for warm-up if provided, otherwise default
        l2 = effective_lambda2 if effective_lambda2 is not None else self.lambda2

        if l2 > 0 and image_embeds is not None and text_embeds is not None:
            # Stable Grounding Loss (Phase 2.6 refinement)
            cosine_sim = F.cosine_similarity(image_embeds, text_embeds, dim=-1)
            l_clip = torch.mean(1 - cosine_sim.clamp(-1, 1))
            
            total = total + l2 * l_clip
            components["l_clip"] = l_clip.item()

        components["total"] = total.item()
        return total, components
