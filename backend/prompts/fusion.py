"""
Adaptive Prompt Fusion Module.

Implements: P = α*Pg + β*Pl + γ*Pm

With three fusion strategies:
1. static: fixed scalar weights
2. learnable: trainable scalars (softmax normalized)
3. dynamic: gating network conditioned on E_image + E_text

α, β, γ = softmax(W · f(E_image, E_text) + b)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class StaticFusion(nn.Module):
    """Fixed scalar weights: α, β, γ from config."""

    def __init__(self, alpha: float = 0.33, beta: float = 0.33, gamma: float = 0.34):
        super().__init__()
        weights = torch.tensor([alpha, beta, gamma])
        weights = F.softmax(weights, dim=0)
        self.register_buffer("weights", weights)

    def forward(self, Pg, Pl, Pm, E_image=None, E_text=None, scale: float = 1.0):
        a, b, g = self.weights
        # Scale vision contribution (g) if needed
        g = g * scale
        total = a + b + g
        
        # Null-safe expansion
        B = E_image.size(0) if E_image is not None else 1
        def safe_p(p):
            if p is None: return torch.zeros((B, 16, 512), device=self.weights.device)
            return p.unsqueeze(0).expand(B, -1, -1) if p.dim() == 2 else p

        fused = (a/total) * safe_p(Pg) + (b/total) * safe_p(Pl) + (g/total) * safe_p(Pm)
        return fused, self.weights.clone()


class LearnableScalarFusion(nn.Module):
    """Trainable scalar weights (softmax-normalized)."""

    def __init__(self):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.ones(3))  # un-normalized

    def forward(self, Pg, Pl, Pm, E_image=None, E_text=None, scale: float = 1.0):
        weights = F.softmax(self.raw_weights, dim=0)
        a, b, g = weights
        # Scale vision contribution if needed
        g = g * scale
        total = a + b + g
        
        # Null-safe expansion
        B = E_image.size(0) if E_image is not None else 1
        def safe_p(p):
            if p is None: return torch.zeros((B, 16, 512), device=self.raw_weights.device)
            return p.unsqueeze(0).expand(B, -1, -1) if p.dim() == 2 else p

        fused = (a/total) * safe_p(Pg) + (b/total) * safe_p(Pl) + (g/total) * safe_p(Pm)
        return fused, weights.detach()


class DynamicGatingFusion(nn.Module):
    """
    Dynamic gating network.
    α, β, γ = softmax(W · f(E_image, E_text) + b)

    f(E_image, E_text) = concat([E_image, E_text]) or just E_image if no text.
    """

    def __init__(self, image_dim: int = 512, text_dim: int = 512):
        super().__init__()
        input_dim = image_dim + text_dim
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        self.tau = 0.7  # Temperature Gate (Phase 2.6 refinement)

    def forward(
        self,
        Pg: torch.Tensor,   # (P, H) or (B, P, H)
        Pl: torch.Tensor,
        Pm: torch.Tensor,
        E_image: torch.Tensor,  # (B, image_dim)
        E_text: torch.Tensor | None = None,
        scale: float = 1.0,
    ):
        B = E_image.size(0)
        if E_text is None:
            E_text = torch.zeros(B, E_image.size(1), device=E_image.device)
        f = torch.cat([E_image, E_text], dim=-1)  # (B, D)
        
        logits = self.gate(f)                      # (B, 3)
        weights = F.softmax(logits / self.tau, dim=-1)        # (B, 3)
        
        # Apply Soft Ramp to Vision (Modality) Weight
        # We modify the RAW weight before re-normalization to preserve learned balance
        alpha, beta, gamma = weights[:, 0], weights[:, 1], weights[:, 2]
        gamma = gamma * scale
        
        # Adaptive Boost Based on Confidence (Phase 2.6 refinement)
        gamma = gamma * (1.1 + 0.2 * (1.0 - gamma))
        gamma = torch.clamp(gamma, max=0.7) # Phase 2.6 refinement: Prevent vision over-dominance
        alpha = alpha * (1.0 - 0.1 * gamma)
        beta  = beta  * (1.0 - 0.1 * gamma)
        
        weights = torch.stack([alpha, beta, gamma], dim=-1)
        
        # Phase 3.5: Soft Constraint Scaling (User Refinement)
        # 1. Shape Safety Check
        assert weights.dim() == 2 and weights.size(1) == 3, f"❌ Fusion weights shape invalid: {weights.shape}"
        
        # 2. Smooth Diversity Guardrail [min ~0.09 after re-norm]
        # Prevents gradient discontinuity from hard clamping while ensuring all modalities contribute.
        weights = 0.1 + 0.8 * weights
        
        # 3. Final Re-normalization
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Variance check for adaptivity (debug mode/preflight)
        if B > 1 and weights.var() < 1e-7:
             logger.debug("⚠️ Fusion weights show extremely low variance (non-adaptive).")

        a = weights[:, 0:1].unsqueeze(-1)  # (B, 1, 1) -> Global (alpha)
        b = weights[:, 1:2].unsqueeze(-1)  # (B, 1, 1) -> Local (beta)
        g = weights[:, 2:3].unsqueeze(-1)  # (B, 1, 1) -> Modality (gamma)

        # Expand prompts to batch if needed
        def expand(p):
            if p is None:
                return torch.zeros((B, 16, 512), device=E_image.device)
            if p.dim() == 2:
                return p.unsqueeze(0).expand(B, -1, -1)
            return p

        fused = a * expand(Pg) + b * expand(Pl) + g * expand(Pm)  # (B, P, H)
        
        # Structure telemetry for the diagram labels
        telemetry = {
            "alpha": float(weights[:, 0].detach().mean()),
            "beta": float(weights[:, 1].detach().mean()),
            "gamma": float(weights[:, 2].detach().mean()),
        }
        return fused, telemetry


class AdaptiveFusionModule(nn.Module):
    """
    Wrapper selecting fusion strategy.
    Returns: fused_prompt (B, P, H), weights (dict) for logging.
    """

    def __init__(
        self,
        strategy: str = "dynamic",
        image_dim: int = 512,
        text_dim: int = 512,
        static_alpha: float = 0.33,
        static_beta: float = 0.33,
        static_gamma: float = 0.34,
    ):
        super().__init__()
        self.strategy = strategy
        if strategy == "static":
            self.fusion = StaticFusion(static_alpha, static_beta, static_gamma)
        elif strategy == "learnable":
            self.fusion = LearnableScalarFusion()
        elif strategy == "dynamic":
            self.fusion = DynamicGatingFusion(image_dim, text_dim)
        self.locked = False # Allow structural locking

    def get_weights(self) -> dict[str, float]:
        """Exposes the internal learned coefficients for the Model Registry."""
        if hasattr(self, "last_weights"):
            return self.last_weights
        return {"alpha": 0.33, "beta": 0.33, "gamma": 0.34}

    def forward(self, Pg, Pl, Pm, E_image=None, E_text=None, scale: float = 1.0):
        fused, weights = self.fusion(Pg, Pl, Pm, E_image, E_text, scale=scale)
        self.last_weights = weights # capture for registry
        return fused, weights

    def get_trainable_params(self) -> list[nn.Parameter]:
        if self.strategy == "static":
            return []
        return list(self.fusion.parameters())
