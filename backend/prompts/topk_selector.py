"""
Top-K sparse prompt token selection for efficient federated communication.

Selects K most important prompt tokens based on gradient magnitude or update norm.
Default K = 30% of total tokens.
"""
from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class SparsePromptUpdate:
    """Sparse update: only top-K token indices and values."""
    indices: list[int]
    values: np.ndarray       # (K, hidden_dim)
    importance: list[float]
    total_tokens: int
    k: int
    selection_method: str


class TopKSelector:
    """
    Selects Top-K prompt tokens for sparse communication.

    Strategies:
    - gradient: rank by gradient magnitude |∇P_i|
    - norm: rank by update norm ||P_i_new - P_i_old||
    - attention: use attention-weighted importance (future)
    """

    def __init__(
        self,
        k_ratio: float = 0.3,
        selection_method: str = "gradient",
        adaptive_k: bool = False,
        min_k: int = 1,
    ):
        self.k_ratio = k_ratio
        self.selection_method = selection_method
        self.adaptive_k = adaptive_k
        self.min_k = min_k

    def compute_k(self, total_tokens: int) -> int:
        k = max(self.min_k, int(total_tokens * self.k_ratio))
        return min(k, total_tokens)

    def select(
        self,
        prompt_tensor: torch.Tensor,           # (P, H)
        gradients: torch.Tensor | None = None, # (P, H)
        old_values: torch.Tensor | None = None,# (P, H)
    ) -> SparsePromptUpdate:
        P, H = prompt_tensor.shape
        k = self.compute_k(P)

        if self.selection_method == "gradient" and gradients is not None:
            importance = gradients.abs().mean(dim=-1).detach().cpu()  # (P,)
        elif self.selection_method == "norm" and old_values is not None:
            importance = (prompt_tensor - old_values).norm(dim=-1).detach().cpu()  # (P,)
        else:
            # Fallback: use prompt norm
            importance = prompt_tensor.norm(dim=-1).detach().cpu()  # (P,)

        topk_vals, topk_idx = torch.topk(importance, k=k, largest=True)
        topk_idx_sorted = topk_idx.sort().values
        topk_importance = importance[topk_idx_sorted]

        selected_values = prompt_tensor[topk_idx_sorted].detach().cpu().numpy()

        return SparsePromptUpdate(
            indices=topk_idx_sorted.tolist(),
            values=selected_values,
            importance=topk_importance.tolist(),
            total_tokens=P,
            k=k,
            selection_method=self.selection_method,
        )

    def communication_bytes(self, k: int, hidden_dim: int) -> int:
        """Approximate bytes for sparse update: k * hidden_dim * 4 bytes (float32)."""
        return k * hidden_dim * 4 + k * 4  # values + indices
