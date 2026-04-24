"""
Server-side sparse global prompt reconstruction.

Rule:
- For received token indices: apply FedAvg aggregation
- For non-received tokens: retain previous global prompt values (no zero-filling)
"""
from __future__ import annotations
import numpy as np
import torch
from collections import defaultdict
from .topk_selector import SparsePromptUpdate


class SparseGlobalReconstructor:
    """
    Reconstructs global prompt Pg from sparse client updates.

    Maintains:
    - token coverage frequency across rounds
    - continuity of non-updated tokens
    """

    def __init__(self, prompt_length: int, hidden_dim: int):
        self.prompt_length = prompt_length
        self.hidden_dim = hidden_dim
        self.token_coverage_count = np.zeros(prompt_length, dtype=np.int32)

    def aggregate(
        self,
        current_global: torch.Tensor,           # (P, H) - current Pg
        client_updates: list[SparsePromptUpdate],
        client_weights: list[float] | None = None,
    ) -> torch.Tensor:
        """
        FedAvg aggregation over sparse updates.

        Args:
            current_global: current global prompt tensor
            client_updates: list of sparse updates from clients
            client_weights: weight per client (e.g., dataset size). Defaults to uniform.

        Returns:
            Updated global prompt tensor (P, H)
        """
        if not client_updates:
            return current_global.clone()

        n = len(client_updates)
        if client_weights is None:
            client_weights = [1.0 / n] * n
        else:
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]

        # Accumulate weighted updates per token index
        token_sum = defaultdict(lambda: np.zeros(self.hidden_dim, dtype=np.float32))
        token_weight = defaultdict(float)

        for update, weight in zip(client_updates, client_weights):
            for i, idx in enumerate(update.indices):
                token_sum[idx] += weight * update.values[i]
                token_weight[idx] += weight

        # Build new global prompt
        new_global = current_global.clone().cpu().numpy()  # (P, H)

        for idx in token_sum:
            # Normalize by total weight received for this token
            w = token_weight[idx]
            if w > 0:
                new_global[idx] = token_sum[idx] / w

            self.token_coverage_count[idx] += 1

        return torch.tensor(new_global, dtype=torch.float32)

    def get_coverage_stats(self) -> dict:
        return {
            "mean_coverage": float(self.token_coverage_count.mean()),
            "min_coverage": int(self.token_coverage_count.min()),
            "max_coverage": int(self.token_coverage_count.max()),
            "coverage_per_token": self.token_coverage_count.tolist(),
        }
