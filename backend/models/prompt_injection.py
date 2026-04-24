"""Prompt injection via prefix tuning / embedding concatenation."""
from __future__ import annotations
import torch
import torch.nn as nn


class PromptInjection(nn.Module):
    """
    Injects fused soft prompt P as prefix embeddings before the decoder input.

    Strategy: Prefix Tuning (default)
    - Fused prompt P: (B, prompt_len, hidden_dim)
    - Decoder input tokens embedded: (B, seq_len, hidden_dim)
    - Combined: (B, prompt_len + seq_len, hidden_dim)

    This is the implementation of Section 6: Prompt Injection Strategy.
    """

    def __init__(
        self,
        prompt_length: int = 16,
        hidden_dim: int = 512,
        strategy: str = "prefix",  # prefix | deep (p-tuning v2)
    ):
        super().__init__()
        self.prompt_length = prompt_length
        self.hidden_dim = hidden_dim
        self.strategy = strategy

    def inject(
        self,
        fused_prompt: torch.Tensor,      # (B, prompt_len, hidden_dim) or (prompt_len, hidden_dim)
        input_embeds: torch.Tensor,       # (B, seq_len, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            combined_embeds: (B, prompt_len + seq_len, hidden_dim)
            prefix_attention_mask: (B, prompt_len + seq_len)
        """
        B = input_embeds.size(0)
        if fused_prompt.dim() == 2:
            fused_prompt = fused_prompt.unsqueeze(0).expand(B, -1, -1)

        assert fused_prompt.size(-1) == self.hidden_dim, (
            f"Prompt hidden dim {fused_prompt.size(-1)} != expected {self.hidden_dim}"
        )

        combined = torch.cat([fused_prompt, input_embeds], dim=1)  # (B, P+S, H)
        return combined

    def forward(
        self,
        fused_prompt: torch.Tensor,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        return self.inject(fused_prompt, input_embeds)
