"""Soft prompt management: Pg, Pl, Pm initialization and storage."""
from __future__ import annotations
import torch
import torch.nn as nn
from loguru import logger


class SoftPrompt(nn.Module):
    """A single trainable soft prompt tensor of shape (prompt_length, hidden_dim)."""

    def __init__(
        self,
        prompt_length: int,
        hidden_dim: int,
        init_strategy: str = "random",
        init_std: float = 0.02,
        token_embeddings: torch.Tensor | None = None,
        name: str = "prompt",
    ):
        super().__init__()
        self.prompt_length = prompt_length
        self.hidden_dim = hidden_dim
        self.name = name

        data = self._initialize(init_strategy, init_std, token_embeddings)
        self.embedding = nn.Parameter(data)
        logger.info(f"SoftPrompt '{name}' initialized with strategy='{init_strategy}' shape={data.shape}")

    def _initialize(
        self,
        strategy: str,
        std: float,
        token_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        if strategy == "random":
            return torch.randn(self.prompt_length, self.hidden_dim) * std
        elif strategy == "token_embed" and token_embeddings is not None:
            # Sample prompt_length token embeddings and average
            n = token_embeddings.size(0)
            indices = torch.randint(0, n, (self.prompt_length,))
            return token_embeddings[indices].clone().detach()
        else:
            logger.warning(f"Unknown strategy '{strategy}', falling back to random.")
            return torch.randn(self.prompt_length, self.hidden_dim) * std

    def forward(self) -> torch.Tensor:
        return self.embedding  # (P, H)


class PromptManager:
    """Manages global, local, and modality prompts for AFSPL."""

    def __init__(
        self,
        prompt_length: int = 16,
        hidden_dim: int = 512,
        init_strategy: str = "random",
        init_std: float = 0.02,
        token_embeddings: torch.Tensor | None = None,
        device: str = "cpu",
    ):
        self.prompt_length = prompt_length
        self.hidden_dim = hidden_dim
        self.device = device

        def make(name: str) -> SoftPrompt:
            return SoftPrompt(
                prompt_length, hidden_dim, init_strategy, init_std,
                token_embeddings, name=name
            ).to(device)

        self.global_prompt = make("global")     # Pg — managed by server
        self.local_prompt = make("local")       # Pl — client-specific
        self.modality_prompt = make("modality") # Pm — modality-specific

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return only local + modality prompts for client-side training."""
        return list(self.local_prompt.parameters()) + list(self.modality_prompt.parameters())

    def set_global_prompt(self, tensor: torch.Tensor) -> None:
        """Load server-sent global prompt (no gradient)."""
        with torch.no_grad():
            self.global_prompt.embedding.copy_(tensor.to(self.device))

    def get_global_prompt_numpy(self):
        return self.global_prompt.embedding.detach().cpu().numpy()

    def get_local_prompt_numpy(self):
        return self.local_prompt.embedding.detach().cpu().numpy()

    def state_dict_trainable(self) -> dict:
        return {
            "local_prompt": self.local_prompt.embedding.detach().cpu(),
            "modality_prompt": self.modality_prompt.embedding.detach().cpu(),
        }

    def load_state_dict_trainable(self, state: dict) -> None:
        with torch.no_grad():
            if "local_prompt" in state:
                self.local_prompt.embedding.copy_(state["local_prompt"].to(self.device))
            if "modality_prompt" in state:
                self.modality_prompt.embedding.copy_(state["modality_prompt"].to(self.device))
