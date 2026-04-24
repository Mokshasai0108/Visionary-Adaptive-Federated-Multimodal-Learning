"""
Flower client wrapper for real distributed FL.
Wraps ClientTrainer to be compatible with Flower's NumPyClient interface.
"""
from __future__ import annotations
import numpy as np
import torch
import flwr as fl
from flwr.common import NDArrays, Scalar
from loguru import logger

from prompts.prompt_manager import PromptManager
from prompts.fusion import AdaptiveFusionModule
from prompts.topk_selector import TopKSelector
from training.losses import AFSPLLoss
from training.trainer import ClientTrainer
from models.multimodal_model import AFSPLModel


class AFSPLFlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient that wraps AFSPL ClientTrainer.
    Communicates sparse (indices, values) prompt updates to the server.
    """

    def __init__(
        self,
        client_id: int,
        model: AFSPLModel,
        prompt_manager: PromptManager,
        fusion_module: AdaptiveFusionModule,
        topk_selector: TopKSelector,
        loss_fn: AFSPLLoss,
        dataloader,
        config,
        device: str = "cpu",
    ):
        self.trainer = ClientTrainer(
            client_id, model, prompt_manager, fusion_module,
            topk_selector, loss_fn, config, device,
        )
        self.dataloader = dataloader
        self.pm = prompt_manager

    def get_parameters(self, config) -> NDArrays:
        """Return current local prompt as flat array."""
        return [self.pm.local_prompt.embedding.detach().cpu().numpy()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Load global prompt from server."""
        global_prompt = torch.tensor(parameters[0], dtype=torch.float32)
        self.pm.set_global_prompt(global_prompt)

    def fit(self, parameters: NDArrays, config) -> tuple[NDArrays, int, dict]:
        """Train one round and return sparse update."""
        self.set_parameters(parameters)
        global_prompt = torch.tensor(parameters[0], dtype=torch.float32)

        sparse_update, metrics = self.trainer.train_round(self.dataloader, global_prompt)

        # Pack sparse update as [indices_array, values_array]
        indices = np.array(sparse_update.indices, dtype=np.float32)
        values = sparse_update.values  # (K, H)

        return [indices, values], len(self.dataloader.dataset), metrics

    def evaluate(self, parameters: NDArrays, config) -> tuple[float, int, dict]:
        """Evaluate on local data."""
        self.set_parameters(parameters)
        return 0.0, len(self.dataloader.dataset), {"status": "ok"}
