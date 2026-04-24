"""
AFSPL Federated Server.
Manages global prompt, coordinates rounds, aggregates sparse updates.
Implements the server-side flow from the spec.
"""
from __future__ import annotations
import json
import time
import copy
from pathlib import Path
import torch
import numpy as np
from loguru import logger

from prompts.prompt_manager import PromptManager
from prompts.sparse_reconstruction import SparseGlobalReconstructor
from prompts.topk_selector import SparsePromptUpdate


class CheckpointManager:
    def __init__(self, save_dir: str, metric: str = "cider"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -float("inf")
        self.metric = metric

    def save(self, state: dict, round_num: int, metric_value: float | None = None, is_best: bool = False):
        path = self.save_dir / f"checkpoint_round_{round_num:04d}.pt"
        torch.save(state, path)
        if is_best:
            best_path = self.save_dir / "best_checkpoint.pt"
            torch.save(state, best_path)
            logger.info(f"Best checkpoint saved at round {round_num} ({self.metric}={metric_value:.4f})")
        logger.debug(f"Checkpoint saved: {path}")


class FederatedServer:
    """
    Simulates the federated server loop.
    Works with ClientTrainer instances in-process (simulation mode).
    Can be extended to use Flower for real distributed FL.
    """

    def __init__(self, config, device: str = "cpu"):
        self.config = config
        self.device = device

        # Step 1: Initialize global prompt
        self.global_prompt_manager = PromptManager(
            prompt_length=config.prompt.prompt_length,
            hidden_dim=512,  # CLIP/T5 dim
            init_strategy=config.prompt.init_strategy,
            init_std=config.prompt.init_std,
            device=device,
        )

        self.reconstructor = SparseGlobalReconstructor(
            config.prompt.prompt_length, 512
        )
        self.checkpoint_mgr = CheckpointManager(
            config.checkpoint.save_dir, config.checkpoint.metric
        )

        self.training_history: list[dict] = []
        self.early_stop_counter = 0
        self.best_metric_value = -float("inf")
        self.is_training = False
        self.current_round = 0
        self.total_comm_bytes = 0
        self.last_fusion_weights = [0.33, 0.33, 0.33] # Default uniform
        self.version = "afsple_v1"

        # Log dir
        Path(config.logging.log_dir).mkdir(parents=True, exist_ok=True)

    def get_global_prompt(self) -> torch.Tensor:
        return self.global_prompt_manager.global_prompt.embedding.detach().clone()

    def aggregate_round(
        self,
        client_updates: list[SparsePromptUpdate],
        client_sizes: list[int] | None = None,
        round_metrics: list[dict] | None = None,
    ) -> dict:
        """
        Steps 3-8: Collect updates, reconstruct, update Pg, log.
        """
        current_Pg = self.get_global_prompt()
        client_weights = None
        if client_sizes:
            total = sum(client_sizes)
            client_weights = [s / total for s in client_sizes]

        # Steps 4-6: Sparse FedAvg reconstruction
        new_Pg = self.reconstructor.aggregate(current_Pg, client_updates, client_weights)

        # Stage 4 Robustness (Phase 3.3): Aggressive Domain Clamp
        # Clamps extreme prompt values (>5.0) to maintain stable attention manifold.
        if torch.abs(new_Pg).max() > 5.0:
            logger.warning(f"Prompt distribution extreme (max={torch.abs(new_Pg).max():.2f}). Clamping to [-5.0, 5.0]")
            new_Pg = torch.clamp(new_Pg, -5.0, 5.0)

        # Step 7: Update global prompt
        with torch.no_grad():
            self.global_prompt_manager.global_prompt.embedding.copy_(new_Pg.to(self.device))

        # Accumulate communication stats
        if round_metrics:
            round_bytes = sum(m.get("comm_bytes", 0) for m in round_metrics)
            self.total_comm_bytes += round_bytes

        coverage = self.reconstructor.get_coverage_stats()
        logger.info(f"Round aggregation done. Coverage: {coverage['mean_coverage']:.2f} avg tokens updated")

        return {"coverage": coverage, "total_comm_bytes": self.total_comm_bytes}

    def log_round(self, round_num: int, metrics: dict) -> None:
        """Step 8: Log round-wise metrics."""
        entry = {
            "round": round_num, 
            "timestamp": time.time(), 
            "topk_ratio": float(self.config.topk.k_ratio),
            **metrics
        }
        self.training_history.append(entry)

        log_path = Path(self.config.logging.log_dir) / "training_history.json"
        with open(log_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        append_path = Path(self.config.logging.log_dir) / "training_log.json"
        with open(append_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def check_early_stopping(self, metric_value: float) -> bool:
        """Check if training should stop."""
        cfg = self.config.early_stopping
        if metric_value > self.best_metric_value + cfg.min_delta:
            self.best_metric_value = metric_value
            self.early_stop_counter = 0
            return False
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= cfg.patience:
                logger.warning(f"Early stopping triggered after {cfg.patience} rounds without improvement.")
                return True
            return False

    def save_checkpoint(self, round_num: int, eval_metric: float | None = None):
        """
        Saves a persistent .pt checkpoint with versioning and device safety.
        """
        Pg_cpu = self.global_prompt_manager.global_prompt.embedding.detach().cpu()
        
        state = {
            "version": self.version,
            "round": round_num,
            "global_prompt": Pg_cpu,
            "training_history": self.training_history,
            "fusion_weights": self.last_fusion_weights,
            "config": self.config.model_dump() if hasattr(self.config, 'model_dump') else {},
            "config_hash": hash(str(self.config)),
            "timestamp": time.time(),
        }
        
        is_best = eval_metric is not None and eval_metric > self.best_metric_value
        self.checkpoint_mgr.save(state, round_num, eval_metric, is_best)
        logger.info(f"💾 Checkpoint saved for round {round_num} (CPU Offloaded)")
