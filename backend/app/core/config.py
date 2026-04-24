"""Central configuration loader using Pydantic + YAML."""
from __future__ import annotations
import os
import yaml
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DataConfig(BaseModel):
    train_dataset: str = "ms_coco"
    eval_dataset: str = "flickr30k"
    data_dir: str = "./data"
    num_clients: int = 10
    dirichlet_alpha: float = 0.5
    train_split: float = 0.9
    image_size: int = 224


class ModelConfig(BaseModel):
    clip_model: str = "ViT-B/32"
    text_decoder: str = "google/flan-t5-small"
    hidden_dim: int = 512
    max_seq_len: int = 64


class PromptConfig(BaseModel):
    prompt_length: int = 16
    init_strategy: Literal["random", "token_embed"] = "random"
    init_std: float = 0.02


class FusionConfig(BaseModel):
    strategy: Literal["static", "learnable", "dynamic"] = "dynamic"
    static_alpha: float = 0.33
    static_beta: float = 0.33
    static_gamma: float = 0.34


class TopKConfig(BaseModel):
    k_ratio: float = 0.3
    selection_method: Literal["gradient", "norm", "attention"] = "gradient"
    adaptive_k: bool = False


class TrainingConfig(BaseModel):
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_rounds: int = 30
    local_epochs: int = 3
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    num_workers: int = 2
    device: str = "auto"


class LossConfig(BaseModel):
    lambda1: float = 1.0
    lambda2: float = 0.5
    clip_temperature: float = 0.07


class FederatedConfig(BaseModel):
    min_clients: int = 5
    min_eval_clients: int = 3
    fraction_fit: float = 0.8
    fraction_evaluate: float = 0.5


class CheckpointConfig(BaseModel):
    save_dir: str = "./checkpoints"
    save_every: int = 5
    metric: str = "cider"


class EarlyStoppingConfig(BaseModel):
    patience: int = 5
    metric: str = "cider"
    min_delta: float = 0.001


class LoggingConfig(BaseModel):
    log_dir: str = "./logs"
    log_level: str = "INFO"
    json_logs: bool = True
    csv_export: bool = True


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


class ProjectConfig(BaseModel):
    name: str = "AFSPL"
    version: str = "1.0.0"
    seed: int = 42
    deterministic: bool = True


class AFSPLConfig(BaseModel):
    project: ProjectConfig = ProjectConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    prompt: PromptConfig = PromptConfig()
    fusion: FusionConfig = FusionConfig()
    topk: TopKConfig = TopKConfig()
    training: TrainingConfig = TrainingConfig()
    loss: LossConfig = LossConfig()
    federated: FederatedConfig = FederatedConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    logging: LoggingConfig = LoggingConfig()
    api: APIConfig = APIConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AFSPLConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_device(self) -> str:
        import torch
        if self.training.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.training.device


# Global config instance
_config: Optional[AFSPLConfig] = None


def get_config() -> AFSPLConfig:
    global _config
    if _config is None:
        cfg_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
        if cfg_path.exists():
            _config = AFSPLConfig.from_yaml(cfg_path)
        else:
            _config = AFSPLConfig()
    return _config


def set_config(cfg: AFSPLConfig) -> None:
    global _config
    _config = cfg
