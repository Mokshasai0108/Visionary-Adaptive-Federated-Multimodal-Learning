"""Pydantic schemas for training endpoints."""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class TrainStartRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    num_rounds: int = Field(default=30, ge=1, le=100)
    num_clients: int = Field(default=10, ge=2, le=50)
    fusion_strategy: Literal["static", "learnable", "dynamic"] = "dynamic"
    prompt_length: int = Field(default=16, ge=10, le=20)
    batch_size: int = Field(default=32, ge=4, le=128)
    learning_rate: float = Field(default=1e-4, gt=0)
    lambda1: float = 1.0
    lambda2: float = 0.5
    k_ratio: float = Field(default=0.3, ge=0.1, le=1.0)
    subset_ratio: float = Field(default=1.0, ge=0.01, le=1.0)
    use_test_mode: bool = False
    seed: int = 42
    model_name: str = "default"
    is_recovery: bool = False
    checkpoint_path: Optional[str] = None


class TrainStatusResponse(BaseModel):
    status: str
    current_round: int
    total_rounds: int
    is_training: bool
    best_metric: float
    current_metrics: Optional[dict] = None
    total_comm_bytes: int = 0


class InferRequest(BaseModel):
    caption_prefix: Optional[str] = None
    max_new_tokens: int = 50


class InferResponse(BaseModel):
    caption: str
    clip_similarity: float
    confidence: float
    attribution: dict
    fusion_weights: dict
    metadata: Optional[dict] = None


class MetricsResponse(BaseModel):
    training_history: list[dict]
    comparison_table: Optional[list[dict]] = None


class PromptStateResponse(BaseModel):
    global_prompt_norm: float
    local_prompt_norm: float
    modality_prompt_norm: float
    prompt_length: int
    hidden_dim: int
    last_fusion_weights: Optional[dict] = None
    token_coverage: Optional[list[int]] = None


class DiagnosticsResponse(BaseModel):
    client_drift: list[float]
    prompt_collapse_risk: float
    sparse_coverage: dict
    communication_history: list[dict]
    warnings: list[str]
