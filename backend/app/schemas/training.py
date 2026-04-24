"""Pydantic schemas for API request/response."""
from __future__ import annotations
from typing import Optional, Literal, Any
from pydantic import BaseModel, Field


class TrainingStartRequest(BaseModel):
    num_rounds: int = Field(default=20, ge=1, le=100)
    num_clients: int = Field(default=10, ge=1, le=50)
    prompt_length: int = Field(default=16, ge=10, le=20)
    fusion_strategy: Literal["static", "learnable", "dynamic"] = "dynamic"
    topk_ratio: float = Field(default=0.3, ge=0.1, le=1.0)
    lambda1: float = Field(default=1.0, ge=0.0)
    lambda2: float = Field(default=0.5, ge=0.0)
    seed: int = 42
    use_synthetic_data: bool = True  # set False when real data available


class TrainingStatusResponse(BaseModel):
    status: Literal["idle", "running", "completed", "error"]
    current_round: int
    total_rounds: int
    progress_pct: float
    latest_metrics: Optional[dict] = None
    error_message: Optional[str] = None


class InferenceRequest(BaseModel):
    caption_hint: Optional[str] = None
    use_best_checkpoint: bool = True


class InferenceResponse(BaseModel):
    generated_caption: str
    clip_similarity: Optional[float] = None
    fusion_weights: Optional[dict] = None


class MetricsResponse(BaseModel):
    history: list[dict]
    comparison_table: list[dict]
    summary: dict


class PromptsResponse(BaseModel):
    global_prompt_norm: float
    local_prompt_norm: float
    modality_prompt_norm: float
    prompt_length: int
    hidden_dim: int
    fusion_weights_avg: Optional[list[float]] = None
    token_coverage: Optional[list[int]] = None


class DiagnosticsResponse(BaseModel):
    client_drift_scores: list[float]
    prompt_collapse_risk: float
    token_coverage_pct: float
    communication_anomalies: list[str]
    round_latencies: list[float]
