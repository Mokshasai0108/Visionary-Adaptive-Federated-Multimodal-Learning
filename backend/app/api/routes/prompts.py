"""Prompt state inspection endpoint."""
import numpy as np
import torch
from fastapi import APIRouter
from app.schemas.train import PromptStateResponse
from app.services.training_service import get_server, get_state

router = APIRouter(prefix="/prompts", tags=["prompts"])


@router.get("", response_model=PromptStateResponse)
async def get_prompt_state():
    server = get_server()
    state = get_state()
    if server is None:
        return PromptStateResponse(
            global_prompt_norm=0.0, local_prompt_norm=0.0, modality_prompt_norm=0.0,
            prompt_length=0, hidden_dim=0,
        )
    pm = server.global_prompt_manager
    g = pm.global_prompt.embedding.detach()
    coverage = server.reconstructor.get_coverage_stats()
    return PromptStateResponse(
        global_prompt_norm=round(g.norm().item(), 4),
        local_prompt_norm=0.0,
        modality_prompt_norm=0.0,
        prompt_length=pm.prompt_length,
        hidden_dim=pm.hidden_dim,
        last_fusion_weights=state.last_fusion_weights,
        token_coverage=coverage["coverage_per_token"],
    )
