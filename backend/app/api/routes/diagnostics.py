"""Diagnostics endpoint."""
from fastapi import APIRouter
from app.schemas.train import DiagnosticsResponse
from app.services.training_service import get_server, get_state

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


@router.get("", response_model=DiagnosticsResponse)
async def get_diagnostics():
    server = get_server()
    state = get_state()
    warnings = list(state.warnings[-20:])

    coverage = server.reconstructor.get_coverage_stats() if server else {}
    history = server.training_history[-10:] if server else []

    # Approximate client drift as variance of losses across rounds
    losses = [h.get("train_loss", 0) for h in history]
    drift = [abs(losses[i] - losses[i-1]) for i in range(1, len(losses))] if len(losses) > 1 else [0.0]

    # Prompt collapse risk: if global prompt norm is very small
    collapse_risk = 0.0
    if server:
        norm = server.global_prompt_manager.global_prompt.embedding.norm().item()
        collapse_risk = max(0.0, 1.0 - norm / 5.0)

    return DiagnosticsResponse(
        client_drift=drift,
        prompt_collapse_risk=round(collapse_risk, 4),
        sparse_coverage=coverage,
        communication_history=[{"round": h["round"], "comm_bytes": h.get("comm_bytes", 0)} for h in history],
        warnings=warnings,
    )
