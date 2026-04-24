"""Metrics and comparison endpoints."""
import json
from pathlib import Path
from fastapi import APIRouter
from app.schemas.train import MetricsResponse
from app.services.training_service import get_server
from app.core.config import get_config

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("", response_model=MetricsResponse)
async def get_metrics():
    server = get_server()
    history = server.training_history if server else []
    config = get_config()
    comp_path = Path(config.logging.log_dir) / "comparison_summary.json"
    comparison = json.loads(comp_path.read_text()) if comp_path.exists() else None
    return MetricsResponse(training_history=history, comparison_table=comparison)
