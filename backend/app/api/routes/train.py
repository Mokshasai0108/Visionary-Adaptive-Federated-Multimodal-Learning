"""Training control endpoints."""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.schemas.train import TrainStartRequest, TrainStatusResponse
from app.services.training_service import start_training_background, get_state, get_server
from app.core.config import get_config

router = APIRouter(prefix="/train", tags=["training"])


@router.post("/start")
async def start_training(request: TrainStartRequest, background_tasks: BackgroundTasks):
    state = get_state()
    if state.is_training:
        raise HTTPException(status_code=409, detail="Training already in progress.")
    config = get_config()
    background_tasks.add_task(start_training_background, request, config)
    return {"message": "Training started", "rounds": request.num_rounds}


@router.get("/status", response_model=TrainStatusResponse)
async def get_status():
    state = get_state()
    return TrainStatusResponse(
        status=state.status,
        current_round=state.current_round,
        total_rounds=state.total_rounds,
        is_training=state.is_training,
        best_metric=state.best_metric,
        current_metrics=state.current_metrics,
        total_comm_bytes=state.total_comm_bytes,
    )


@router.post("/stop")
async def stop_training():
    state = get_state()
    state.is_training = False
    state.status = "stopped"
    return {"message": "Training stop requested."}
