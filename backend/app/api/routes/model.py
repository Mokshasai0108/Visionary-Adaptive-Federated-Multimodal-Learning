from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pathlib import Path

from app.core.config import get_config
from app.models.loader import load_latest_checkpoint, load_checkpoint_by_name
from app.models.registry import ModelRegistry
from app.services.training_service import get_state
from models.multimodal_model import AFSPLModel

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/status")
def model_status():
    model, meta = ModelRegistry.get()
    return {
        "loaded": model is not None,
        "round": meta.get("round", 0),
        "version": meta.get("version", "unknown"),
        "model_name": meta.get("model_name", "default"),
        "fusion_weights": meta.get("fusion_weights", [0.33, 0.33, 0.34]),
    }


@router.get("/list")
def list_models():
    base = Path("backend/checkpoints")
    if not base.exists():
        return {"models": []}
    models = sorted([d.name for d in base.iterdir() if d.is_dir()])
    return {"models": models}


@router.post("/switch")
def switch_model(model_name: str = Query(..., description="Model directory name to load")):
    state = get_state()
    if state.is_training:
        raise HTTPException(status_code=409, detail="Cannot switch model during active training")

    cfg = get_config()
    device = cfg.get_device()
    model = AFSPLModel(
        clip_model_name="openai/clip-vit-base-patch32",
        decoder_model_name="google/flan-t5-small",
        prompt_length=cfg.prompt.prompt_length,
        device=device,
    )
    model.to(device)

    if not load_checkpoint_by_name(model, model_name, ckpt_dir=cfg.checkpoint.save_dir):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found or failed to load")

    return {"status": "switched", "model": model_name}


@router.post("/reload")
def reload_model():
    state = get_state()
    if state.is_training:
        raise HTTPException(status_code=409, detail="Training active → skipping reload")

    cfg = get_config()
    device = cfg.get_device()
    model = AFSPLModel(
        clip_model_name="openai/clip-vit-base-patch32",
        decoder_model_name="google/flan-t5-small",
        prompt_length=cfg.prompt.prompt_length,
        device=device,
    )
    model.to(device)

    loaded = load_latest_checkpoint(model, ckpt_dir=cfg.checkpoint.save_dir)
    if not loaded:
        logger.warning("Model reload requested but no checkpoint found. Model registered in cold-start mode.")
        return {
            "status": "reloaded",
            "loaded": False,
            "message": "No checkpoint found. Cold-start model registered.",
        }

    return {"status": "reloaded", "loaded": True}
