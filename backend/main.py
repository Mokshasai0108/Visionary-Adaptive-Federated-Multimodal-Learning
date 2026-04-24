"""AFSPL FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import train, infer, metrics, prompts, diagnostics, health
from app.api.routes.model import router as model_router
from app.core.config import get_config
from app.models.loader import load_latest_checkpoint
from app.models.registry import ModelRegistry
from models.multimodal_model import AFSPLModel

app = FastAPI(
    title="AFSPL API",
    description="Adaptive Federated Soft Prompt Learning — Research Backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(train.router)
app.include_router(infer.router)
app.include_router(metrics.router)
app.include_router(prompts.router)
app.include_router(diagnostics.router)
app.include_router(health.router)
app.include_router(model_router)


@app.on_event("startup")
async def startup():
    cfg = get_config()
    logger.info(f"AFSPL API started | project={cfg.project.name} v{cfg.project.version}")
    device = cfg.get_device()

    # Load or cold-start the multimodal model at service startup.
    model = AFSPLModel(
        clip_model_name="openai/clip-vit-base-patch32",
        decoder_model_name="google/flan-t5-small",
        prompt_length=cfg.prompt.prompt_length,
        device=device,
    )
    model.to(device)
    loaded = load_latest_checkpoint(model, ckpt_dir=cfg.checkpoint.save_dir)

    if not loaded:
        ModelRegistry.set(model, {"round": 0, "version": "cold_start"})
        logger.warning("[WARNING] No trained model loaded. System in cold-start mode.")
    else:
        logger.info("[SYSTEM] Loaded existing checkpoint on startup.")

    import nltk
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    uvicorn.run("main:app", host=cfg.api.host, port=cfg.api.port, reload=cfg.api.reload)
