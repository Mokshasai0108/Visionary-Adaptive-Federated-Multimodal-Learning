from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
import torch
from loguru import logger
from app.models.registry import ModelRegistry
from models.diffusion_mirror import AFSPLMirrorSuite
from app.services.resource_manager import MultimodalResourceManager
import numpy as np

router = APIRouter(prefix="/mirror", tags=["mirror"])

# Initialize suite on CPU
_mirror_suite = AFSPLMirrorSuite(device="cuda" if torch.cuda.is_available() else "cpu")
_res_manager = MultimodalResourceManager()

class MirrorRequest(BaseModel):
    prompt: str
    alpha: float = 0.5
    seed: int = 42

@router.post("/dream")
async def dream(request: MirrorRequest):
    """
    Phase 3 Mirror Endpoint.
    Generates Baseline vs. Steered images and calculates research metrics.
    """
    # 1. Get Models from Registry
    model, meta = ModelRegistry.get()
    if not model:
        raise HTTPException(status_code=503, detail="Captioning model not initialized.")

    # 2. VRAM Swap: Activate Diffusion
    try:
        _res_manager.activate_diffusion(model, _mirror_suite.pipe)
    except Exception as e:
        logger.error(f"VRAM Swap failed: {e}")
        raise HTTPException(status_code=500, detail="Resource swapper failure.")

    try:
        # 3. Get AFSPL knowledge (Pg)
        Pg = meta.get("global_prompt")
        
        # 4. Generate Side-by-Side
        torch.manual_seed(request.seed)
        img_base, img_steer = _mirror_suite.dream(
            prompt=request.prompt,
            Pg_emb=Pg,
            alpha=request.alpha
        )

        # 5. Calculate Research Diagnostics (Provable Contribution)
        # We need model on GPU to score the images fast
        _res_manager.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(_res_manager.device)

        with torch.no_grad():
            score_base = float(model.compute_clip_similarity(img_base, request.prompt))
            score_steer = float(model.compute_clip_similarity(img_steer, request.prompt))
            
            # Semantic Shift Metric (Delta)
            # Encode images to embeddings to measure distance
            emb_base = model.encode_image(
                model.clip_processor(images=img_base, return_tensors="pt").pixel_values.to(_res_manager.device)
            )
            emb_steer = model.encode_image(
                model.clip_processor(images=img_steer, return_tensors="pt").pixel_values.to(_res_manager.device)
            )
            
            delta = float(1.0 - torch.nn.functional.cosine_similarity(emb_base, emb_steer).item())

        return {
            "baseline": _mirror_suite.pil_to_base64(img_base),
            "steered": _mirror_suite.pil_to_base64(img_steer),
            "metrics": {
                "alpha": request.alpha,
                "clip_base": round(score_base, 4),
                "clip_steer": round(score_steer, 4),
                "delta": round(delta, 4)
            },
            "ablation": meta.get("ablation", {})
        }

    except Exception as e:
        logger.error(f"Mirror Dream failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # We stay in diffusion mode until something else triggers captioning
        pass
