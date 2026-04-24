"""Inference endpoint."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.train import InferResponse
from app.services.training_service import get_server, get_state
import torch

router = APIRouter(prefix="/infer", tags=["inference"])


from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import torch
import io
from loguru import logger

from app.models.registry import ModelRegistry
from app.services.runtime_locks import can_infer_on_gpu
from app.schemas.train import InferResponse

@router.post("")
async def run_inference(
    image: UploadFile = File(...),
    text_prompt: str = Form(default=""),
    max_length: int = Form(default=32)
):
    """
    Multimodal Inference with Phase 2.6 Diagnostics (Multipart Support).
    Supports direct image upload from UI.
    """
    model, meta = ModelRegistry.get()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded or training in progress.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    try:
        # Load and process image from UploadFile
        img_data = await image.read()
        raw_image = Image.open(io.BytesIO(img_data)).convert("RGB")
        pixel_values = model.clip_processor(images=raw_image, return_tensors="pt").pixel_values.to(device)

        # Handle Prefix Steering
        prefix_ids = None
        if text_prompt.strip():
            prefix_ids = model.tokenizer(
                text_prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)

        # Acquire Global weights
        Pg = meta.get("global_prompt")
        if Pg is not None:
            Pg = Pg.to(device)

        # 🚀 Phase 2.6: Run Hardened Generation
        results = model.generate_from_pixels(
            pixel_values, 
            Pg=Pg, 
            prefix_ids=prefix_ids,
            max_length=max_length
        )
        
        caption = results["caption"]
        confidence = results["confidence"]
        attribution = results["attribution"]
        
        # Calculate Semantic Alignment (CLIP Score)
        clip_score = model.compute_clip_similarity(pixel_values, caption)
        
        # 🧯 Phase 2.6: Scientific Fallback Policy
        if confidence < 0.15 and float(clip_score) < 0.15:
            caption = "AFSPL low-confidence: visual-text alignment uncertain."

        return {
            "caption": caption,
            "clip_similarity": round(float(clip_score), 4),
            "confidence": confidence,
            "attribution": attribution,
            "ablation": results.get("ablation", {}),
            "fusion_weights": meta.get("fusion_weights", [0.33, 0.33, 0.34]),
            "metadata": {
                "round": meta.get("round", 0),
                "token_count": results.get("token_count", 0)
            }
        }

    except Exception as e:
        logger.error(f"Inference API Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Inference API Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
