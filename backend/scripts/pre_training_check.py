"""
pre_training_check.py
--------------------
Phase 2.6 Hardening: Final diagnostic check before the big 8-hour training run.
Verifies VRAM, Data Paths, Gradient Flow, and Tokenization Sanity.
"""
import sys
import os
from pathlib import Path
import torch
from loguru import logger

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import get_config
from models.multimodal_model import AFSPLModel
from training.dataset import CaptionDataset, load_coco_json
from training.losses import AFSPLLoss
from torch.utils.data import DataLoader

def run_health_check():
    logger.info("🚀 Starting AFSPL Pre-Training Health Check...")
    config = get_config()
    device = config.get_device()
    
    # 1. GPU / VRAM Check
    logger.info("--- Stage 1: Environment ---")
    if torch.cuda.is_available():
        free_bytes = torch.cuda.mem_get_info()[0]
        free_gb = free_bytes / 1e9
        logger.info(f"[GPU] Found {torch.cuda.get_device_name(0)}")
        logger.info(f"[GPU] Free VRAM: {free_gb:.2f} GB")
        if free_gb < 2.0:
            logger.error("❌ CRITICAL: Not enough free VRAM (>2.0GB required for T5-Small + CLIP)")
            return False
        logger.success("Environment: PASS")
    else:
        logger.warning("[GPU] CUDA not available. Running on CPU (Testing only).")

    # 2. Data Paths Check
    logger.info("--- Stage 2: Data Paths ---")
    coco_path = Path(config.data.data_dir) / "coco" / "captions_train2017.json"
    if not coco_path.exists():
        logger.error(f"❌ CRITICAL: MS COCO metadata not found at {coco_path}")
        return False
    
    # Try one sample loading
    try:
        samples = load_coco_json(str(coco_path), limit=100)
        if not samples:
            logger.error("❌ CRITICAL: MS COCO metadata found, but NO VALID IMAGE SAMPLES were loaded (check path resolution)")
            return False
        logger.info(f"[DATA] Sample check: Successfully resolved {len(samples)} image paths")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Dataset loading failed: {e}")
        return False
        
    logger.success("Data Paths: PASS")

    # 3. Model & Gradient Flow Check
    logger.info("--- Stage 3: Model & Gradient Flow ---")
    model = AFSPLModel(
        prompt_length=config.prompt.prompt_length,
        device=device
    ).to(device)
    
    # Enforce Freeze
    for n, p in model.named_parameters():
        if "clip" in n.lower() or "decoder" in n.lower():
            p.requires_grad = False
            
    trainable = [p for p in model.get_trainable_params() if p.requires_grad]
    logger.info(f"Trainable Parameters: {len(trainable)} tensors")
    
    # Dummy Forward/Backward
    try:
        # Real shapes
        img = torch.randn(1, 3, 224, 224).to(device)
        txt = torch.randint(0, 1000, (1, 32)).to(device)
        mask = torch.ones(1, 32).to(device)
        fused = torch.randn(1, config.prompt.prompt_length, 512).to(device)
        
        model.train()
        loss_fn = AFSPLLoss()
        
        # Step
        loss_ce, logits = model(img, txt, mask, fused, labels=txt)
        total_loss, _ = loss_fn(logits, txt)
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error("❌ CRITICAL: NaN or Inf loss detected")
            return False
            
        total_loss.backward()
        
        # Check grads
        for p in trainable:
            if p.grad is None:
                logger.error(f"❌ CRITICAL: Missing gradient flow for trainable parameter {p.shape}")
                return False
        
        logger.success("Model & Gradient Flow: PASS")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Forward/Backward pass failed: {e}")
        return False

    # 4. Projection & Tokenization Sanity
    logger.info("--- Stage 4: Projection Sanity ---")
    with torch.no_grad():
        feats = model.encode_image(img)
        vt = model.projector(feats) 
        logger.info(f"[SANITY] vision_tokens shape: {vt.shape} (Expected [1, 4, 512])")
        assert vt.shape == (1, 4, 512), "❌ CRITICAL: Vision tokens not 4"
    logger.success("Projection Sanity: PASS")

    # 5. Fusion Adaptivity Check (Diagram Alignment: α, β, γ)
    logger.info("--- Stage 5: Fusion Adaptivity ---")
    from prompts.fusion import AdaptiveFusionModule
    fusion_module = AdaptiveFusionModule("dynamic", 512, 512).to(device)
    
    with torch.no_grad():
        weights_list = []
        for i in range(3):
            # Vary the inputs slightly to ensure adaptivity
            d_img = torch.randn(1, 512).to(device) + i * 0.1
            d_txt = torch.randn(1, 512).to(device)
            f_Pg = torch.randn(1, config.prompt.prompt_length, 512).to(device)
            f_Pl = torch.randn(1, config.prompt.prompt_length, 512).to(device)
            f_Pm = torch.randn(1, config.prompt.prompt_length, 512).to(device)
            
            _, telemetry = fusion_module(f_Pg, f_Pl, f_Pm, d_img, d_txt)
            weights = torch.tensor([telemetry["alpha"], telemetry["beta"], telemetry["gamma"]])
            weights_list.append(weights)
            
        w_tensor = torch.stack(weights_list)
        logger.info(f"[FUSION] Mean α, β, γ: {w_tensor.mean(dim=0).cpu().numpy()}")
        
        assert (w_tensor > 0).all(), "❌ Negative fusion weights detected"
        assert torch.allclose(w_tensor.sum(dim=-1), torch.ones(3), atol=1e-5), "❌ Fusion weights not normalized"
        
        # Check variance (adaptivity)
        variance = w_tensor.var(dim=0).mean()
        logger.info(f"[FUSION] Sensitivity Variance: {variance:.6f}")
        assert variance > 1e-4, f"❌ Fusion collapsed (variance {variance:.6f} < 1e-4). Architecture is not dynamic."
    
    logger.success("Fusion Adaptivity: PASS")

    # 6. Shared Multimodal Space Consistency Check
    logger.info("--- Stage 6: Shared Space Consistency ---")
    with torch.no_grad():
        d_img = torch.randn(1, 3, 224, 224).to(device)
        d_fused = torch.randn(1, config.prompt.prompt_length, 512).to(device)
        d_prefix = torch.randint(0, 1000, (1, 8)).to(device)
        
        # Path 1: Direct shared representation
        shared_space = model.get_shared_representation(d_img, d_fused, d_prefix)
        
        # Path 2: Test internal consistency by running one more time
        # (This confirms the method is deterministic for the same parameters/state)
        shared_space_v2 = model.get_shared_representation(d_img, d_fused, d_prefix)
        
        diff = (shared_space - shared_space_v2).abs().max()
        logger.info(f"[SHARED] Consistency Delta: {diff:.6f}")
        assert torch.allclose(shared_space, shared_space_v2, atol=1e-6), "❌ Shared Multimodal Space is non-deterministic!"
        
        # Verify Norms
        f_norm = shared_space.norm(dim=-1).mean()
        logger.info(f"[SHARED] Mean Latent Norm: {f_norm:.4f}")
        assert 0.5 < f_norm < 1.5, f"❌ Out of range latent Norm: {f_norm:.4f} (Expected near 1.0 due to SymmNorm)"

    logger.success("Shared Space Consistency: PASS")

    logger.info("\n" + "="*40)
    logger.success("✅ ALL PRE-TRAINING CHECKS PASSED")
    logger.info("The system is in RUNNING CONDITION for tomorrow.")
    logger.info("="*40)
    return True

if __name__ == "__main__":
    success = run_health_check()
    if not success:
        sys.exit(1)
