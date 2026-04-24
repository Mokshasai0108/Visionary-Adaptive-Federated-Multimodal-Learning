import torch
from pathlib import Path
from app.models.registry import ModelRegistry

def is_non_trivial(tensor: torch.Tensor) -> bool:
    """Verify that a tensor has been trained (non-zero variance)."""
    if tensor.numel() <= 1: return True
    return torch.std(tensor).item() > 1e-6

def load_latest_checkpoint(model, ckpt_dir: str = "backend/checkpoints", training_mode: bool = False):
    """Load the most recent checkpoint (best if exists) and register it."""
    ckpt_path = Path(ckpt_dir)
    
    best_path = next(ckpt_path.rglob("ckpt_best.pt"), None)
    path = best_path if (best_path and best_path.exists()) else sorted(ckpt_path.rglob("ckpt_round_*.pt"))[-1] if list(ckpt_path.rglob("ckpt_round_*.pt")) else None
    
    if not path:
        print("[LOAD] No checkpoints found. Cold starting.")
        ModelRegistry.set(model, {
            "global_prompt": None,
            "fusion_weights": {"alpha": 0.33, "beta": 0.33, "gamma": 0.34},
            "round": 0,
            "version": "cold_start",
            "model_name": "default",
        })
        return False

    try:
        data = torch.load(path, map_location=model.device)
        version = data.get("version", "unknown")
        
        # 1. State Loading (Phase 3 Hardening)
        if "model_state" in data:
            # Phase 3 Sanity Checks
            if version == "afsple_v3":
                has_cross_attn = any("encdecattention" in k.lower() for k in data["model_state"].keys())
                assert has_cross_attn, "❌ Phase 3 checkpoint missing cross-attention weights!"
                
                # Verify weights are NOT default (Triviality Check)
                trained = any(is_non_trivial(v) for k, v in data["model_state"].items() if "encdecattention" in k.lower())
                if not trained: 
                    print(f"[LOAD WARNING] Grounding weights in {path} appear untrained (trivial variance).")

            model.load_state_dict(data["model_state"], strict=True)
            
            # Re-link Bridge conditionally
            if version == "afsple_v3" and training_mode:
                model.apply_selective_unfreeze()
            else:
                model.eval()

        # 2. Prompt & Registry Update
        Pg = data.get("Pg")
        if Pg is not None: Pg = Pg.to(model.device)
        
        ModelRegistry.set(model, {
            "global_prompt": Pg,
            "fusion_weights": data.get("fusion_weights", {"alpha": 0.33, "beta": 0.33, "gamma": 0.34}),
            "round": data.get("round", 0),
            "version": version,
            "model_name": data.get("model_name", "default"),
        })
        print(f"[LOAD] Success ({version}) -> {path} | Bridge: {'ACTIVE (Train)' if training_mode and version=='afsple_v3' else 'LOCKED (Infer)'}")
        return True
    except Exception as e:
        print(f"[LOAD ERROR] {e}")
        return False


def load_checkpoint_by_name(model, model_name: str, ckpt_dir: str = "backend/checkpoints", training_mode: bool = False):
    """Load a checkpoint from a named model directory and register it."""
    ckpt_path = Path(ckpt_dir) / model_name / "ckpt_best.pt"
    if not ckpt_path.exists():
        print(f"[LOAD] Model not found: {model_name}")
        return False

    try:
        data = torch.load(ckpt_path, map_location=model.device)
        version = data.get("version", "unknown")
        
        if "model_state" in data:
            model.load_state_dict(data["model_state"], strict=True)
            if version == "afsple_v3" and training_mode:
                model.apply_selective_unfreeze()
            else:
                model.eval()

        Pg = data["Pg"].to(model.device)
        ModelRegistry.set(model, {
            "global_prompt": Pg,
            "fusion_weights": data.get("fusion_weights", {"alpha": 0.33, "beta": 0.33, "gamma": 0.34}),
            "round": data.get("round", 0),
            "version": version,
            "model_name": model_name,
        })
        print(f"[LOAD] Success → {model_name}")
        return True
    except Exception as e:
        print(f"[LOAD ERROR] {e}")
        return False
