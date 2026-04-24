"""
run_inference_cli.py
--------------------
Generate a caption for any local image file via command line.
No browser or API server needed.

Usage:
    python scripts/run_inference_cli.py --image path/to/photo.jpg
    python scripts/run_inference_cli.py --image photo.jpg --checkpoint checkpoints/best_checkpoint.pt
    python scripts/run_inference_cli.py --image photo.jpg --num_captions 3
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="AFSPL Caption Generator")
    parser.add_argument("--image",        required=True, help="Path to input image")
    parser.add_argument("--checkpoint",   default=None,  help="Path to checkpoint .pt file")
    parser.add_argument("--num_captions", type=int, default=1)
    parser.add_argument("--prompt_length",type=int, default=16)
    parser.add_argument("--fusion",       default="dynamic",
                        choices=["static","learnable","dynamic"])
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"ERROR: Image not found: {img_path}")
        sys.exit(1)

    from app.core.config import get_config
    from app.core.reproducibility import set_global_seed
    from models.multimodal_model import AFSPLModel
    from prompts.prompt_manager import PromptManager
    from prompts.fusion import AdaptiveFusionModule

    cfg    = get_config()
    device = cfg.get_device()
    set_global_seed(cfg.project.seed)

    print(f"Device: {device}")
    print(f"Loading models (first run downloads ~300 MB from HuggingFace)...")

    model = AFSPLModel(
        prompt_length=args.prompt_length,
        device=device,
    )
    model.to(device)

    pm     = PromptManager(args.prompt_length, 512, device=device)
    fusion = AdaptiveFusionModule(args.fusion, 512, 512).to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "global_prompt" in ckpt:
                pm.set_global_prompt(ckpt["global_prompt"])
                print(f"Loaded checkpoint from {ckpt_path}")
        else:
            print(f"WARNING: checkpoint not found at {ckpt_path}, using random prompts")

    # Load and preprocess image
    pil_img = Image.open(img_path).convert("RGB")
    pixel_values = model.vision_encoder.preprocess_images([pil_img]).to(device)

    # Encode
    with torch.no_grad():
        E_image = model.encode_image(pixel_values)
        fused, weights = fusion(
            pm.global_prompt.embedding,
            pm.local_prompt.embedding,
            pm.modality_prompt.embedding,
            E_image=E_image,
        )

    # Generate
    print(f"\nImage: {img_path.name}")
    print(f"Fusion weights: alpha={weights[0]:.3f}  beta={weights[1]:.3f}  gamma={weights[2]:.3f}")
    print(f"\nGenerated caption{'s' if args.num_captions>1 else ''}:")
    res = model.generate_from_pixels(pixel_values, Pg=fused)
    caption = res.get("caption", "N/A")
    print(f"  [1] {caption}")
    
    if "attribution" in res:
        attr = res["attribution"]
        print(f"\nAttribution: Prefix={attr.get('prefix',0):.2f} Prompts={attr.get('prompts',0):.2f} Vision={attr.get('vision',0):.2f}")


if __name__ == "__main__":
    main()
