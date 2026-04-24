import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForText2Image
from loguru import logger
import PIL.Image
import io
import base64

class AFSPLMirrorSuite:
    """
    Phase 3: Bidirectional Mirror Engine.
    Integrates Stable Diffusion Turbo with AFSPL-Steering.
    """
    _instance = None
    
    def __init__(self, model_id: str = "stabilityai/sd-turbo", device: str = "cpu"):
        logger.info(f"Loading Diffusion Engine: {model_id}")
        self.device = device
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
            variant="fp16" if "cuda" in str(device) else None
        )
        # Move to CPU initially; managed by ResourceManager
        self.pipe.to("cpu")
        self.pipe.set_progress_bar_config(disable=True)
        logger.info("Diffusion Engine loaded (resting on CPU).")

    @torch.no_grad()
    def dream(self, prompt: str, Pg_emb: torch.Tensor, alpha: float = 0.5):
        """
        Generates both Baseline and AFSPL-Steered images.
        """
        device = self.pipe.device
        dtype = self.pipe.unet.dtype
        
        # 1. Get Base Text Embeddings
        # We pre-calculate embeddings so we can manipulate them
        prompt_inputs = self.pipe.tokenizer(
            prompt, 
            padding="max_length", 
            max_length=self.pipe.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        text_emb = self.pipe.text_encoder(prompt_inputs.input_ids)[0] # (B, T, D)
        
        # 2. Baseline Generation (Alpha = 0)
        img_base = self.pipe(
            prompt_embeds=text_emb,
            num_inference_steps=1,
            guidance_scale=0.0, # SD-Turbo usually uses 0.0 guidance
        ).images[0]
        
        # 3. AFSPL-Steered Generation
        # Pg_emb: [K, D] or [1, K, D]
        if Pg_emb is not None:
            if Pg_emb.dim() == 2:
                Pg_emb = Pg_emb.unsqueeze(0) # [1, K, D]
            
            # Precison Refinement: Normalize + Pool
            Pg_vec = Pg_emb.to(device).to(dtype).mean(dim=1, keepdim=True) # [1, 1, D]
            Pg_vec = F.normalize(Pg_vec, dim=-1)
            
            # Gated scaling (alpha capped at 0.5 influence)
            Pg_scale = float(alpha) * 0.5
            steered_emb = text_emb + Pg_scale * Pg_vec
            
            img_steer = self.pipe(
                prompt_embeds=steered_emb,
                num_inference_steps=1,
                guidance_scale=0.0,
            ).images[0]
        else:
            img_steer = img_base
            
        return img_base, img_steer

    def pil_to_base64(self, img: PIL.Image.Image) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
