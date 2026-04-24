from __future__ import annotations
import torch
import torch.nn as nn
from loguru import logger
import traceback
from typing import Dict, Optional, Tuple, Any

# Centralized Contracts (Issue R2)
from utils.contracts import enforce_generation_contract, validate_captions
from .clip_encoder import CLIPVisionEncoder
from .text_decoder import FrozenTextDecoder
from .prompt_injection import PromptInjection


class MultimodalProjector(nn.Module):
    """
    Upgraded Phase 2.6 Projector.
    Projects 1 CLIP feature into 4 T5 tokens with internal normalization.
    """
    def __init__(self, clip_dim: int = 512, decoder_dim: int = 512, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.image_expand = nn.Linear(clip_dim, num_tokens * decoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, D_clip -> B, 4 * D_t5
        z = self.image_expand(x)
        z = z.view(z.size(0), self.num_tokens, -1) # [B, 4, D_t5]
        return z


class AFSPLModel(nn.Module):
    """
    Full AFSPL multimodal pipeline - Phase 2.6 Hardened.
    Frozen: CLIP encoder, T5 decoder
    Trainable: soft prompts, fusion module, 4-token projector
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        decoder_model_name: str = "google/flan-t5-small",
        prompt_length: int = 16,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.prompt_length = prompt_length

        self.vision_encoder = CLIPVisionEncoder(clip_model_name, device)
        self.text_decoder = FrozenTextDecoder(decoder_model_name, device)
        self.hidden_dim = self.text_decoder.hidden_dim
        
        # Stage 4 Robustness (Phase 3.2): Numerical Shell
        self.prompt_norm = nn.LayerNorm(self.hidden_dim)

        self.projector = MultimodalProjector(
            clip_dim=self.vision_encoder.hidden_dim,
            decoder_dim=self.hidden_dim,
            num_tokens=4,
        )
        # Phase 2.6 refinement: Learned Vision Positioning
        self.vision_pos = nn.Parameter(torch.randn(1, 4, self.hidden_dim) * 0.02)
        
        # Phase 2.6 refinement: Learnable Scales to maintain signal after normalization
        # Phase 5: Reverted to 1.0 to prevent vocabulary distortion while keeping 5.0 clamp
        self.prompt_scale = nn.Parameter(torch.ones(1))
        self.vision_scale = nn.Parameter(torch.ones(1))
        
        # Phase 3: Adaptive Fusion Bridge
        from prompts.fusion import AdaptiveFusionModule
        self.fusion = AdaptiveFusionModule(
            strategy="dynamic",
            image_dim=self.vision_encoder.hidden_dim,
            text_dim=self.hidden_dim
        )

        self.prompt_injector = PromptInjection(prompt_length, self.hidden_dim)

        # Phase 2.6: Ablation handles for research verification
        self.ablation_config = {
            "use_prefix": True,
            "use_prompts": True,
            "use_image": True
        }

    def apply_selective_unfreeze(self):
        """Phase 3 Proxy: Activate grounding bridge."""
        self.text_decoder.apply_selective_unfreeze()

    def get_trainable_params(self) -> list[nn.Parameter]:
        """
        AFSPL Phase 3 Passthrough.
        Automatically collects all parameters marked with requires_grad=True.
        Includes Projector, Scales, Positional Embeddings, and unfrozen Decoder layers.
        """
        trainable = [p for p in self.parameters() if p.requires_grad]
        
        # Also check for external fusion if passed separately
        if hasattr(self, "fusion"):
            trainable += [p for p in self.fusion.parameters() if p.requires_grad]
            
        return trainable

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder.encode_image(pixel_values)

    def encode_text_clip(self, input_ids, attention_mask) -> torch.Tensor:
        return self.vision_encoder.encode_text(input_ids, attention_mask)

    def get_shared_representation(
        self,
        pixel_values: torch.Tensor,
        fused_prompt: torch.Tensor,
        prefix_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        AFSPL SHARED MULTIMODAL SPACE (E_i + P, E_t + P)
        -----------------------------------------------
        Constructs the unified latent representation from Image and Prompts.
        This is the hook point for future Generative Decoders (e.g. Diffusion).
        """
        B = pixel_values.size(0)
        device = pixel_values.device

        # 1. Image Tokens: E_i
        image_embeds = self.encode_image(pixel_values) # (B, D_clip)
        img_tok = self.projector(image_embeds)         # (B, 4, H)
        img_tok = img_tok + self.vision_pos.to(device)

        # 2. Prompt Tokens: P (fused)
        if fused_prompt is None:
            # Fallback zero prompts to prevent dimension crashes
            fused_P = torch.zeros((B, self.prompt_length, self.hidden_dim), device=device)
        else:
            fused_P = fused_prompt if fused_prompt.dim() == 3 else fused_prompt.unsqueeze(0).expand(B, -1, -1)
        
        # Stage 4 Robustness (Phase 2.2): Numerical Shell
        # Applied BEFORE scaling to provide a stable distribution for normalization.
        fused_P = self.prompt_norm(fused_P)
        
        # 3. Symmetric Normalization & Scaling (Diagram: Latent Alignment)
        fused_P = torch.nn.functional.normalize(fused_P, dim=-1) * self.prompt_scale
        img_tok = torch.nn.functional.normalize(img_tok, dim=-1) * self.vision_scale

        parts = []
        # 4. Optional Prefix Text: E_t (Context Input from diagram)
        if prefix_ids is not None:
            prefix_emb = self.text_decoder.model.shared(prefix_ids)
            # Apply symmetric normalization to context as well
            prefix_emb = torch.nn.functional.normalize(prefix_emb, dim=-1)
            parts.append(prefix_emb)
        
        parts.append(fused_P)
        parts.append(img_tok)

        # Concatenate into the final shared multimodal context
        shared_space = torch.cat(parts, dim=1) # (B, T_enc, H)
        return shared_space

    def forward(
        self,
        pixel_values: torch.Tensor,
        caption_input_ids: torch.Tensor,
        caption_attention_mask: torch.Tensor,
        fused_prompt: torch.Tensor,            # (B, P, H) or (P, H)
        labels: torch.Tensor | None = None,
        prefix_ids: torch.Tensor | None = None,
    ):
        """
        Hardened Forward Pass using the Shared Multimodal Space.
        """
        # --- SHARED MULTIMODAL SPACE ---
        encoder_embeds = self.get_shared_representation(
            pixel_values, fused_prompt, prefix_ids
        )
        
        # 5. Decoder Pass (Language Model block in diagram)
        loss, logits = self.text_decoder.forward_with_prefix(
            prefix_embeds=encoder_embeds,
            input_ids=caption_input_ids,
            attention_mask=caption_attention_mask,
            labels=labels,
        )
        return loss, logits

    @property
    def tokenizer(self):
        return self.text_decoder.tokenizer

    @property
    def clip_processor(self):
        return self.vision_encoder.processor

    def compute_clip_similarity(self, image_input: torch.Tensor | PIL.Image.Image, caption: str) -> torch.Tensor:
        """
        Calculates the semantic alignment between an image (Tensor or PIL) and a caption.
        """
        device = next(self.parameters()).device
        
        # 1. Process Image
        if isinstance(image_input, PIL.Image.Image):
            pixel_values = self.clip_processor(images=image_input, return_tensors="pt").pixel_values.to(device)
        else:
            pixel_values = image_input.to(device)

        # 2. Encode image
        img_emb = self.encode_image(pixel_values) # (B, D)
        
        # 3. Encode text
        tokens = self.vision_encoder.tokenizer(
            caption, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        txt_emb = self.vision_encoder.encode_text(tokens.input_ids, tokens.attention_mask)
        
        # 4. Cosine Similarity
        img_norm = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_norm = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        return (img_norm * txt_norm).sum(dim=-1).mean()

    @torch.no_grad()
    def generate_from_pixels(
        self,
        pixel_values: torch.Tensor,
        Pg: torch.Tensor | None = None,
        Pl: torch.Tensor | None = None,
        Pm: torch.Tensor | None = None,
        max_length: int = 30,
        num_beams: int = 1,
        fast: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        AFSPL Hardened Generation Wrapper (Issues 2, 3, R1, R3).
        Dual-contract return: {sequences, captions}.
        """
        batch_size = pixel_values.size(0)
        device = pixel_values.device
        
        try:
            # 1. Pipeline preparation: Resolve Fused Prompt
            if kwargs.get("fused_prompt") is not None:
                f_P = kwargs.get("fused_prompt")
            else:
                E_image = self.encode_image(pixel_values)
                E_text = None # Default context-less generation
                f_scale = kwargs.get("scale", 1.0)
                f_P, _ = self.fusion(Pg=Pg, Pl=Pl, Pm=Pm, E_image=E_image, E_text=E_text, scale=f_scale)

            # Stage 4 Diagnostic (Priority 3): Log Prompt Distribution
            # This helps identify "dangerous" aggregation regions after federated update.
            logger.info(f"[PROMPT STATS] shape={f_P.shape}, mean={f_P.mean():.4f}, std={f_P.std():.4f}, "
                        f"min={f_P.min():.4f}, max={f_P.max():.4f}, device={f_P.device}")

            # 2. Get shared representation
            inputs_embeds = self.get_shared_representation(
                pixel_values=pixel_values,
                fused_prompt=f_P,
                prefix_ids=kwargs.get("prefix_ids")
            )
            
            # 3. Generate path selection (Ref R3)
            # Hardened: Use explicit tuple and int casting for shape to prevent type-errors
            seq_len = int(inputs_embeds.size(1))
            attention_mask = torch.ones((int(batch_size), seq_len), device=device, dtype=torch.long)
            
            if fast:
                sequences = self.text_decoder.generate_fast(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_length,
                    num_beams=num_beams
                )
                gen_out = {"sequences": sequences}
            else:
                gen_out = self.text_decoder.generate_with_diagnostics(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_length,
                    num_beams=num_beams,
                    **kwargs
                )
            
            # 4. Centralized Contract Enforcement (Ref R2, Priority 4)
            raw_sequences = gen_out.get("sequences")
            if raw_sequences is None:
                raise RuntimeError("Decoder returned None before contract enforcement")

            sequences = enforce_generation_contract(
                raw_sequences, 
                batch_size, 
                self.tokenizer.pad_token_id, 
                device
            )
            
            # Stage 4 Resilient Beam Reshape (Priority 5)
            # Safely handle (B*beams, L) vs (B, L) mismatches
            if num_beams > 1 and sequences.size(0) == batch_size * num_beams:
                sequences = sequences.view(batch_size, num_beams, -1)[:, 0, :]
            elif sequences.size(0) != batch_size:
                logger.warning(f"[BEAM] Shape mismatch {sequences.shape}, truncating to {batch_size}")
                sequences = sequences[:batch_size]

            raw_captions = gen_out.get("captions")
            if not raw_captions and sequences is not None:
                raw_captions = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            
            captions = validate_captions(raw_captions, batch_size)

            return {
                "sequences": sequences,
                "captions": captions,
                "confidence": gen_out.get("confidence", 0.0),
                "entropy": gen_out.get("entropy", 0.0),
                "attribution": gen_out.get("attribution", {})
            }

        except Exception as e:
            # Phase 2.5: Restore Safe Fallback (after debugging)
            import traceback
            logger.error(f"[GEN FAIL] {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            
            # Centralized Contract Fallback (Ref R2)
            fallback_ids = enforce_generation_contract(None, batch_size, self.tokenizer.pad_token_id, device)
            fallback_captions = validate_captions(None, batch_size)
            
            return {
                "sequences": fallback_ids,
                "captions": fallback_captions,
                "confidence": 0.0,
                "entropy": 0.0,
                "attribution": {}
            }
