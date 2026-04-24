"""Frozen T5/Flan-T5 text decoder with prompt-conditioned generation."""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
from loguru import logger

# Phase 3.5: Unified Generation Safety Config
AFSPL_GEN_CONFIG = {
    "max_new_tokens": 30,
    "min_new_tokens": 8,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.7,
    "repetition_penalty": 2.0, # Phase 5: Ultra-Hardened to kill "hospice" loops
    "no_repeat_ngram_size": 3,
    "early_stopping": False,
}
class FrozenTextDecoder(nn.Module):
    """Flan-T5-small decoder kept frozen. Only accepts prefix-injected inputs."""

    def apply_selective_unfreeze(self):
        """
        Phase 3 Architecture: Activate the vision-language bridge.
        Unfreezes cross-attention and layer norms to enable visual grounding.
        """
        # Freeze everything first
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # Unfreeze ONLY cross-attention and layer norms
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            if "EncDecAttention" in name or "layer_norm" in name:
                param.requires_grad = True
                unfrozen_count += 1
        
        # Keep text encoder (decoder part of T5 architecture) in train mode for these layers
        self.model.train() 
        logger.info(f"Text decoder: Phase 3 Bridge ACTIVE ({unfrozen_count} tensors).")

    def __init__(self, model_name: str = "google/flan-t5-small", device: str = "cpu"):
        super().__init__()
        self.device = device
        logger.info(f"Loading text decoder: {model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_dim = self.model.config.d_model  # 512 for flan-t5-small

        # Initialize in Selective Unfreeze state (Phase 3 Default)
        self.apply_selective_unfreeze()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.shared  # T5 shared embedding layer

    def tokenize(self, texts: list[str], max_length: int = 64) -> dict:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def forward_with_prefix(
        self,
        prefix_embeds: torch.Tensor,    # (B, prompt_len, hidden)
        input_ids: torch.Tensor,         # (B, seq_len)
        attention_mask: torch.Tensor,    # (B, seq_len)
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Prefix tuning: prepend prompt embeddings before decoder inputs.
        Returns: loss (if labels provided) or logits
        """
        # Get token embeddings for input
        token_embeds = self.model.shared(input_ids)  # (B, seq_len, hidden)

        # Concatenate prompt prefix + token embeddings
        combined_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)  # (B, P+S, H)

        # Extend attention mask for prefix
        B, P, _ = prefix_embeds.shape
        prefix_mask = torch.ones(B, P, device=attention_mask.device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # (B, P+S)

        # Encoder: use image embedding cross-attention placeholder (T5 encoder)
        # For T5, we pass inputs_embeds to encoder
        encoder_outputs = self.model.encoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
        )

        if labels is not None:
            # Shift labels for teacher forcing
            decoder_input_ids = self.model._shift_right(labels)
            outputs = self.model(
                encoder_outputs=encoder_outputs,
                attention_mask=combined_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
            )
            return outputs.loss, outputs.logits
        else:
            return encoder_outputs, combined_mask

    @torch.no_grad()
    def generate_fast(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 30,
        **kwargs
    ) -> torch.Tensor:
        """
        AFSPL Optimized Training Sampling (Ref R3).
        Minimal overhead for scheduled sampling. Returns ONLY sequences.
        """
        encoder_outputs = self.model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        # Stage 4 Diagnostic (Priority 7): Empty Sequence Guard
        sequences = self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **{**AFSPL_GEN_CONFIG, "max_new_tokens": max_new_tokens},
            num_beams=kwargs.get("num_beams", 1),
            return_dict_in_generate=False,
            output_scores=False,
            output_attentions=False,
        )
        
        if sequences.numel() == 0 or sequences.size(1) == 0:
            logger.warning("[GEN] Empty sequence generated in fast path, using fallback.")
            pad_id = self.tokenizer.pad_token_id
            sequences = torch.full((int(inputs_embeds.size(0)), 1), pad_id, device=sequences.device, dtype=torch.long)
            
        return sequences

    @torch.no_grad()
    def generate_caption(
        self,
        prefix_embeds: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        max_new_tokens: int = 50,
    ) -> list[str]:
        """Generate captions given prompt prefix embeddings."""
        B, P, H = prefix_embeds.shape
        
        # If input_ids are provided (Context Prefix), encode them
        if input_ids is not None:
            token_embeds = self.model.shared(input_ids)
            combined_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        else:
            combined_embeds = prefix_embeds

        B, T, _ = combined_embeds.shape
        combined_mask = torch.ones(B, T, device=prefix_embeds.device, dtype=torch.long)

        encoder_outputs = self.model.encoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
        )
        encoder_outputs = self.model.encoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
        )
        generated = self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=combined_mask,
            **AFSPL_GEN_CONFIG
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    @torch.no_grad()
    def generate_with_diagnostics(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        zones: dict[str, tuple[int, int]] | None = None,
        max_new_tokens: int = 50,
        **kwargs,
    ) -> dict:
        """
        AFSPL Explainability Engine - Hardened (Issues 1, 6, 7, 9, 10, 11).
        Generates caption + attribution + batch-aggregated diagnostics.
        """
        B = inputs_embeds.size(0)
        # Stage 4 Diagnostic (Priority 2): Temporary Bypass
        # Force off to test if diagnostics loops are the intermittent crash trigger.
        compute_diagnostics = False 
        
        # 1. Encode
        encoder_outputs = self.model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        
        # 2. Generate (Conditional scores/attentions - Issue 7)
        outputs = self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **{**AFSPL_GEN_CONFIG, "max_new_tokens": max_new_tokens},
            num_beams=kwargs.get("num_beams", 1),
            return_dict_in_generate=True,
            output_scores=compute_diagnostics,
            output_attentions=compute_diagnostics,
        )
        
        # Stage 4 Diagnostic (Phase 1.2): Safe 2D Enforcement (reshape)
        sequences = outputs.sequences
        if sequences is None:
            raise RuntimeError("Decoder returned None sequences")

        if sequences.dim() != 2:
            try:
                # Phase 1.2: use reshape, not view
                sequences = sequences.reshape(sequences.size(0), -1)
            except Exception as e:
                raise RuntimeError(f"Invalid sequence shape: {sequences.shape}") from e
        
        # Phase 1.3: batch-size logging
        batch_size = inputs_embeds.size(0)
        logger.debug(f"[GEN DEBUG] sequences shape={sequences.shape}, expected_batch={batch_size}")
        
        # Phase 1.4: empty sequence protection
        if sequences.size(1) == 0:
            raise RuntimeError("Empty sequence generated")

        outputs.sequences = sequences
        
        # 3. Decode Captions
        captions = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        
        # 4. Batch-Safe Diagnostics (Issue 6, 9, 10, 11)
        all_confidences = []
        all_entropies = []
        len_tokens = 0
        
        if compute_diagnostics and hasattr(outputs, "scores"):
            probs_list = [torch.softmax(s, dim=-1) for s in outputs.scores]
            
            for b in range(B):
                sample_probs = []
                sample_entropies = []
                seq_len = outputs.sequences.size(1)
                
                # Boundary Safety (Issue 1)
                max_steps = min(len(probs_list), seq_len - 1)
                
                for i in range(max_steps):
                    token_id = outputs.sequences[b, i+1]
                    p = probs_list[i][b, token_id]
                    sample_probs.append(p)
                    # Correct Sample-Specific Entropy (Issue 11)
                    ent = -(probs_list[i][b] * torch.log(probs_list[i][b] + 1e-6)).sum()
                    sample_entropies.append(ent)

                if sample_probs:
                    # Device Consistency via torch.stack (Issue 9)
                    probs_tensor = torch.stack(sample_probs)
                    log_probs = torch.log(torch.clamp(probs_tensor, min=1e-6))
                    conf = torch.exp(log_probs.mean()).item()
                    
                    # Length Penalty
                    l_pen = min(1.0, len(sample_probs) / 8.0)
                    all_confidences.append(conf * l_pen)
                    all_entropies.append(torch.stack(sample_entropies).mean().item())
                    len_tokens = max(len_tokens, len(sample_probs))
            
        confidence = float(np.mean(all_confidences)) if all_confidences else 0.0
        mean_entropy = float(np.mean(all_entropies)) if all_entropies else 0.0

        # 5. attribution (Explainability - Conditioned)
        attribution_shares = {"prefix": 0.33, "prompts": 0.33, "vision": 0.34}
        
        if compute_diagnostics and getattr(outputs, "cross_attentions", None):
            all_step_attns = []
            for step_attn in outputs.cross_attentions:
                # Average over layers/heads for batch sample 0 (standard for Heatmaps)
                layer_attns = torch.stack(step_attn) # (L, B, H, T_out, T_in)
                mean_layer_attn = layer_attns.mean(dim=(0, 2)) # (B, T_out, T_in)
                all_step_attns.append(mean_layer_attn)
            
            A = torch.cat(all_step_attns, dim=1)[0] # (T_gen, T_in)
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
            A_mean = A.mean(dim=0)
            
            attr = {}
            for name, (start, end) in zones.items():
                attr[name] = float(A_mean[start:end].sum()) if start < end else 0.0
            
            total_attr = sum(attr.values()) + 1e-6
            attribution_shares = {k: v/total_attr for k, v in attr.items()}

        return {
            "sequences": outputs.sequences,
            "captions": captions,
            "confidence": round(confidence, 4),
            "entropy": round(mean_entropy, 4),
            "attribution": attribution_shares,
            "token_count": len_tokens
        }
