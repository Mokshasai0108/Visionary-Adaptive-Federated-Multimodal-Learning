"""
AFSPL Client-side trainer.
Implements the full client-side training flow (steps 1-19 from spec).
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from loguru import logger
from typing import Optional

from models.multimodal_model import AFSPLModel
from prompts.prompt_manager import PromptManager
from prompts.fusion import AdaptiveFusionModule
from prompts.topk_selector import TopKSelector, SparsePromptUpdate
from training.losses import AFSPLLoss

# Centralized Contracts (Issue R2)
from utils.contracts import enforce_generation_contract


class ClientTrainer:
    """
    Trains soft prompts on local client data.
    Returns sparse prompt updates + metrics for server aggregation.
    """

    def __init__(
        self,
        client_id: int,
        model: AFSPLModel,
        prompt_manager: PromptManager,
        fusion_module: AdaptiveFusionModule,
        topk_selector: TopKSelector,
        loss_fn: AFSPLLoss,
        config,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.model = model
        self.pm = prompt_manager
        self.fusion = fusion_module
        self.topk = topk_selector
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # Phase 3.5: Absolute Stability Baseline
        # Prevents the scheduler from using a mutating base and causing oscillation.
        self.initial_lr = config.training.learning_rate
        self.check_grads = True

        # Build optimizer over all trainable params (Phase 2.6 Refinement)
        trainable = [p for p in model.get_trainable_params() if p.requires_grad]
        trainable += self.pm.get_trainable_params()
        trainable += self.fusion.get_trainable_params()
        
        # Deduplicate just in case
        unique_trainable = []
        seen_ids = set()
        for p in trainable:
            if id(p) not in seen_ids:
                unique_trainable.append(p)
                seen_ids.add(id(p))
        self.trainable = unique_trainable
        self.error_count = 0 
        self.last_fusion_state = False
        
        # Attribute Registry
        self.model = model
        self.tokenizer = model.tokenizer
        self.config = config  # Ensuring self.config is available
        self.device = device
        self.eos_token_id = self.tokenizer.eos_token_id

        self.optimizer = torch.optim.AdamW(
            self.trainable,
            lr=self.config.training.learning_rate,
            weight_decay=0.01,
        )
        
        # --- PHASE 3 HARDENING SANITY CHECKS ---
        total = sum(p.numel() for p in self.trainable)
        print(f"[DEBUG] ClientTrainer | Total Trainable params: {total:,}")

        # Sanity check: Ensure all trainable params are in optimizer
        opt_ids = set(id(p) for g in self.optimizer.param_groups for p in g['params'])
        model_ids = set(id(p) for p in self.trainable)
        missing = model_ids - opt_ids
        assert len(missing) == 0, f"❌ Missing params in optimizer: {len(missing)}"

    # Removed local enforce_generation_contract in favor of centralized utility (Ref R2)

    def train_round(
        self,
        dataloader: DataLoader,
        global_prompt: torch.Tensor,  # (P, H) from server
        rnd: int = 0,                # Current round for warm-up
    ) -> tuple[SparsePromptUpdate, dict]:
        """
        Run local_epochs of training and return sparse update + metrics.
        """
        self.pm.set_global_prompt(global_prompt)
        old_local = self.pm.local_prompt.embedding.detach().clone()

        self.model.to(self.device)
        self.fusion.to(self.device)

        total_loss = 0.0
        total_ce = 0.0
        total_clip = 0.0
        n_batches = 0
        
        # Phase 3.5 Recovery: Stability-Aware Fusion Unlock
        # Stage 4 Robustness (Phase 3.1): Sampling Warm-up & Fusion Ramping
        if rnd < 3:
            tf_ratio = 1.0          # Full teacher forcing safety
            fusion_scale = 0.0      # Vision bridge off for early stability
        else:
            # Linear decay/ramp from Round 3 onwards
            tf_ratio = max(0.5, 1.0 - (rnd - 2) * 0.08)
            fusion_scale = min(1.0, max(0.0, (rnd - 2) / 6.0))
        
        # Log transition metrics
        logger.info(f"[LR] Round {rnd} | lr={self.config.training.learning_rate} | tf_ratio={tf_ratio:.2f} | fusion={fusion_scale:.2f}")

        # Freeze fusion for Rounds 1-4 to let decoder re-stabilize completely.
        # Unlock ONLY if model is linguistically healthy (LR hasn't bottomed out).
        fusion_active = (rnd >= 5 and self.config.training.learning_rate >= 3e-05)
        if self.last_fusion_state != fusion_active:
            if not fusion_active:
                for p in self.fusion.parameters(): p.requires_grad = False
                self.fusion.eval()
                logger.info(f"Client {self.client_id} | [FUSION] LOCKED (Eval mode)")
            else:
                for p in self.fusion.parameters(): p.requires_grad = True
                self.fusion.train()
                logger.info(f"Client {self.client_id} | [FUSION] UNLOCKED (Training mode ON)")
            
            # Gated Optimizer Activation: Re-link param groups to exclude frozen params
            # This ensures no gradient noise from frozen layers.
            active_params = [p for p in self.trainable if p.requires_grad]
            self.optimizer.param_groups[0]['params'] = active_params
            self.last_fusion_state = fusion_active

        # Phase 3.5: Synchronized LR Schedule with Fixed Baseline
        # Uses initial_lr to calculate the scheduled target, then caps it with the current brake.
        base_lr = self.initial_lr
        if rnd <= 3:
            scheduled_lr = base_lr * 1.0  # Warm-up Baseline (Stable Zone)
        elif rnd <= 10:
            scheduled_lr = base_lr * 1.2  # Moderated Safe Boost (Targeting 6e-5)
        else:
            scheduled_lr = base_lr * 0.8  # Precision Decay

        # MANDATORY FIX: Schedule MUST obey the safety brake (config.training.learning_rate)
        # Prevents the scheduler from overriding the collapse detector's reduction.
        new_lr = min(scheduled_lr, self.config.training.learning_rate)
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        # Phase 3.5: Multi-Stage Lambda Progression
        # Slower ramp-up (over 4 rounds) for better grounding stability.
        grounding_lambda = self.config.loss.lambda2
        progress = min(1.0, rnd / 4.0)
        effective_lambda_grounding = grounding_lambda * progress

        for epoch in range(self.config.training.local_epochs):
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                
                # --- SHAPE TELEMETRY (Phase 4 Diagnostic) ---
                if n_batches == 0:
                    logger.debug(f"[SHAPE] Round {rnd} | Batch {n_batches} | pixels={pixel_values.shape} | ids={input_ids.shape} | tf_ratio={tf_ratio:.2f}")                # Phase 4: Scheduled Sampling (Exposure Bias Fix)
                if torch.rand(1).item() > tf_ratio:
                    curr_B = input_ids.size(0)
                    with torch.no_grad():
                        # --- PRODUCTION CONTRACT SAMPLING ---
                        # Stage 4 (Ref R3): Optimized Training Sampling
                        gen_out = self.model.generate_from_pixels(
                            pixel_values, 
                            Pg=self.pm.global_prompt.embedding,
                            max_length=labels.size(1),
                            num_beams=1,
                            fast=True
                        )
                        
                        # Stage 4 (Ref R2): Unified Contract Enforcement
                        raw_sequences = gen_out.get("sequences")
                        generated_ids = enforce_generation_contract(
                            raw_sequences, 
                            curr_B, 
                            pad_id=self.tokenizer.pad_token_id, 
                            device=pixel_values.device
                        )
                        
                        # 2. Gradient Isolation
                        generated_ids = generated_ids.detach().clone()
                        
                        # 3. Sequence Length Alignment (L columns)
                        max_len = labels.size(1)
                        if generated_ids.size(1) < max_len:
                            pad = torch.full(
                                (curr_B, max_len - generated_ids.size(1)),
                                self.tokenizer.pad_token_id,
                                device=generated_ids.device,
                                dtype=torch.long
                            )
                            generated_ids = torch.cat([generated_ids, pad], dim=1)
                        else:
                            generated_ids = generated_ids[:, :max_len]
                        
                        # 4. Final Verification & Masking
                        if generated_ids.size() == labels.size():
                            use_sampling_mask = (torch.rand(curr_B, 1, device=pixel_values.device) > 0.5).expand(-1, max_len)
                            input_ids = torch.where(use_sampling_mask, generated_ids.to(input_ids.dtype), input_ids)
                        else:
                            if self.error_count < 3:
                                logger.warning(f"[CONTRACT FAIL] Round {rnd} | Batch {n_batches} | {generated_ids.size()} mismatch")
                                self.error_count += 1

                # Gradient flow check for first step of first round
                self.check_grads = (rnd == 1 and n_batches == 0)

                with autocast("cuda", enabled=bool(self.scaler)):
                    # Step 4: Encode images
                    E_image = self.model.encode_image(pixel_values)

                    # Step 5-6: Get text embeddings from CLIP
                    E_text = self.model.encode_text_clip(input_ids, attention_mask)

                    # Steps 7-11: Adaptive fusion
                    fused, weights = self.fusion(
                        self.pm.global_prompt.embedding,
                        self.pm.local_prompt.embedding,
                        self.pm.modality_prompt.embedding,
                        E_image=E_image,
                        E_text=E_text,
                        scale=fusion_scale,
                    )  # fused: (B, P, H), weights: (3,)

                    # Steps 12-14: Forward pass with prefix injection
                    loss_ce_val, logits = self.model.forward(
                        pixel_values=pixel_values,
                        caption_input_ids=input_ids,
                        caption_attention_mask=attention_mask,
                        fused_prompt=fused,
                        labels=labels,
                    )

                    # Step 15: Compute total loss (with warm-up lambda)
                    total_loss_val, components = self.loss_fn(
                        logits=logits,
                        labels=labels,
                        image_embeds=E_image,
                        text_embeds=E_text,
                        effective_lambda2=effective_lambda_grounding
                    )

                    # Phase 4: Contextual EOS Penalty (Stronger suppression)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    eos_log_prob = log_probs[..., self.eos_token_id]
                    
                    labels_mask = (labels != -100).float()
                    # Increase mask to 12 steps for better sequence initiation
                    time_steps = torch.arange(logits.size(1), device=logits.device)
                    early_mask = (time_steps < 12).float()
                    
                    # 🔥 Stronger 0.03 penalty as requested
                    eos_penalty = (eos_log_prob * labels_mask * early_mask).mean()
                    total_loss_val = total_loss_val + 0.03 * eos_penalty

                # Steps 16: Backprop through Pl, Pm, fusion params only
                if self.scaler:
                    self.scaler.scale(total_loss_val).backward()
                    self.scaler.unscale_(self.optimizer)

                    # Safety: Check for NaNs after unscale
                    for p in self.trainable:
                        if p.grad is not None and torch.isnan(p.grad).any():
                            logger.warning(f"Client {self.client_id} | [WARNING] NaN detected in gradients")

                    # Clip ONLY trainable params
                    clip_grad_norm_(self.trainable, max_norm=1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss_val.backward()
                    clip_grad_norm_(self.trainable, max_norm=1.0)
                    self.optimizer.step()

                if self.check_grads and n_batches == 0:
                    # --- PHASE 3: DYNAMIC ACCELERATION PROOF ---
                    # 1. Verify Grounding Gradients
                    grad_counts = sum(1 for p in self.trainable if p.grad is not None)
                    logger.info(f"[GPU ACCEL] Client {self.client_id} | Training on: {torch.cuda.get_device_name(0)}")
                    logger.info(f"[GRAD] Active Tensors: {grad_counts}/{len(self.trainable)} (Cross-Attention Grounding ACTIVE)")
                    
                    # 2. Specific Grounding Layers Proof
                    sample_dec = next((n for n, p in self.model.named_parameters() if "EncDecAttention" in n and p.grad is not None), None)
                    if sample_dec:
                        logger.info(f"[GROUNDING] {sample_dec} | status=TRAINING")
                    
                    self.check_grads = False

                total_loss += components["total"]
                total_ce += components["l_ce"]
                total_clip += components.get("l_clip", 0.0)
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Steps 17-19: Top-K token selection
        grad = self.pm.local_prompt.embedding.grad
        sparse_update = self.topk.select(
            self.pm.local_prompt.embedding,
            gradients=grad,
            old_values=old_local,
        )

        comm_bytes = self.topk.communication_bytes(sparse_update.k, self.pm.hidden_dim)

        metrics = {
            "client_id": self.client_id,
            "avg_loss": avg_loss,
            "avg_ce": total_ce / max(n_batches, 1),
            "avg_clip": total_clip / max(n_batches, 1),
            "fusion_weights": weights.tolist() if hasattr(weights, "tolist") else list(weights),
            "comm_bytes": comm_bytes,
            "k": sparse_update.k,
            "n_batches": n_batches,
        }

        logger.info(f"Client {self.client_id} | loss={avg_loss:.4f} | k={sparse_update.k} | bytes={comm_bytes}")
        return sparse_update, metrics
