"""
Training service: orchestrates simulated federated training loop.
Runs in a background thread. Exposes state for API polling.
"""
from __future__ import annotations
import time
import threading
import traceback
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from pathlib import Path
import tempfile
import os
from torch.utils.data import DataLoader, RandomSampler
from training.dataset import CaptionDataset, dirichlet_partition
from training.losses import AFSPLLoss
from training.trainer import ClientTrainer
from federated.server import FederatedServer
from prompts.prompt_manager import PromptManager
from prompts.fusion import AdaptiveFusionModule
from prompts.topk_selector import TopKSelector, SparsePromptUpdate
from models.multimodal_model import AFSPLModel
from app.models.registry import ModelRegistry
from evaluation.evaluator import evaluate_batch
from app.services.runtime_locks import acquire_training, release_training

CKPT_DIR = Path("backend/checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CHECKPOINT_PREFIX = "default"

def save_checkpoint_atomic(data: dict, path: Path) -> None:
    """Write checkpoint safely using a temporary file and atomic replace."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.save(data, tmp.name)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)

def save_checkpoint(rnd: int, model: AFSPLModel, Pg: torch.Tensor, fusion_weights, metrics: dict, config: AFSPLConfig, state, model_name: str = MODEL_CHECKPOINT_PREFIX) -> None:
    ckpt_dir = CKPT_DIR / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "model_state": model.state_dict(),
        "Pg": Pg.detach().cpu(),
        "fusion_weights": fusion_weights,
        "metrics": metrics,
        "round": rnd,
        "config": config.dict() if hasattr(config, "dict") else config,
        "version": "afsple_v3", # Phase 3: Cross-Attention Grounding
        "model_name": model_name,
    }
    
    # --- PHASE 3 HARDENING: SAVE-TIME ASSERTION ---
    # Ensure cross-attention keys are actually present in the state_dict
    has_cross_attn = any("encdecattention" in k.lower() for k in model.state_dict().keys())
    assert has_cross_attn, "❌ Model missing cross-attention layers before saving!"
    
    # round‑specific checkpoint
    save_checkpoint_atomic(data, ckpt_dir / f"ckpt_round_{rnd}.pt")
    # best‑metric checkpoint
    if metrics.get("cider", 0) > getattr(state, "best_metric", 0):
        state.best_metric = metrics["cider"]
        save_checkpoint_atomic(data, ckpt_dir / "ckpt_best.pt")

from app.core.config import AFSPLConfig
from app.core.reproducibility import set_global_seed
from models.multimodal_model import AFSPLModel
from prompts.prompt_manager import PromptManager
from prompts.fusion import AdaptiveFusionModule
from prompts.topk_selector import TopKSelector
from training.losses import AFSPLLoss
from training.dataset import (
    SyntheticDataset,
    dirichlet_partition,
    load_coco_json,
    load_flickr30k_json,
    CaptionDataset,
    get_client_dataloader,
)
from training.trainer import ClientTrainer
from federated.server import FederatedServer
from evaluation.comparisons import generate_comparison_table
from training.dataset_utils import sample_subset, split_train_val
from torch.utils.data import DataLoader, Subset


class TrainingState:
    """Shared mutable state for training loop."""
    def __init__(self):
        self.is_training = False
        self.current_round = 0
        self.total_rounds = 0
        self.status = "idle"
        self.best_metric = 0.0
        self.current_metrics: dict = {}
        self.total_comm_bytes = 0
        self.last_fusion_weights: list[float] = [0.33, 0.33, 0.34]
        self.warnings: list[str] = []
        self.error: str | None = None
        self.lock = threading.Lock()


_state = TrainingState()
_server: FederatedServer | None = None


def get_state() -> TrainingState:
    return _state


def get_server() -> FederatedServer | None:
    return _server


def start_training_background(request, config: AFSPLConfig) -> None:
    """Launch training in daemon thread."""
    t = threading.Thread(target=_run_training, args=(request, config), daemon=True)
    t.start()


def _run_training(request: TrainStartRequest, config: AFSPLConfig):
    # --- WINDOWS HARDENING ---
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass
        
    try:
        global _server
        state = get_state()

        with state.lock:
            if state.is_training:
                logger.warning("Training already running.")
                return
            state.is_training = True
            state.status = "initializing"
            state.current_round = 0
            state.total_rounds = request.num_rounds
            state.error = None

        set_global_seed(request.seed, config.project.deterministic)
        device = config.get_device()
        logger.info(f"Starting training on device={device}")

        # Override config from request
        config.training.max_rounds = request.num_rounds
        config.data.num_clients = request.num_clients
        config.fusion.strategy = request.fusion_strategy
        config.prompt.prompt_length = request.prompt_length
        config.training.batch_size = request.batch_size
        config.training.learning_rate = request.learning_rate
        config.loss.lambda1 = request.lambda1
        config.loss.lambda2 = request.lambda2
        config.topk.k_ratio = request.k_ratio

        # Build model (frozen)
        model = AFSPLModel(
            clip_model_name="openai/clip-vit-base-patch32",
            decoder_model_name="google/flan-t5-small",
            prompt_length=config.prompt.prompt_length,
            device=device,
        )
        model.to(device)

        # --- PHASE 3: ACTIVATE GROUNDING BRIDGE ---
        model.apply_selective_unfreeze()
        
        # Verification Summary
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"[CHECK] Phase 3 Bridge Active. Total Trainable: {trainable_count:,}")
        
        # Log what actually trains (one-time summary)
        for n, p in model.named_parameters():
            if p.requires_grad:
                logger.info(f"[TRAIN] {n} -> {p.numel():,} parameters")

        # 4. Build Server & Clients
        _server = FederatedServer(config, device)

        # 1. Load dataset
        
        # --- PHASE 3.5: GATED RECOVERY ROLLBACK ---
        # Triggered ONLY via explicit request to prevent accidental model overrides.
        if request.is_recovery:
            ckpt_path = Path(request.checkpoint_path) if request.checkpoint_path else (CKPT_DIR / MODEL_CHECKPOINT_PREFIX / "ckpt_round_2.pt")
            
            if not ckpt_path.exists():
                raise FileNotFoundError(f"❌ Strict Recovery Failure: {ckpt_path} not found! Aborting to prevent further damage.")
            
            # Hard load bypassing 'best' logic
            logger.info(f"[RECOVERY] Initiating explicit rollback via request to: {ckpt_path}")
            data = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(data["model_state"], strict=False)
            
            # Manually register the recovered state
            ModelRegistry.set(model, {
                "round": data.get("round", 0),
                "metrics": data.get("metrics", {}),
                "global_prompt": data.get("Pg", None),
                "fusion_weights": data.get("fusion_weights", {"alpha": 0.33, "beta": 0.33, "gamma": 0.34}),
                "model_name": data.get("model_name", "default"),
                "version": data.get("version", "unknown"),
            })
            
            restored_round = ModelRegistry.get_round(model)
            baseline_cider = ModelRegistry.get_metrics(model).get("cider", 0.0)
            logger.info(f"[RECOVERY] Phase 3.5 Stable State Restored: Round {restored_round}")
            # 2. Sanity Check for Decoder Recovery (Round 1 of restoration = rnd 3)
            # Use positional formatting to keep loguru from parsing braces in sample captions
            logger.info("Restored baseline: {:.4f}", baseline_cider)
            logger.info("[SANITY] Restoring from round 2 context...")
        all_samples: list[dict] = []
        if config.data.train_dataset.lower() == "ms_coco":
            coco_path = Path(config.data.data_dir) / "coco" / "captions_train2017.json"
            all_samples = load_coco_json(str(coco_path))
        elif config.data.train_dataset.lower() == "flickr30k":
            flickr_root = Path(config.data.data_dir) / "flickr30k"
            all_samples = load_flickr30k_json(
                str(flickr_root / "captions.json"),
                str(flickr_root / "flickr30k-images"),
            )
        
        if not all_samples:
            raise ValueError(f"No samples found for {config.data.train_dataset}")

        # --- 30% PROFILE LOGIC ---
        # 1. Randomized Subset
        samples = sample_subset(all_samples, ratio=request.subset_ratio, seed=request.seed)
        logger.info(f"[DATA] Using {int(request.subset_ratio*100)}% subset → {len(samples)} samples")

        # 2. Train/Val Split (90/10)
        train_samples, val_samples = split_train_val(samples, val_ratio=0.1)
        logger.info(f"[DATA] Train: {len(train_samples)}, Val: {len(val_samples)}")

        # 3. Evaluation Setup (Lightweight: 300 samples)
        eval_samples = val_samples[:300]
        logger.info(f"Evaluation restricted to {len(eval_samples)} samples for speed.")
        
        if len(eval_samples) == 0:
            eval_samples = train_samples[:50] # Fallback

        # 3. Evaluation Setup
        val_dataset = CaptionDataset(
            eval_samples,
            model.vision_encoder.processor,
            model.text_decoder.tokenizer,
            max_seq_len=config.model.max_seq_len,
            image_size=config.data.image_size,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0, # Scale to 0 for Windows Shared Memory Stability
            pin_memory=(device == "cuda"),
        )

        # 4. Partitioning
        partitions = dirichlet_partition(
            len(train_samples),
            config.data.num_clients,
            alpha=5.0, # Phase 3.5 Balanced Heterogeneity (Removes mega-clients)
            seed=request.seed,
        )

        # Build Trainers
        clients = []
        loss_fn = AFSPLLoss(
            config.loss.lambda1,
            config.loss.lambda2,
            config.loss.clip_temperature,
        )
        
        for cid in range(config.data.num_clients):
            pm = PromptManager(config.prompt.prompt_length, 512, device=device)
            fusion = AdaptiveFusionModule("dynamic", 512, 512).to(device)
            
            # --- PHASE 3.5: ONE-TIME BIAS INJECTION ---
            # Set the research-grade vision bias once at initialization. 
            # This allows the model to LEARN from this baseline rather than being reset every round.
            with torch.no_grad():
                target_bias = torch.log(torch.tensor([0.30, 0.30, 0.40]))
                for module in fusion.modules():
                    if isinstance(module, nn.Linear) and module.out_features == 3:
                        module.bias.copy_(target_bias)
            
            topk = TopKSelector(config.topk.k_ratio)
            client = ClientTrainer(cid, model, pm, fusion, topk, loss_fn, config, device)
            clients.append((client, partitions[cid]))

        # 5. Build Persistent DataLoaders (Permanent Kernel Equilibrium)
        client_loaders = []
        logger.info("Pre-building {} persistent dataloaders (alpha=5.0)...", config.data.num_clients)
        for cid in range(config.data.num_clients):
            client, partition = clients[cid]
            client_samples = [train_samples[i] for i in partition]
            dataset = CaptionDataset(
                client_samples, 
                model.vision_encoder.processor, 
                model.text_decoder.tokenizer, 
                config.model.max_seq_len
            )
            sampler = RandomSampler(dataset)
            dl = DataLoader(
                dataset, 
                batch_size=config.training.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=0, # Scale to 0 for Windows Stability
                pin_memory=(device == "cuda"),
            )
            client_loaders.append(dl)

        with state.lock:
            state.status = "running"

        alpha = 0.7 # Smoothing factor for UI

        # 5. Tokenization Sanity Check
        logger.info("--- Tokenization Sanity Check ---")
        random_indices = np.random.choice(len(train_samples), 3, replace=False)
        for idx in random_indices:
            sample = train_samples[idx]
            cap_text = sample["caption"]
            tokens = model.text_decoder.tokenizer(cap_text, padding=True, truncation=True, return_tensors="pt")
            decoded = model.text_decoder.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            # Use positional arguments with no keys to prevent loguru from parsing the caption for braces
            logger.info("Sample [{}] | Raw: [COMPLETED] | Decoded: {}", idx, decoded)
            print(f"   > Raw: {cap_text}")
        
        # 6. Federated Rounds
        for rnd in range(1, config.training.max_rounds + 1):
            if not state.is_training: break
            logger.info(f"=== Round {rnd}/{config.training.max_rounds} ===")
            
            # 🔒 Acquire Lock for GPU Training
            acquire_training()
            try:
                t0 = time.time()
                global_prompt = _server.get_global_prompt()

                client_updates, round_losses, client_sizes, client_weights = [], [], [], []
                
                # Phase 4: Synchronize Fusion Bridge (Unlock at Round 3)
                if rnd < 3:
                    model.fusion.locked = True
                else:
                    model.fusion.locked = False
                
                # Federated selection: Select 2 clients for Phase 4 gradient diversity
                n_select = 2
 
                indices = torch.randperm(config.data.num_clients)[:n_select].tolist()

                for idx in indices:
                    client, partition = clients[idx]
                    # Reuse Persistent Loader
                    client_dl = client_loaders[idx]
                    update, c_metrics = client.train_round(client_dl, global_prompt, rnd=rnd)
                    client_updates.append(update)
                    round_losses.append(c_metrics["avg_loss"])
                    client_sizes.append(len(partition))
                    # Capture real weights
                    client_weights.append(client.fusion.get_weights())

                # Aggregation & Cleanup
                agg_info = _server.aggregate_round(client_updates, client_sizes)
                
                # Phase 4 Robustness: Prevent CUDA fragmentation
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
                # Dynamic weight telemetry (Flattened for diagram alignment)
                curr_weights = {"alpha": 0.33, "beta": 0.33, "gamma": 0.34}
                if client_weights:
                    # Mean across selected clients
                    for k in curr_weights.keys():
                        curr_weights[k] = float(np.mean([cw[k] for cw in client_weights]))
                
                # Stability Guard (Diagram alignment refinement)
                if rnd % 3 == 0 and curr_weights["gamma"] > 0.75:
                    logger.warning(f"⚠️ Round {rnd} | Vision dominance (gamma={curr_weights['gamma']:.4f}) is high. Linguistic fluency might degrade.")

                _server.last_fusion_weights = curr_weights

                # 5. Real Evaluation (Phase 3.5: Calibrated Frequency for Speed)
                eval_metrics = {"bleu4": 0.0, "cider": 0.0, "avg_len": 0.0}
                if rnd == 1 or rnd % 5 == 0: # 5-Round Interval Pruning
                    curr_Pg = _server.get_global_prompt()
                    eval_metrics = evaluate_batch(model, val_loader, device, Pg=curr_Pg, Pl=curr_Pg, Pm=curr_Pg)
                    
                    # Stage 4 Robustness (Phase 3.3): Log samples cleanly
                    samples = eval_metrics.get("sample_list", [])
                    for i, cap in enumerate(samples):
                        logger.info(f"[SAMPLE {i}] {cap}")
                
                # 2. Dynamic Collapse Detector (Acting on NEXT round)
                if eval_metrics["avg_len"] < 2.0:
                    logger.warning(f"⚠️ [GENERATION FAILURE] Model produced empty captions in Round {rnd}! Forcing linguistic exploration.")
                
                # Stage 4 Robustness (Phase 3.4): Relaxed Collapse Threshold
                # Allows for shorter sequences during early fusion without triggering the LR brake.
                if eval_metrics["avg_len"] < 1.0:
                    logger.warning(f"[COLLAPSE WARNING] Evaluation unreliable (len={eval_metrics['avg_len']:.2f})")
                    old_lr = config.training.learning_rate
                    config.training.learning_rate = max(old_lr * 0.5, 3e-05)
                    logger.warning(f"⚠️ Collapse detected — reducing global LR for next round: {old_lr} -> {config.training.learning_rate}")
                
                # 🚀 Phase 2.6 Bridge: Push metrics + ablation state to Global Registry
                ModelRegistry.set(model, {
                    "round": rnd,
                    "global_prompt": _server.get_global_prompt(),
                    "fusion_weights": curr_weights,
                    "metrics": eval_metrics,
                    "ablation": getattr(model, "ablation_config", {}),
                    "model_name": getattr(request, "model_name", "default"),
                })

                save_checkpoint(
                    rnd,
                    model,
                    _server.get_global_prompt(),
                    curr_weights,
                    eval_metrics,
                    config,
                    state,
                    model_name=getattr(request, "model_name", "default"),
                )
            finally:
                # 🔓 Release Lock for Inference to use GPU
                release_training()

            # 6. Smooth Metrics for UI
            curr_loss = float(np.mean(round_losses)) if round_losses else 0.0
            prev_metrics = state.current_metrics or {}
            
            smoothed_loss = alpha * prev_metrics.get("avg_loss", curr_loss) + (1-alpha) * curr_loss
            smoothed_bleu = alpha * prev_metrics.get("bleu4", eval_metrics["bleu4"]) + (1-alpha) * eval_metrics["bleu4"]

            # Communication Efficiency (Diagram alignment)
            # Optimized Calculation (Research Grade): Precision trainable param count
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            dense_bytes = trainable_params * 4 * n_select 
            comm_bytes = agg_info.get("total_comm_bytes", 0)
            comm_mb = comm_bytes / 1e6
            efficiency = (comm_bytes / dense_bytes) if dense_bytes > 0 else 1.0

            # Comm Efficiency Sanity (Diagram Alignment / 30% Hardening)
            # Ensure Top-K is actually reducing traffic (k=0.3 should be ~0.3 efficiency)
            if rnd > 1: # Skip round 1 as base case
                assert efficiency < 0.5, f"❌ Top-K efficiency failure: {efficiency:.4f} (Expected < 0.5)"

            round_log = {
                "round": rnd, "avg_loss": round(curr_loss, 4),
                "bleu4": eval_metrics["bleu4"], "cider": eval_metrics["cider"],
                "n_clients": n_select,
                "comm_mb": round(comm_mb, 4),
                "comm_efficiency": round(efficiency, 4),
                **curr_weights,
                "ablation": getattr(model, "ablation_config", {})
            }
            _server.log_round(rnd, round_log)

            with state.lock:
                state.current_round = rnd
                state.progress = int((rnd / config.training.max_rounds) * 100)
                state.current_metrics = {
                    "avg_loss": round(smoothed_loss, 4),
                    "bleu4": round(smoothed_bleu, 4),
                    "cider": eval_metrics["cider"]
                }
            
            # (Redundant second save removed for I/O efficiency)

        with state.lock:
            state.status = "completed"
            state.is_training = False

    except Exception as e:
        # Surgical Fix: Remove emoji to prevent UnicodeEncodeError on Windows CP1252
        print(f"[ERROR] Training Loop Exception: {e}")
        traceback.print_exc()
        with state.lock:
            state.status = "error"
            state.error = str(e)
            
            state.is_training = False