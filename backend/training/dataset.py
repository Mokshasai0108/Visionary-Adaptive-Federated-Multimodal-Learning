"""
Dataset and Non-IID data partitioning using Dirichlet distribution.
Supports Flickr30k and MS COCO datasets with configurable train/eval splits.
"""
from __future__ import annotations
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from loguru import logger
import re

def clean_caption(c: str) -> str:
    """Stabilize training text by removing noise and normalizing whitespace."""
    c = str(c).lower().strip()
    c = re.sub(r'\s+', ' ', c) # collapse spaces, tabs, newlines
    return c


class CaptionDataset(Dataset):
    """Generic image-caption dataset from a list of samples."""

    def __init__(
        self,
        samples: list[dict],  # [{"image_path": str, "caption": str}]
        processor,
        tokenizer,
        max_seq_len: int = 64,
        image_size: int = 224,
    ):
        self.samples = samples
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        caption = sample["caption"]

        try:
            image = Image.open(image_path).convert("RGB")
            # Speed Boost: Reduce processor overhead
            image = image.resize((self.image_size, self.image_size)) 
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size), color=128)

        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        encoding = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),
            "caption": caption,
        }


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for local testing without downloading Flickr30k.
    Generates random tensors with dummy captions.
    """

    def __init__(self, size: int = 200, hidden_dim: int = 512, max_seq_len: int = 64):
        self.size = size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.captions = [f"A photo of object {i % 20}" for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return {
            "pixel_values": torch.randn(3, 224, 224),
            "input_ids": torch.randint(0, 1000, (self.max_seq_len,)),
            "attention_mask": torch.ones(self.max_seq_len, dtype=torch.long),
            "labels": torch.randint(0, 1000, (self.max_seq_len,)),
            "caption": self.captions[idx],
        }


def load_flickr30k_json(json_path: str, images_root: str, limit: int | None = None) -> list[dict]:
    """
    Loads samples from captions.json and validates image existence.
    Filters out samples with broken paths or captions that are too short.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        logger.error(f"Captions file not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    if limit is not None:
        raw_samples = raw_samples[:limit]

    valid_samples = []
    logger.info(f"Validating {len(raw_samples)} Flickr30k samples...")
    
    for s in raw_samples:
        # 1. Resolve Path
        img_rel = s["image_path"]
        # If path is already absolute, use it; otherwise join with root
        img_path = Path(img_rel)
        if not img_path.is_absolute():
            img_path = Path(images_root) / img_path.name
        
        # 2. Check existence
        if not img_path.exists():
            continue

        # 3. Filter malformed captions
        caption = clean_caption(s.get("caption", ""))
        if len(caption) < 4:
            continue

        valid_samples.append({
            "image_path": str(img_path),
            "caption": caption
        })

    logger.info(f"Loaded {len(valid_samples)} valid samples from {json_path}")
    return valid_samples


def load_coco_json(json_path: str, limit: int | None = None) -> list[dict]:
    """
    Loads COCO image-caption pairs from a JSON file.
    Accepts both relative and absolute image paths and validates file existence.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        logger.error(f"COCO captions file not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    if limit is not None:
        raw_samples = raw_samples[:limit]

    valid_samples = []
    logger.info(f"Validating {len(raw_samples)} COCO samples...")
    for s in raw_samples:
        img_rel = s.get("image_path", "")
        img_path = Path(img_rel)

        if not img_path.is_absolute() or not img_path.exists():
            # 1. Try relative to JSON (stripping redundant 'data/coco/' if present)
            pure_name = img_path.name
            # Try to find it in train2017 or val2017 subfolders relative to JSON
            candidate = json_path.parent / "train2017" / pure_name
            if not candidate.exists():
                candidate = json_path.parent / "val2017" / pure_name
            
            if candidate.exists():
                img_path = candidate
            else:
                # 2. Try the literal path relative to JSON
                candidate = json_path.parent / img_rel
                if candidate.exists():
                    img_path = candidate
                else:
                    # 3. Try literal path relative to current WD
                    candidate = Path.cwd() / img_rel
                    if candidate.exists():
                        img_path = candidate
        
        if not img_path.exists():
            continue

        caption = clean_caption(s.get("caption", ""))
        if len(caption) < 4:
            continue

        valid_samples.append({
            "image_path": str(img_path),
            "caption": caption,
        })

    logger.info(f"Loaded {len(valid_samples)} valid samples from {json_path}")
    return valid_samples


def dirichlet_partition(
    dataset_size: int,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> list[list[int]]:
    """
    Partition dataset indices across clients using Dirichlet distribution.
    Guarantees at least 1 sample per client.
    """
    rng = np.random.default_rng(seed)

    # Step 1: Sample proportions
    proportions = rng.dirichlet(alpha=np.ones(num_clients) * alpha)

    # Step 2: Convert to counts
    counts = (proportions * dataset_size).astype(int)

    # Step 3: Ensure minimum 1 sample per client
    counts[counts == 0] = 1

    # Step 4: Adjust total to match dataset_size
    diff = dataset_size - counts.sum()
    while diff != 0:
        for i in range(num_clients):
            if diff == 0:
                break
            if diff > 0:
                counts[i] += 1
                diff -= 1
            elif counts[i] > 1:
                counts[i] -= 1
                diff += 1

    # Step 5: Assign indices
    indices = np.arange(dataset_size)
    rng.shuffle(indices)

    partitions = []
    start = 0
    for c in counts:
        partitions.append(indices[start : start + c].tolist())
        start += c

    logger.info(f"Safe Dirichlet partition sizes: {[len(p) for p in partitions]}")
    return partitions


def get_client_dataloader(
    all_samples: list[dict],
    client_indices: list[int],
    processor,
    tokenizer,
    batch_size: int = 32,
    num_workers: int = 2,
    max_seq_len: int = 64,
    shuffle: bool = True,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = None,
) -> DataLoader:
    client_samples = [all_samples[i] for i in client_indices]
    dataset = CaptionDataset(client_samples, processor, tokenizer, max_seq_len)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
