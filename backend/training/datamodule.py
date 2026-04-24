"""
DataModule: handles Flickr30k loading (real) or synthetic fallback.
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from loguru import logger

from .dataset import CaptionDataset, SyntheticDataset, dirichlet_partition


class AFSPLDataModule:
    """
    Manages dataset creation, non-IID partitioning, and dataloader construction.

    Real data (Flickr30k / MS-COCO):
      Expects data_dir/flickr30k/images/ and data_dir/flickr30k/captions.json
      captions.json format: [{"image_path": "...", "caption": "..."}]

    Falls back to SyntheticDataset if data_dir is unavailable.
    """

    def __init__(self, config, processor, tokenizer):
        self.config = config
        self.processor = processor
        self.tokenizer = tokenizer
        self.train_samples = None
        self.eval_samples = None
        self.partitions: list[list[int]] = []

    def setup(self) -> None:
        cfg = self.config
        data_dir = Path(cfg.data.data_dir)
        flickr_captions = data_dir / "flickr30k" / "captions.json"

        if flickr_captions.exists():
            logger.info("Loading Flickr30k from disk.")
            with open(flickr_captions) as f:
                all_samples = json.load(f)
            n_train = int(len(all_samples) * cfg.data.train_split)
            self.train_samples = all_samples[:n_train]
            self.eval_samples = all_samples[n_train:]
            self._use_synthetic = False
        else:
            logger.warning("Flickr30k not found — using SyntheticDataset for development.")
            self._use_synthetic = True
            self._synthetic_train = SyntheticDataset(size=1000)
            self._synthetic_eval = SyntheticDataset(size=100)

        # Partition for federated training
        train_size = len(self.train_samples) if not self._use_synthetic else len(self._synthetic_train)
        self.partitions = dirichlet_partition(
            train_size,
            cfg.data.num_clients,
            cfg.data.dirichlet_alpha,
            cfg.project.seed,
        )

    def get_client_loader(self, client_id: int, batch_size: int, num_workers: int) -> DataLoader:
        partition = self.partitions[client_id]
        if self._use_synthetic:
            ds = Subset(self._synthetic_train, partition)
        else:
            samples = [self.train_samples[i] for i in partition]
            ds = CaptionDataset(samples, self.processor, self.tokenizer,
                                self.config.model.max_seq_len, self.config.data.image_size)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def get_eval_loader(self, batch_size: int, num_workers: int) -> DataLoader:
        if self._use_synthetic:
            ds = self._synthetic_eval
        else:
            ds = CaptionDataset(self.eval_samples, self.processor, self.tokenizer,
                                self.config.model.max_seq_len, self.config.data.image_size)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def partition_stats(self) -> dict:
        return {
            "num_clients": len(self.partitions),
            "sizes": [len(p) for p in self.partitions],
            "min_size": min(len(p) for p in self.partitions),
            "max_size": max(len(p) for p in self.partitions),
        }
