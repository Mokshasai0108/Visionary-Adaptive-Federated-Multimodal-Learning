"""Frozen CLIP ViT-B/32 vision + text encoder."""
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from loguru import logger


class CLIPVisionEncoder(nn.Module):
    """Frozen CLIP image encoder. Returns (B, hidden_dim) embeddings."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        super().__init__()
        self.device = device
        logger.info(f"Loading CLIP model: {model_name}")
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.hidden_dim = self.clip.config.projection_dim  # 512
        # Freeze ALL CLIP parameters
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()
        logger.info("CLIP encoder loaded and frozen.")

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224) preprocessed images
        Returns:
            image_embeds: (B, hidden_dim) L2-normalized
        """
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output           # (B, vision_hidden)
        projected = self.clip.visual_projection(pooled)  # (B, 512)
        projected = projected / projected.norm(dim=-1, keepdim=True)
        return projected

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            text_embeds: (B, hidden_dim) L2-normalized
        """
        text_outputs = self.clip.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooled = text_outputs.pooler_output
        projected = self.clip.text_projection(pooled)
        projected = projected / projected.norm(dim=-1, keepdim=True)
        return projected

    def preprocess_images(self, images) -> torch.Tensor:
        """PIL images → pixel_values tensor."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        return inputs["pixel_values"].to(self.device)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encode_image(pixel_values)
