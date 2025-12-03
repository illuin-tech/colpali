"""
ColGemma3 Model - Implementation for late interaction retrieval.

This module implements ColGemma3 for late interaction retrieval, properly loading
pretrained weights from Gemma3Model.

Key features:
    - Proper weight loading (no random initialization of base model)
    - Device/dtype matching for custom projection layer
    - Multi-vector embeddings for MaxSim scoring
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers.models.gemma3 import Gemma3Config, Gemma3Model


class ColGemma3(nn.Module):
    """
    ColGemma3 model for late interaction retrieval.

    This model extends Gemma3 to produce multi-vector embeddings suitable for
    efficient document retrieval. Each input (image or text) is encoded into a
    sequence of contextualized vectors, which can be compared using MaxSim scoring.

    Args:
        base_model (Gemma3Model): Pretrained base model
        mask_non_image_embeddings (bool, optional): If True, masks all embeddings
            except image tokens during forward pass. Defaults to False.

    Example:
        >>> model = ColGemma3.from_pretrained("google/gemma-3-4b-it")
        >>> embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
        >>> print(embeddings.shape)  # (batch_size, seq_len, 128)
    """

    def __init__(
        self,
        base_model: Gemma3Model,
        mask_non_image_embeddings: bool = False,
    ):
        super().__init__()

        # Store the base model directly
        self.model = base_model
        self.config = base_model.config

        # Projection layer for late interaction (reduce to 128 dims)
        self.dim = 128
        hidden_size = (
            self.config.text_config.hidden_size
            if hasattr(self.config, "text_config")
            else self.config.hidden_size
        )
        self.custom_text_proj = nn.Linear(hidden_size, self.dim)

        # Move custom projection layer to the same device/dtype as base model
        base_device = next(self.model.parameters()).device
        base_dtype = next(self.model.parameters()).dtype
        self.custom_text_proj = self.custom_text_proj.to(
            device=base_device, dtype=base_dtype
        )

        # Configuration
        self.mask_non_image_embeddings = mask_non_image_embeddings

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = None,
        model_name_or_path: str = None,
        **kwargs
    ):
        """
        Load pretrained ColGemma3 model.

        This method can load either:
        1. A base Gemma3Model (custom_text_proj randomly initialized)
        2. A saved ColGemma3 model (with trained custom_text_proj weights)

        Args:
            pretrained_model_name_or_path (str): HuggingFace model ID or local path (compatibility)
            model_name_or_path (str): HuggingFace model ID or local path (alternative)
            **kwargs: Additional arguments for model loading (dtype, device_map, etc.)

        Returns:
            ColGemma3: Initialized model with pretrained weights

        Note:
            If loading a base Gemma3 model, the custom_text_proj layer will be randomly initialized.
            If loading a saved ColGemma3 model, all weights including custom_text_proj will be loaded.
        """
        import os
        import warnings

        # Support both parameter names for compatibility
        model_path = pretrained_model_name_or_path or model_name_or_path
        if model_path is None:
            raise ValueError("Must provide either pretrained_model_name_or_path or model_name_or_path")

        # Check if this is a saved ColGemma3 model by looking for custom_text_proj weights
        is_colgemma3_checkpoint = False
        custom_proj_weights = {}

        # Function to check and load custom_text_proj from safetensors file
        def check_and_load_custom_proj(safetensors_path):
            nonlocal is_colgemma3_checkpoint, custom_proj_weights
            try:
                from safetensors import safe_open
                with safe_open(safetensors_path, framework="pt") as f:
                    keys = list(f.keys())

                    # Check for custom_text_proj weights
                    custom_keys = [k for k in keys if "custom_text_proj" in k]

                    if custom_keys:
                        is_colgemma3_checkpoint = True

                        # Pre-load custom_text_proj weights
                        for key in custom_keys:
                            param_name = key.replace("custom_text_proj.", "")
                            custom_proj_weights[param_name] = f.get_tensor(key)
                    return True
            except Exception as e:
                warnings.warn(f"Could not check checkpoint format: {e}")
                return False

        # Check for safetensors file in local directory
        if os.path.isdir(model_path):
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                check_and_load_custom_proj(safetensors_path)
        else:
            # For HuggingFace Hub models, we need to download/access the file
            try:
                from huggingface_hub import snapshot_download
                cache_dir = snapshot_download(
                    repo_id=model_path,
                    allow_patterns=["model.safetensors"],
                    **{k: v for k, v in kwargs.items() if k in ['token', 'revision']}
                )
                safetensors_path = os.path.join(cache_dir, "model.safetensors")
                if os.path.exists(safetensors_path):
                    check_and_load_custom_proj(safetensors_path)
            except Exception as e:
                warnings.warn(f"Could not check HF model for custom_text_proj: {e}")

        # Load the base model
        base_model = Gemma3Model.from_pretrained(
            model_path, **kwargs
        )

        # Create ColGemma3 with the loaded base model
        model = cls(base_model=base_model)

        # If this is a ColGemma3 checkpoint, load the custom_text_proj weights
        if is_colgemma3_checkpoint and custom_proj_weights:
            try:
                model.custom_text_proj.load_state_dict(custom_proj_weights, strict=True)
            except Exception as e:
                warnings.warn(f"Could not load custom_text_proj weights: {e}")

        return model

    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device

    @property
    def dtype(self):
        """Get the dtype of the model."""
        return next(self.parameters()).dtype

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass - generates multi-vector embeddings.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            pixel_values: Image pixels (batch_size, channels, height, width)
            **kwargs: Additional arguments

        Returns:
            torch.Tensor: Multi-vector embeddings (batch_size, seq_len, 128)
        """
        # Clean kwargs
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        kwargs.pop("return_dict", None)

        # Ensure correct dtype for images
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.dtype)

        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        # Get last hidden states: (batch_size, seq_len, hidden_size)
        last_hidden_states = outputs.last_hidden_state

        # Project to lower dimension: (batch_size, seq_len, 128)
        proj = self.custom_text_proj(last_hidden_states)

        # L2 normalize each token embedding
        proj = proj / proj.norm(dim=-1, keepdim=True)

        # Apply attention mask (zero out padding tokens)
        if attention_mask is not None:
            proj = proj * attention_mask.unsqueeze(-1)

        # Optionally mask non-image tokens
        if pixel_values is not None and self.mask_non_image_embeddings:
            if hasattr(self.config, "image_token_id"):
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
                proj = proj * image_mask

        return proj

    def get_input_embeddings(self):
        """Get input embeddings layer from the base model."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set input embeddings layer in the base model."""
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        """Get output embeddings layer from the base model."""
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings layer in the base model."""
        self.model.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        """
        Resize token embeddings to accommodate new vocabulary size.

        Args:
            new_num_tokens (int, optional): New vocabulary size.
            pad_to_multiple_of (int, optional): Pad to multiple of this value.

        Returns:
            nn.Embedding: Resized embedding layer.
        """
        model_embeds = self.model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )

        # Update vocab size in config
        if hasattr(self.config, "text_config") and hasattr(
            self.config.text_config, "vocab_size"
        ):
            self.config.text_config.vocab_size = model_embeds.num_embeddings
        if hasattr(self.model.config, "text_config") and hasattr(
            self.model.config.text_config, "vocab_size"
        ):
            self.model.config.text_config.vocab_size = model_embeds.num_embeddings

        return model_embeds

    @property
    def patch_size(self) -> int:
        """
        Get vision patch size.

        Returns:
            int: Patch size (default 14 for Gemma3).
        """
        if hasattr(self.config, "vision_config") and hasattr(
            self.config.vision_config, "patch_size"
        ):
            return self.config.vision_config.patch_size
        return 14  # Default for Gemma3
