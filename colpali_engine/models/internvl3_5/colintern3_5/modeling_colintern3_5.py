from typing import ClassVar
import torch
from torch import nn
from transformers.models.internvl import InternVLConfig, InternVLModel

class ColIntern3_5(InternVLModel):  # noqa: N801
    """
    ColIntern3.5 model implementation for multi-vector retrieval, based on the InternVL3.5-1B vision-language backbone.
    Applies a linear projection to produce 128-dimensional token-wise embeddings for ColBERT late interaction.

    Args:
        config (InternVLConfig): Configuration of the InternVL3.5 model.
        mask_non_image_embeddings (bool, optional): If True, mask out all non-image token embeddings (e.g., text tokens) during forward pass. Defaults to False.
    """
    main_input_name: ClassVar[str] = "doc_input_ids"

    def __init__(self, config: InternVLConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.config.text_config.hidden_size, self.dim)
        self.padding_side = "right"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Extract torch_dtype from kwargs to ensure proper dtype handling
        torch_dtype = kwargs.get('torch_dtype', None)
        
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = super()._checkpoint_conversion_mapping
            
        model = super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)
        
        # Ensure all parameters are in the correct dtype if specified
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
            # Specifically ensure the custom projection layer is also converted
            if hasattr(model, 'custom_text_proj'):
                model.custom_text_proj = model.custom_text_proj.to(dtype=torch_dtype)
                
        return model

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Remove unsupported HF arguments if present
        kwargs.pop("return_dict", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        # Forward through base InternVL model to obtain hidden states
        outputs = super().forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
        last_hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        # Project hidden states to low-dimensional embeddings
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, seq_length, 128)
        # L2 normalize the embeddings
        proj = proj / proj.norm(dim=-1, keepdim=True)
        # Mask out padding positions
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)
        # Optionally mask out non-image token embeddings (keep only image patch embeddings)
        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask
        return proj

    @property
    def patch_size(self) -> int:
        ps = getattr(self.config.vision_config, "patch_size", 0)
        return ps[0] if isinstance(ps, (list, tuple)) else ps

    @property
    def spatial_merge_size(self) -> int:
        return int(1 / self.config.downsample_ratio)
