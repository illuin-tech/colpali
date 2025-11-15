from typing import ClassVar

import torch
from torch import nn
from transformers.models.qwen3_vl_moe import Qwen3VLMoeConfig, Qwen3VLMoeModel


class ColQwen3MoE(Qwen3VLMoeModel):
    """
    ColQwen3-MoE model implementation. This adapts the Qwen3-VL-MoE backbone to the ColPali multi-vector
    retrieval setting.

    Args:
        config (Qwen3VLMoeConfig): Model configuration.
        mask_non_image_embeddings (bool): When ``True`` only image embeddings are preserved.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"

    def __init__(self, config: Qwen3VLMoeConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.config.text_config.hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = super()._checkpoint_conversion_mapping
        return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        attention_mask = kwargs.get("attention_mask")
        has_pixel_values = "pixel_values" in kwargs and kwargs["pixel_values"] is not None

        if has_pixel_values:
            image_grid_thw = kwargs.get("image_grid_thw")
            if image_grid_thw is None:
                raise ValueError("`image_grid_thw` must be provided when `pixel_values` is passed.")

            if not torch.is_tensor(image_grid_thw):
                image_grid_thw = torch.as_tensor(image_grid_thw, device=kwargs["pixel_values"].device)

            offsets = image_grid_thw.prod(dim=1)
            unpadded = [
                pixel_sequence[: int(offset.item())]
                for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)
            ]

            if unpadded:
                kwargs["pixel_values"] = torch.cat(unpadded, dim=0)
            else:
                kwargs["pixel_values"] = None

        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)

        last_hidden_states = (
            super()
            .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
            .last_hidden_state
        )

        proj = self.custom_text_proj(last_hidden_states)
        proj = proj / proj.norm(dim=-1, keepdim=True)

        if attention_mask is not None:
            proj = proj * attention_mask.unsqueeze(-1)

        if has_pixel_values and self.mask_non_image_embeddings and kwargs.get("input_ids") is not None:
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask

        return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size

    @property
    def temporal_patch_size(self) -> int:
        return getattr(self.visual.config, "temporal_patch_size", 1)

