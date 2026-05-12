from typing import ClassVar

import torch
from torch import nn

from .configuration_bidirlm_omni import BidirLMOmniConfig
from .modeling_bidirlm_omni import BidirLMOmniModel


class ColQwen3Omni(BidirLMOmniModel):
    """
    ColQwen3-Omni model wrapper for BidirLM-Omni checkpoints.

    The backbone is the BidirLM-Omni bidirectional encoder: Qwen3-style text and vision towers with an audio tower.
    This class adds the Col-style projection head used for multi-vector retrieval.
    """

    config_class = BidirLMOmniConfig
    main_input_name: ClassVar[str] = "doc_input_ids"
    _checkpoint_conversion_mapping = {
        r"^base_model\.model\.custom_text_proj": "custom_text_proj",
        r"^model\.audio_tower": "audio_tower",
        r"^model\.visual": "visual",
        r"^model\.language_model": "language_model",
        r"^model\.": "",
    }

    def __init__(
        self,
        config: BidirLMOmniConfig,
        mask_non_image_embeddings: bool = False,
        **kwargs,
    ):
        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", None))
        attn_impl = kwargs.pop("attn_implementation", None)
        use_cache = kwargs.pop("use_cache", None)

        super().__init__(config=config)

        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.config, "text_config"):
            hidden_size = getattr(self.config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(f"Unable to determine text hidden size for {type(self.config).__name__}.")

        self.dim = 128
        self.custom_text_proj = nn.Linear(hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()

        if dtype is not None:
            self.to(dtype=dtype)
        if use_cache is not None:
            self.config.use_cache = use_cache
        if attn_impl is not None and hasattr(self, "set_attn_implementation"):
            self.set_attn_implementation(attn_impl)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = dict(getattr(super(), "_checkpoint_conversion_mapping", {}))
            key_mapping.update(cls._checkpoint_conversion_mapping)
        return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if "pixel_values" in kwargs and kwargs["pixel_values"].ndim == 3:
            offsets = kwargs["image_grid_thw"].prod(dim=1)
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )

        if "pixel_values_videos" in kwargs and kwargs["pixel_values_videos"].ndim == 3:
            offsets = kwargs["video_grid_thw"].prod(dim=1)
            kwargs["pixel_values_videos"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values_videos"], offsets)],
                dim=0,
            )

        model_dtype = next(self.parameters()).dtype
        for key in ("pixel_values", "pixel_values_videos", "input_features"):
            if key in kwargs and kwargs[key].is_floating_point() and kwargs[key].dtype != model_dtype:
                kwargs[key] = kwargs[key].to(dtype=model_dtype)

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
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask

        return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
