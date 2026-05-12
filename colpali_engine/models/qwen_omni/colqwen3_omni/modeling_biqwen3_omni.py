from typing import ClassVar, Literal

import torch

from .configuration_bidirlm_omni import BidirLMOmniConfig
from .modeling_bidirlm_omni import BidirLMOmniModel


class BiQwen3Omni(BidirLMOmniModel):
    """
    BiQwen3-Omni model wrapper for BidirLM-Omni checkpoints.

    The backbone is the BidirLM-Omni bidirectional encoder: Qwen3-style text and vision towers with an audio tower.
    Representations are pooled to obtain a single vector representation.
    """

    config_class = BidirLMOmniConfig
    main_input_name: ClassVar[str] = "doc_input_ids"
    _checkpoint_conversion_mapping = {
        r"^model\.audio_tower": "audio_tower",
        r"^model\.visual": "visual",
        r"^model\.language_model": "language_model",
        r"^model\.": "",
    }

    def __init__(self, config: BidirLMOmniConfig, **kwargs):
        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", None))
        attn_impl = kwargs.pop("attn_implementation", None)
        use_cache = kwargs.pop("use_cache", None)

        super().__init__(config=config)
        self.padding_side = "left"
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
            key_mapping = getattr(cls, "_checkpoint_conversion_mapping", None)
        return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)

    def forward(
        self,
        pooling_strategy: Literal["cls", "last", "mean"] = "mean",
        *args,
        **kwargs,
    ) -> torch.Tensor:
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

        if pooling_strategy == "cls":
            pooled_output = last_hidden_states[:, 0]
        elif pooling_strategy == "last":
            pooled_output = last_hidden_states[:, -1]
        elif pooling_strategy == "mean":
            mask = kwargs["attention_mask"].unsqueeze(-1)
            pooled_output = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")

        return pooled_output / pooled_output.norm(dim=-1, keepdim=True)

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
