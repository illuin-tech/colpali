from typing import ClassVar, Literal

import torch
from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLModel


class BiQwen3(Qwen3VLModel):
    """
    BiQwen3 implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    Representations are pooled to obtain a single vector representation. Based on the Qwen3-VL backbone.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"
    _checkpoint_conversion_mapping = {
        r"^model\.visual": "visual",
        r"^model\.language_model": "language_model",
        r"^model\.": "",
    }

    def __init__(self, config: Qwen3VLConfig, **kwargs):
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
        pooling_strategy: Literal["cls", "last", "mean"] = "last",
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for BiQwen3 model.

        Args:
            pooling_strategy: The strategy to use for pooling the hidden states.
            *args: Variable length argument list.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Dense embeddings (batch_size, hidden_size).
        """
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"].prod(dim=1).tolist()
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )
        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)

        last_hidden_states = (
            super()
            .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
            .last_hidden_state
        )  # (batch_size, sequence_length, hidden_size)

        if pooling_strategy == "cls":
            pooled = last_hidden_states[:, 0]
        elif pooling_strategy == "last":
            pooled = last_hidden_states[:, -1]
        elif pooling_strategy == "mean":
            mask = kwargs["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")

        return pooled / pooled.norm(dim=-1, keepdim=True)

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
