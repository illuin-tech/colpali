from typing import ClassVar, Literal

import torch
from transformers.models.qwen2_vl import Qwen2VLConfig, Qwen2VLModel


class BiQwen2(Qwen2VLModel):
    """
    BiQwen2 is an implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    Representations are pooled to obtain a single vector representation. Based on the Qwen2.5-VL backbone.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config=config)
        self.padding_side = "left"
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = super()._checkpoint_conversion_mapping
        return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)

    def forward(
        self,
        pooling_strategy: Literal["cls", "last", "mean"] = "last",
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for BiQwen2.5 model.

        Args:
            pooling_strategy: The strategy to use for pooling the hidden states.
            *args: Variable length argument list.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Dense embeddings (batch_size, hidden_size).
        """
        # Handle the custom "pixel_values" input obtained with `ColQwen2Processor` through unpadding
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]  # (batch_size,)
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

        # Get CLS token embedding, last token, or mean pool over sequence
        if pooling_strategy == "cls":
            # Use CLS token (first token) embedding
            pooled_output = last_hidden_states[:, 0]  # (batch_size, hidden_size)
        elif pooling_strategy == "last":
            # use last token since we are left padding
            pooled_output = last_hidden_states[:, -1]  # (batch_size, hidden_size)
        elif pooling_strategy == "mean":
            # Mean pooling over sequence length
            mask = kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, 1)
            pooled_output = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1)  # (batch_size, hidden_size)
        else:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")

        # L2 normalization
        pooled_output = pooled_output / pooled_output.norm(dim=-1, keepdim=True)
        return pooled_output
