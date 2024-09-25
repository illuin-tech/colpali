from typing import ClassVar

import torch
from torch import nn
from transformers.models.qwen2_vl import Qwen2VLConfig, Qwen2VLForConditionalGeneration, Qwen2VLPreTrainedModel


class ColQwen2(Qwen2VLForConditionalGeneration):
    """
    ColQwen2 model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config=config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.hidden_size, self.dim)
        self.padding_side = "left"
        self.post_init()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)
        # from kwargs, get the pixel_value shape and input_ids shape

        # print(f"input_ids shape: {kwargs['input_ids'].shape}")
        #
        # if "pixel_values" in kwargs:
        #     print(f"pixel_values shape: {kwargs['pixel_values'].shape}")

        # The following code is a hack to make sure the scatter in DDP is done correctly when training on multiple GPUs
        if "pixel_values" in kwargs:
            # compute pixel_values offsets
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            # print(offsets)
            # iterate over the pixel_values and keep only their offset
            # new_pixel_values = []
            # for pv, o in zip(kwargs["pixel_values"], offsets):
            #     new_pixel_values.append(pv[:o])

            kwargs["pixel_values"] = torch.cat([pv[:o] for pv, o in zip(kwargs["pixel_values"], offsets)], dim=0)
            print(kwargs["pixel_values"].shape)

        inputs = self.prepare_inputs_for_generation(*args, **kwargs, use_cache=False)
        # print(inputs.keys())
        outputs = super().forward(**inputs, output_hidden_states=True)  # (batch_size, sequence_length, hidden_size)

        # outputs = super().forward(*args, **kwargs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)
        return proj
