from typing import ClassVar

import torch
from torch import nn

from .configuration_florence2 import Florence2Config
from .modeling_florence2 import Florence2VisionLanguageModel


class ColFlor(Florence2VisionLanguageModel):
    """
    ColFlor model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: Florence2Config):
        super().__init__(config=config)

        self.dim = 128
        self.custom_text_proj = nn.Linear(self.config.text_config.d_model, self.dim)

        self.padding_side = "right"
        self.post_init()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)

        # Create Full Attention Mask that includes both the image and text
        full_attention_mask = kwargs['attention_mask']
        # make sure pixel_values are in the same dtype as the model
        if 'pixel_values' in kwargs:
            full_attention_mask = kwargs['full_attention_mask'].type(self.dtype)
            del kwargs['full_attention_mask']
            kwargs['pixel_values'] = kwargs['pixel_values'].type(self.dtype)

        outputs = super().forward(*args, **kwargs)

        last_hidden_states = outputs['encoder_last_hidden_state']  # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)
        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        proj = proj * full_attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)
        return proj
